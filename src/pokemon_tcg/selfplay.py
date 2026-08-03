"""Stage-3 (SELFPLAY) rollout collection: parallel self-play games with a
recording, sampling policy — rl_pipeline_v1.md §3.1.

Architecture note (deviation from the recorded PufferLib decision, stated
loudly): parallelism is GAME-LEVEL via stdlib multiprocessing, not step-level
via PufferLib's emulation layer. The substance of the 2026-08-01 decision —
multiprocessing, never Ray — is kept. The wrapper layer is not, because cabt
is driven callback-style (`env.run([agent0, agent1])`, the same tested path
scripts/benchmark_agents.py uses), one game is ~0.9 s with ~3 ms policy calls
(env-bound, not inference-bound), and forcing a two-player turn-based env
into a Gym step() mold buys batched inference we don't need. If step-level
vectorization ever becomes the bottleneck, PufferLib is the fallback.

Everything stays in RAM (disk budget: rl_pipeline_v1.md standing rules).
A rollout row is exactly one training example in the Stage-1/2 sense: one
(features, action) pick, including DECLINE slots and autoregressive
multi-select unrolls, so the PPO learner consumes the same tensor layout
ILDataset produces.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import config
from .il_dataset import MAX_OPTIONS, encode_observation
from .il_model import PTCGImitationPolicy

_CG_DIR = config.PROJECT_ROOT / "data" / "external" / "cg-lib"
if (_CG_DIR / "cg" / "api.py").exists() and str(_CG_DIR) not in sys.path:
    sys.path.insert(0, str(_CG_DIR))

DECK_PATH = config.PROJECT_ROOT / "agents" / "il_agent" / "deck.csv"

# Feature keys stored per rollout row (everything the model forward needs).
FEATURE_KEYS = (
    "slot_card_id", "slot_scalar", "global_scalar", "select_type",
    "select_context", "opt_type", "opt_attack", "opt_special", "opt_scalar",
    "opt_ref_card_id", "opt_ref_scalar", "opt_mask",
)


def load_deck() -> list[int]:
    return [int(x) for x in DECK_PATH.read_text().split() if x.strip()][:60]


def as_env_agent(fn):
    """Wrap a callable as a plain single-parameter function.

    kaggle_environments inspects `__code__.co_argcount` to decide what to
    pass: a 1-arg function gets `observation` alone, but a BOUND METHOD
    counts `self` and gets called with (observation, configuration) ->
    TypeError -> instant ERROR/loss. Every callable handed to env.run() goes
    through this wrapper; never pass a bound method directly.
    """

    def agent_fn(obs):
        return fn(obs)

    return agent_fn


def _safe_choice(select) -> list[int]:
    n = len(select["option"])
    lo = max(select.get("minCount") or 0, 0)
    hi = min(select["maxCount"] if select.get("maxCount") is not None else n, n)
    k = lo if lo > 0 else min(1, hi)
    return list(range(min(k, hi)))


class SamplingPolicy:
    """`agent(obs) -> list[int]` over a PTCGImitationPolicy that SAMPLES from
    the masked option distribution (temperature 1.0 = the policy itself; PPO
    importance ratios assume 1.0) and optionally records every pick.

    Mirrors agents/il_agent/agent_core.py's decision loop (decline slot,
    autoregressive multi-select with re-masking, never-crash fallback); the
    fallback path is never recorded — fallback decisions are not policy
    samples and must not enter the PPO buffer. `fallbacks` counts them.
    """

    def __init__(self, ckpt_dir: str | Path, temperature: float = 1.0,
                 record: bool = False, deck: list[int] | None = None) -> None:
        self.model = PTCGImitationPolicy.from_pretrained(str(ckpt_dir)).eval()
        self.temperature = temperature
        self.record = record
        self.deck = deck or load_deck()
        self.rows: list[dict] = []
        self.fallbacks = 0
        self.decisions = 0

    def reset(self) -> None:
        self.rows = []
        self.fallbacks = 0
        self.decisions = 0

    def _sample_pick(self, obs: dict, exclude: frozenset, add_decline: bool):
        feats = encode_observation(obs, exclude=exclude)
        if feats is None:
            return None
        n_real = feats.pop("n_real_options")
        batch = {k: v.unsqueeze(0) for k, v in feats.items()}
        with torch.no_grad():
            logits = self.model(**batch)["logits"][0]
        upto = n_real + (1 if add_decline else 0)
        legal = logits[:upto]
        logp_all = torch.log_softmax(legal / self.temperature, dim=-1)
        choice = int(torch.multinomial(logp_all.exp(), 1).item())
        if self.record:
            row = {k: feats[k].numpy() for k in FEATURE_KEYS}
            row["action"] = choice
            row["logprob"] = float(logp_all[choice].item())
            row["n_legal"] = int(upto)
            self.rows.append(row)
        return choice, n_real

    def agent(self, obs: dict) -> list[int]:
        select = obs.get("select")
        try:
            if select is None:
                return self.deck
            n_opts = len(select.get("option") or [])
            if n_opts == 0:
                return []
            if n_opts > MAX_OPTIONS:
                self.fallbacks += 1
                return _safe_choice(select)
            self.decisions += 1
            max_count = select["maxCount"] if select.get("maxCount") is not None else 1
            min_count = select.get("minCount") or 0
            add_decline = min_count == 0

            picked: list[int] = []
            excluded: set[int] = set()
            for _ in range(max(max_count, 1)):
                out = self._sample_pick(obs, frozenset(excluded), add_decline)
                if out is None:
                    self.fallbacks += 1
                    return _safe_choice(select)
                choice, n_real = out
                if add_decline and choice == n_real:
                    break  # sampled the DECLINE slot: stop picking
                if choice >= n_opts or choice in excluded:
                    self.fallbacks += 1
                    return _safe_choice(select)
                picked.append(choice)
                excluded.add(choice)
                if len(picked) >= max_count:
                    break
            if len(picked) < min_count:
                self.fallbacks += 1
                return _safe_choice(select)
            return picked
        except Exception:
            self.fallbacks += 1
            try:
                return _safe_choice(select) if select else []
            except Exception:
                return []


# ---------------------------------------------------------------------------
# Worker-process side. Globals are per-process caches (spawn context): each
# worker loads the env module, the learner actor, and any opponent it is
# asked to play, exactly once.
# ---------------------------------------------------------------------------
_W: dict = {}


def worker_init(learner_ckpt: str, temperature: float) -> None:
    torch.set_num_threads(1)
    from kaggle_environments import make as kaggle_make  # slow import, once per worker

    _W["make"] = kaggle_make
    _W["learner_ckpt"] = learner_ckpt
    _W["learner"] = SamplingPolicy(learner_ckpt, temperature=temperature, record=True)
    _W["opponents"] = {}


def _get_opponent(spec: tuple[str, str]):
    """spec = ("ckpt", <dir>) for a policy opponent (sampling, not recorded),
    or ("module", <benchmark agent name>) for a public-pool/heuristic agent."""
    key = spec
    if key in _W["opponents"]:
        return _W["opponents"][key]
    kind, ref = spec
    if kind == "ckpt":
        # "__mirror__" = current learner weights. A SEPARATE non-recording
        # instance, never learner.agent itself -- the learner's buffer must
        # only contain its own seat's decisions.
        if ref == "__mirror__":
            ref = _W["learner_ckpt"]
        opp = as_env_agent(SamplingPolicy(ref, temperature=1.0, record=False).agent)
    elif kind == "module":
        sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
        import benchmark_agents

        opp = benchmark_agents.load_agent(ref)
    else:
        raise ValueError(f"unknown opponent spec {spec}")
    _W["opponents"][key] = opp
    return opp


def play_one_game(task: tuple[int, tuple[str, str], int]) -> dict:
    """Worker entry: play one game, return the learner seat's rollout.

    task = (game_idx, opponent_spec, learner_seat). Returns numpy-backed rows
    (cheap to pickle) + outcome remapped to {0, 0.5, 1} ({} rows on crash).
    """
    game_idx, opp_spec, learner_seat = task
    learner: SamplingPolicy = _W["learner"]
    learner.reset()
    opponent = _get_opponent(opp_spec)

    agents = [None, None]
    agents[learner_seat] = as_env_agent(learner.agent)
    agents[1 - learner_seat] = opponent

    t0 = time.time()
    env = _W["make"]("cabt")
    trace = env.run(agents)
    final = trace[-1]
    r = [final[i]["reward"] for i in range(2)]
    # None reward = that seat crashed/errored out = loss for that seat
    # (same convention as benchmark_agents.play_match / winning_agent).
    scored = [float("-inf") if x is None else x for x in r]
    mine, theirs = scored[learner_seat], scored[1 - learner_seat]
    outcome = 0.5 if mine == theirs else (1.0 if mine > theirs else 0.0)

    return {
        "game_idx": game_idx,
        "opponent": opp_spec,
        "learner_seat": learner_seat,
        "outcome": outcome,
        "rows": learner.rows,
        "decisions": learner.decisions,
        "fallbacks": learner.fallbacks,
        "seconds": time.time() - t0,
    }


def sample_league(rng: random.Random, league_ckpts: list[str],
                  pool_agents: list[str], mix=(0.5, 0.3, 0.2)) -> tuple[str, str]:
    """League opponent draw per rl_pipeline_v1.md §3.1: ~mirror / past ckpts /
    public pool. Falls back to mirror when a bucket is empty."""
    u = rng.random()
    if u < mix[0] or (not league_ckpts and not pool_agents):
        return ("ckpt", "__mirror__")
    if u < mix[0] + mix[1] and league_ckpts:
        return ("ckpt", rng.choice(league_ckpts))
    if pool_agents:
        return ("module", rng.choice(pool_agents))
    return ("ckpt", "__mirror__")
