"""PufferLib 3.0 integration: cabt as a single-agent Gymnasium env.

Runs in the PPO side venv (.venv-ppo: py3.12, numpy<2, pufferlib==3.0.0) —
PufferLib 4.0 dropped Python-env support entirely and 3.0's numpy<2 pin can't
meet the main venv's py3.13, hence the split (user decision 2026-08-03).

Control inversion: kaggle_environments' cabt is callback-driven, but the
statuses on env.state tell whose turn it is, so we drive it step-wise:
opponent turns and the deck-selection step are resolved inside this wrapper,
and only the LEARNER's real decisions surface as RL steps. Multi-select
decisions (maxCount > 1) surface as one RL step per pick with prior picks
re-masked — the same autoregressive unroll as il_dataset/selfplay — and the
accumulated list is submitted to the engine when the policy declines or hits
maxCount.

Observation = one flat float32 Box: the 12 encode_observation() tensors
packed at fixed offsets (OBS_LAYOUT). The policy splits them back out and
recasts ids to long — float32 is exact for every id in vocab (< 2^24).
Action = Discrete(MAX_OPTIONS); illegal picks are structurally prevented by
the policy masking from the packed opt_mask, and the env still guards by
falling back to the lowest legal pick (never crash, counted in infos).

Reward: terminal only, {0, 0.5, 1} (win/draw/loss remap; never -1).
"""

from __future__ import annotations

import glob
import os
import random
import sys
import time
from pathlib import Path

import gymnasium
import numpy as np
import torch

from . import config
from .il_dataset import MAX_OPTIONS, encode_observation
from .deck_pool import DeckPool, deck_override_agent
from .selfplay import SamplingPolicy, _safe_choice, as_env_agent, load_deck

_CG_DIR = config.PROJECT_ROOT / "data" / "external" / "cg-lib"
if (_CG_DIR / "cg" / "api.py").exists() and str(_CG_DIR) not in sys.path:
    sys.path.insert(0, str(_CG_DIR))

# (key, shape, is_integer) — must match encode_observation()'s output exactly.
# Probed at import time from a dummy encode is tempting but circular; these are
# the shapes asserted by tests/test_puffer_env.py against a real observation.
from .il_dataset import N_GLOBAL_SCALARS, N_OPT_SCALARS, N_REF_SCALARS, N_SLOT_SCALARS, N_STATE_SLOTS  # noqa: E402

OBS_LAYOUT = (
    ("slot_card_id", (N_STATE_SLOTS,), True),
    ("slot_scalar", (N_STATE_SLOTS, N_SLOT_SCALARS), False),
    ("global_scalar", (N_GLOBAL_SCALARS,), False),
    ("select_type", (), True),
    ("select_context", (), True),
    ("opt_type", (MAX_OPTIONS,), True),
    ("opt_attack", (MAX_OPTIONS,), True),
    ("opt_special", (MAX_OPTIONS,), True),
    ("opt_scalar", (MAX_OPTIONS, N_OPT_SCALARS), False),
    ("opt_ref_card_id", (MAX_OPTIONS, 2), True),
    # NOTE: encode_observation emits the ref scalars already flattened per
    # option: (MAX_OPTIONS, 2 * N_REF_SCALARS) = (48, 6), NOT (48, 2, 3) --
    # verified against a real observation 2026-08-03 after a shape mismatch.
    ("opt_ref_scalar", (MAX_OPTIONS, 2 * N_REF_SCALARS), False),
    ("opt_mask", (MAX_OPTIONS,), True),
)
_SIZES = [int(np.prod(s)) if s else 1 for _, s, _ in OBS_LAYOUT]
_OFFSETS = np.concatenate([[0], np.cumsum(_SIZES)])
OBS_SIZE = int(_OFFSETS[-1])


_layout_checked = False


def pack_obs(feats: dict) -> np.ndarray:
    global _layout_checked
    if not _layout_checked:
        # One-time guard: a silent OBS_LAYOUT/encoder drift packs fine (flat
        # copy) but unpacks into garbage shapes at the policy -- fail loudly
        # here instead of as a matmul error deep inside the model.
        for key, shape, _ in OBS_LAYOUT:
            actual = tuple(feats[key].shape)
            assert actual == shape, (
                f"OBS_LAYOUT drift: {key} is {actual}, layout says {shape}")
        _layout_checked = True
    out = np.empty(OBS_SIZE, dtype=np.float32)
    for (key, _, _), off, size in zip(OBS_LAYOUT, _OFFSETS, _SIZES, strict=False):
        out[off:off + size] = feats[key].numpy().astype(np.float32).reshape(-1)
    return out


def split_obs(obs: torch.Tensor) -> dict:
    """[B, OBS_SIZE] float32 -> the model's feature dict (ids recast to long)."""
    b = obs.shape[0]
    feats = {}
    for (key, shape, is_int), off, size in zip(OBS_LAYOUT, _OFFSETS, _SIZES, strict=False):
        t = obs[:, off:off + size].reshape(b, *shape) if shape else obs[:, off]
        if key == "opt_mask":
            feats[key] = t > 0.5
        elif is_int:
            feats[key] = t.long()
        else:
            feats[key] = t
    return feats


class LatestCheckpointOpponent:
    """Sampling-policy opponent that hot-reloads the newest checkpoint under a
    directory every `refresh_s` seconds — this is what keeps mirror self-play
    genuinely on-policy while pufferl owns the learner weights."""

    def __init__(self, ckpt_root: str, fallback: str, refresh_s: float = 120.0) -> None:
        self.ckpt_root = ckpt_root
        self.fallback = fallback
        self.refresh_s = refresh_s
        self._loaded_from: str | None = None
        self._next_check = 0.0
        self._policy: SamplingPolicy | None = None

    def _newest(self) -> str:
        cands = sorted(glob.glob(os.path.join(self.ckpt_root, "**", "config.json"),
                                 recursive=True), key=os.path.getmtime)
        return str(Path(cands[-1]).parent) if cands else self.fallback

    def agent(self, obs: dict) -> list[int]:
        now = time.time()
        if self._policy is None or now >= self._next_check:
            self._next_check = now + self.refresh_s
            src = self._newest()
            if src != self._loaded_from:
                try:
                    self._policy = SamplingPolicy(src, temperature=1.0, record=False)
                    self._loaded_from = src
                except Exception:
                    if self._policy is None:
                        self._policy = SamplingPolicy(self.fallback, temperature=1.0, record=False)
                        self._loaded_from = self.fallback
        return self._policy.agent(obs)


class PTCGGym(gymnasium.Env):
    """Single-agent view of one cabt game vs a lineage-sampled opponent.

    OPPONENTS ARE LINEAGE-ONLY: the live mirror (hot-reloaded learner weights)
    and our own frozen checkpoints. Rule-based baselines and other competitors'
    submissions ("externals") are deliberately NOT training opponents — keeping
    them unseen is what makes the local evaluation pool a real out-of-sample
    filter. The diversity axis here is the DECK, not the opponent.
    """

    metadata = {"render_modes": []}

    def __init__(self, league: list[tuple[str, str]] | None = None,
                 mirror_root: str | None = None,
                 anchor_ckpt: str | None = None,
                 mix: tuple[float, float] = (0.625, 0.375),
                 opp_hold: int = 4,
                 alternate_seats: bool = False,
                 deck_pool: DeckPool | None = None,
                 mirror_deck: bool = False,
                 seed: int = 0) -> None:
        super().__init__()
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(MAX_OPTIONS)
        self.rng = random.Random(seed)
        # Deck pool: None keeps the historical single hardcoded deck, so every
        # pre-pool run reproduces bit-for-bit. With a pool, BOTH seats' decks
        # are redrawn every episode -- independently by default, or as a pair
        # under mirror_deck (the same-deck-both-seats control). Without the
        # opponent-side draw a checkpoint opponent keeps serving
        # selfplay.load_deck()'s single list, which is how three self-play
        # generations trained on one mirror matchup.
        self.deck_pool = deck_pool
        self.mirror_deck = mirror_deck
        self.deck = load_deck() if deck_pool is None else deck_pool.entries[0][1]
        self.deck_name = "load_deck" if deck_pool is None else deck_pool.names[0]
        self.opp_deck = self.deck
        self.opp_deck_name = self.deck_name
        anchor = anchor_ckpt or str(config.MODELS_DIR / "s2" / "e1_seed43")
        self.anchor = anchor
        self.mix = mix
        self.league = league or [("ckpt", str(config.MODELS_DIR / "il_agent"))]
        # Opponent persistence: keep the drawn opponent for `opp_hold`
        # consecutive episodes so per-opponent win-rate estimates harvested
        # from rollouts are runs, not single samples. Purely a variance knob
        # now -- it used to feed PFSP pool weighting, which was deleted with
        # the public-pool bucket. The DECK still redraws every episode.
        self.opp_hold = max(1, int(opp_hold))
        self._opp_left = 0
        self._opp_slug = "mirror"
        self._mirror = (LatestCheckpointOpponent(mirror_root, anchor)
                        if mirror_root else None)
        # Strict seat alternation (exploiter runs): a policy trained vs a
        # FROZEN opponent must not specialize in first-player-only exploits,
        # so flip seats every episode instead of sampling 50/50. First seat
        # comes from the per-env rng so parallel workers desynchronize.
        self.alternate_seats = alternate_seats
        self._next_seat = self.rng.randint(0, 1)
        self._opp_cache: dict = {}
        self._env = None
        self._illegal = 0
        # kaggle env is created lazily; import cost paid once per worker proc
        from kaggle_environments import make as kaggle_make
        self._make = kaggle_make

    # -- opponent plumbing ---------------------------------------------------
    def _draw_opponent(self):
        """Two buckets only -- mirror and league, both LINEAGE.

        Returns (agent_fn, slug); slug feeds the wr_<slug> terminal info. An
        empty league falls back to the mirror rather than to anything external:
        there is no third bucket to fall through into any more.
        """
        u = self.rng.random()
        if u < self.mix[0] or not self.league:
            if self._mirror is not None:
                return self._mirror.agent, "mirror"
            return self._cached(("ckpt", self.anchor)), "mirror"
        spec = self.league[self.rng.randrange(len(self.league))]
        return self._cached(spec), f"league_{Path(spec[1]).name}"

    def _cached(self, spec: tuple[str, str]):
        if spec in self._opp_cache:
            return self._opp_cache[spec]
        kind, ref = spec
        if kind == "ckpt":
            fn = as_env_agent(SamplingPolicy(ref, temperature=1.0, record=False).agent)
        else:
            sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
            import benchmark_agents
            fn = benchmark_agents.load_agent(ref)
        self._opp_cache[spec] = fn
        return fn

    # -- engine driving ------------------------------------------------------
    def _statuses(self):
        return [self._env.state[i]["status"] for i in range(2)]

    def _obs_of(self, i: int) -> dict:
        return self._env.state[i]["observation"]

    def _advance_until_learner(self):
        """Submit opponent/deck actions until the learner faces a real select,
        or the game ends. Returns (obs_feats, done)."""
        while not self._env.done:
            sts = self._statuses()
            actions = [None, None]
            learner_turn = False
            for i in range(2):
                if sts[i] != "ACTIVE":
                    actions[i] = []
                    continue
                obs = self._obs_of(i)
                sel = obs.get("select")
                if i == self.learner_seat:
                    if sel is None:
                        actions[i] = self.deck
                    elif not (sel.get("option") or []):
                        actions[i] = []
                    else:
                        learner_turn = True
                else:
                    try:
                        actions[i] = self.opponent(obs) if sel is not None else self.opponent(obs)
                    except Exception:
                        actions[i] = _safe_choice(sel) if sel else []
            if learner_turn:
                self._pending_actions = actions  # opponent slot already filled
                return self._begin_select(self._obs_of(self.learner_seat))
            self._env.step(actions)
        return None, True

    def _begin_select(self, obs: dict):
        sel = obs["select"]
        self._sel_obs = obs
        self._picked: list[int] = []
        self._excluded: set[int] = set()
        self._max_count = sel["maxCount"] if sel.get("maxCount") is not None else 1
        self._min_count = sel.get("minCount") or 0
        self._n_opts = len(sel["option"])
        feats = encode_observation(obs, exclude=frozenset(self._excluded))
        if feats is None:  # unencodable: safe-choice the whole select, move on
            return self._submit(_safe_choice(sel))
        self._n_real = feats.pop("n_real_options")
        return pack_obs(feats), False

    def _submit(self, action_list: list[int]):
        acts = self._pending_actions
        acts[self.learner_seat] = action_list
        self._env.step(acts)
        return self._advance_until_learner()

    # -- gym API -------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self._env = self._make("cabt")
        self._env.reset(2)
        if self.alternate_seats:
            self.learner_seat = self._next_seat
            self._next_seat = 1 - self._next_seat
        else:
            self.learner_seat = self.rng.randint(0, 1)
        if self.deck_pool is not None:
            # Two independent draws per episode: the learner's deck and the
            # opponent's. mirror_deck collapses them to one draw (the control
            # arm). Drawn BEFORE the opponent draw so the wrapper installed
            # below already sees this episode's values.
            self.deck_name, self.deck = self.deck_pool.sample(self.rng)
            if self.mirror_deck:
                self.opp_deck_name, self.opp_deck = self.deck_name, self.deck
            else:
                self.opp_deck_name, self.opp_deck = self.deck_pool.sample(self.rng)
        if self._opp_left <= 0:
            self.opponent, self._opp_slug = self._draw_opponent()
            if self.deck_pool is not None:
                # Force the opponent onto the pooled deck. Without this a
                # checkpoint opponent submits SamplingPolicy's own load_deck()
                # list and a module opponent submits its bundled one, so
                # widening only the learner's side leaves every opponent on the
                # same deck forever. The wrapper's lambda reads self.opp_deck at
                # call time, so a HELD opponent (opp_hold persistence) keeps
                # tracking each new episode's draw.
                self.opponent = deck_override_agent(self.opponent,
                                                    lambda: self.opp_deck)
            self._opp_left = self.opp_hold
        self._opp_left -= 1
        self._illegal = 0
        obs, done = self._advance_until_learner()
        if done:  # pathological instant end; hand back a zero obs, will reset
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        action = int(action)
        add_decline = self._min_count <= len(self._picked)
        decline_idx = self._n_real if add_decline else None

        legal = action < self._n_real and action not in self._excluded and action < self._n_opts
        is_decline = decline_idx is not None and action == decline_idx
        if not (legal or is_decline):
            self._illegal += 1
            sel = self._sel_obs["select"]
            remaining = [i for i in range(min(self._n_opts, self._n_real))
                         if i not in self._excluded]
            if self._min_count <= len(self._picked) or not remaining:
                is_decline, legal = True, False
            else:
                action, legal = remaining[0], True

        done_select = False
        if is_decline:
            done_select = True
        else:
            self._picked.append(action)
            self._excluded.add(action)
            if len(self._picked) >= self._max_count:
                done_select = True

        if not done_select:
            feats = encode_observation(self._sel_obs, exclude=frozenset(self._excluded))
            if feats is None:
                done_select = True
            else:
                self._n_real = feats.pop("n_real_options")
                return pack_obs(feats), 0.0, False, False, {}

        obs, done = self._submit(self._picked)
        if done:
            r = [self._env.state[i]["reward"] for i in range(2)]
            scored = [float("-inf") if x is None else x for x in r]
            mine, theirs = scored[self.learner_seat], scored[1 - self.learner_seat]
            reward = 0.5 if mine == theirs else (1.0 if mine > theirs else 0.0)
            # Free outcome harvest through pufferl's info->stats aggregation.
            # Three cuts, because a POOLED win rate is exactly the number that
            # hides a policy dominating one deck and losing the rest:
            #   wr_<opp slug>   opponent type (mirror / league_<ckpt>)
            #   wrd_<deck>      the learner's deck this episode
            #   wrm_<a>~<b>     the full matchup, learner deck ~ opponent deck
            info = {"illegal_picks": self._illegal,
                    f"wr_{self._opp_slug}": reward}
            if self.deck_pool is not None:
                info[f"wrd_{self.deck_name}"] = reward
                info[f"wrm_{self.deck_name}~{self.opp_deck_name}"] = reward
            return (np.zeros(OBS_SIZE, dtype=np.float32), reward, True, False,
                    info)
        return obs, 0.0, False, False, {}


def make_puffer_env(league=None, mirror_root=None, anchor_ckpt=None,
                    opp_hold=4, mix=(0.625, 0.375),
                    alternate_seats=False, deck_pool=None, mirror_deck=False,
                    seed=0, buf=None):
    """Creator for pufferlib.vector.make — one GymnasiumPufferEnv per call.

    `deck_pool` is a built DeckPool, passed by value so every spawned worker
    samples from an identical pool (rebuilding per worker would re-read files
    and could drift if the tree changes mid-run).
    """
    import pufferlib.emulation

    torch.set_num_threads(1)
    env = PTCGGym(league=league, mirror_root=mirror_root,
                  anchor_ckpt=anchor_ckpt, opp_hold=opp_hold,
                  mix=mix, alternate_seats=alternate_seats,
                  deck_pool=deck_pool, mirror_deck=mirror_deck, seed=seed)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
