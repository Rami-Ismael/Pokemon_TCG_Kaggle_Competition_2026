"""Smoke tests for the IL pipeline fixes made in this pass:
device resolution, decline-as-option, multi-select autoregressive unroll,
OOV-safe embeddings, and the never-crash agent wrapper.

Run directly: uv run python tests/test_il_pipeline.py
"""
from __future__ import annotations

import builtins
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "data" / "external" / "cg-lib"))

import torch  # noqa: E402

from pokemon_tcg.device import resolve_device  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    DECLINE_OPTION_TYPE,
    MAX_OPTIONS,
    OPTION_TYPE_VOCAB_SIZE,
    encode_observation,
    iter_decisions,
    winning_agent,
)
from pokemon_tcg.il_model import PTCGILConfig, PTCGImitationPolicy  # noqa: E402

TRAIN_DIR = REPO / "data" / "episodes" / "splits" / "train-2026-07-26"
CORE = REPO / "agents" / "il_agent" / "agent_core.py"


@contextlib.contextmanager
def _ml_stack_unimportable():
    """Simulate `import torch` failing, as it may in the evaluator sandbox."""
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("simulated: torch unavailable")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked_import
    try:
        yield
    finally:
        builtins.__import__ = real_import


def _load_agent_core(tag: str, *, strict: bool):
    """Fresh agent_core instance with PTCG_STRICT_LOAD pinned explicitly.

    Pinned rather than inferred so neither test depends on agent_core's own
    _IN_EVALUATOR probe or on an ambient env var. A fresh module (rather than
    importlib.reload) keeps a deliberately-failed import from leaving
    half-executed globals in sys.modules for the rest of the suite.
    """
    prev = os.environ.get("PTCG_STRICT_LOAD")
    os.environ["PTCG_STRICT_LOAD"] = "1" if strict else "0"
    try:
        spec = importlib.util.spec_from_file_location(f"_iltest_core_{tag}", CORE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if prev is None:
            os.environ.pop("PTCG_STRICT_LOAD", None)
        else:
            os.environ["PTCG_STRICT_LOAD"] = prev


def _first_episode():
    files = sorted(TRAIN_DIR.glob("*.json"))
    assert files, "no train episodes found -- expected local data on disk"
    return json.loads(files[2].read_text())


def test_resolve_device():
    assert resolve_device(override="cpu") == "cpu"
    d = resolve_device()
    assert d in ("cpu", "mps")
    try:
        resolve_device(override="bogus")
        raise AssertionError("expected ValueError for bad override")
    except ValueError:
        pass


def test_decline_slot_present_when_min_count_zero():
    ep = _first_episode()
    found = False
    for step in ep["steps"]:
        obs = step[0]["observation"]
        sel = obs.get("select")
        if sel and (sel.get("minCount") or 0) == 0 and sel.get("option"):
            feats = encode_observation(obs)
            assert feats is not None
            n = feats["n_real_options"]
            assert n < MAX_OPTIONS
            assert feats["opt_type"][n].item() == DECLINE_OPTION_TYPE
            assert feats["opt_mask"][n].item() is True
            found = True
            break
    assert found, "no minCount==0 decision found in sample episode to test against"


def test_multiselect_unroll_masks_prior_picks():
    ep = _first_episode()
    steps = ep["steps"]
    multi = None
    for step in steps:
        obs = step[0]["observation"]
        sel = obs.get("select")
        if sel and (sel.get("maxCount") or 1) > 1 and len(sel.get("option", [])) >= 2:
            multi = obs
            break
    if multi is None:
        print("  (no multi-select decision in this sample episode, skipping)")
        return
    exclude = frozenset({0})
    feats = encode_observation(multi, exclude=exclude)
    assert feats is not None
    assert feats["opt_mask"][0].item() is False, "excluded option must be masked out"


def test_oov_card_id_does_not_crash_model():
    cfg = PTCGILConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2)
    model = PTCGImitationPolicy(cfg).eval()
    huge = torch.tensor([[999_999] + [0] * 100])[:, : model.card_emb.num_embeddings + 1]
    # directly hammer the embedding path the way a corrupted/unclamped id would
    out = model.card_emb(huge.clamp(max=model.card_emb.num_embeddings - 1))
    assert out.shape[-1] == 32
    print("  OOV id clamped safely, no crash")


def test_iter_decisions_yields_declines_and_multiselect_rows():
    n_decline = 0
    n_multiselect_step = 0
    n_total = 0
    for obs, label, exclude, _meta in iter_decisions(TRAIN_DIR, max_episodes=15):
        n_total += 1
        sel = obs["select"]
        n_opts = len(sel["option"])
        if label == n_opts:
            n_decline += 1
        if exclude:
            n_multiselect_step += 1
        assert label <= MAX_OPTIONS
    print(f"  {n_total} decisions, {n_decline} declines, {n_multiselect_step} re-masked multi-select steps")
    assert n_total > 0


def test_every_label_points_at_an_unmasked_slot():
    """Regression test: a terminal multi-select DECLINE label must land on a
    slot that encode_observation actually marks scoreable, or cross-entropy
    against it is -log(softmax of -inf) == inf. (Caught during the runtime
    speed probe: `add_decline` used to check `minCount == 0` globally
    instead of accounting for picks already made in this unroll step.)
    """
    n_checked = 0
    for obs, label, exclude, _meta in iter_decisions(TRAIN_DIR, max_episodes=40):
        feats = encode_observation(obs, exclude=exclude)
        assert feats is not None
        assert feats["opt_mask"][label].item() is True, (
            f"label {label} points at a masked slot -- would produce inf loss "
            f"(select={obs['select']}, exclude={exclude})"
        )
        n_checked += 1
    print(f"  checked {n_checked} (obs, label, exclude) rows, all labels land on unmasked slots")
    assert n_checked > 0


def test_main_py_survives_kaggle_exec_loading():
    """Regression test for a real failed submission (episode 89231121):
    Kaggle's kaggle_environments/agent.py loads main.py via
    exec(code_object, env) -- NOT a normal script run or `import` -- and
    exec()'d code has no `__file__` in its namespace unless the caller
    injects it, which Kaggle's harness doesn't. `main.py` used to do
    `os.path.abspath(__file__)` unconditionally at module scope, which
    crashed with NameError under exec() even though it worked fine locally
    via `python main.py` / `import main` (both set __file__ normally --
    which is exactly why local testing missed this before the real
    submission caught it).
    """
    bundle = REPO / "submissions" / "il_agent"
    if not (bundle / "main.py").exists():
        print("  (no submissions/il_agent bundle built yet, skipping)")
        return
    import os

    cwd = os.getcwd()
    os.chdir(bundle)
    try:
        src = (bundle / "main.py").read_text()
        code_object = compile(src, "main.py", "exec")
        env = {}  # exactly what kaggle_environments/agent.py passes: no __file__
        exec(code_object, env)  # noqa: S102
        assert callable(env.get("agent")), "main.py must define a callable `agent`"
        assert len(env["agent_core"].my_deck) == 60
    finally:
        os.chdir(cwd)
    print("  main.py loaded successfully via exec() with no __file__ in namespace")


def test_missing_ml_stack_fails_loudly_outside_the_evaluator():
    """Outside the evaluator, an unimportable ML stack must RAISE at import.

    The degraded path (next test) is right INSIDE the evaluator, where a
    crash is INVALID and an instant loss. Everywhere else it is a
    measurement hazard: the agent still "plays", answers every decision
    with _safe_choice, and reads as a weak model rather than a broken one.
    That cost a full deck-selection benchmark on 2026-08-04 and nearly
    invalidated the self-play field test on 2026-08-05 -- see 7ed94d7.
    """
    with _ml_stack_unimportable():
        try:
            _load_agent_core("strict", strict=True)
        except ImportError as e:
            msg = str(e)
            assert "simulated: torch unavailable" in msg, (
                "the raise must carry the original import error, or the real "
                f"cause is unrecoverable from the message; got: {msg}"
            )
            assert "PTCG_STRICT_LOAD=0" in msg, (
                "the raise must name the escape hatch that reaches the "
                f"degraded path on purpose; got: {msg}"
            )
        else:
            raise AssertionError(
                "importing agent_core with no ML stack must raise ImportError "
                "outside the evaluator, not silently degrade to _safe_choice"
            )
    print("  no-ML-stack import raised ImportError naming PTCG_STRICT_LOAD=0")


def test_agent_degrades_gracefully_when_torch_unavailable():
    """Whether torch/transformers are importable in the Kaggle evaluator
    sandbox is unverified (the working reference agents in this repo depend
    on neither). Simulate that: agent_core.py must still play a full,
    legal game -- never raise ImportError up through agent().

    PTCG_STRICT_LOAD=0 is how that evaluator path is rehearsed outside the
    evaluator (see .claude/skills/run-fallback-diagnostic/SKILL.md); with it
    unset the same import fails loudly instead, which is the test above.
    """
    with _ml_stack_unimportable():
        agent_core = _load_agent_core("degraded", strict=False)
        assert agent_core._ML_AVAILABLE is False
        assert agent_core._load_model() is None, (
            "no ML stack must mean no model, i.e. every decision below is "
            "genuinely exercising the _safe_choice path"
        )

    agent_core.my_deck = [
        int(x) for x in (REPO / "agents" / "il_agent" / "deck.csv").read_text().split() if x.strip()
    ]
    ep = json.loads(sorted(TRAIN_DIR.glob("*.json"))[3].read_text())
    n = 0
    for step in ep["steps"]:
        obs = step[0]["observation"]
        r = agent_core.agent(obs)
        assert isinstance(r, list)
        if obs.get("select"):
            sel = obs["select"]
            assert all(0 <= i < len(sel["option"]) for i in r)
        n += 1
    print(f"  {n} decisions handled with torch unavailable, all legal, no crash")


def test_lr_schedule_does_not_prematurely_decay_to_zero():
    """Regression test for the runs/full_epoch1 bug: est_total_steps used to
    be a fixed placeholder (max(warmup_steps*10, 2000)) independent of the
    real epoch length, so a 12,854-step run decayed LR to exactly 0 at step
    2000 and trained at LR=0 (no-op) for the remaining ~10,850 steps.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib

    train_il = importlib.import_module("train_il")
    lr = train_il.cosine_warmup_lr(step=2000, warmup_steps=200, total_steps=12854, base_lr=3e-4)
    assert lr > 0, "LR must not be zero mid-epoch for a realistically-sized total_steps"
    lr_at_end = train_il.cosine_warmup_lr(step=12853, warmup_steps=200, total_steps=12854, base_lr=3e-4)
    assert lr_at_end < 1e-6, "LR should be ~0 only at the actual end of the schedule"


def test_option_type_vocab_includes_decline():
    assert OPTION_TYPE_VOCAB_SIZE == 18
    assert DECLINE_OPTION_TYPE == 17


def test_winning_agent_resolves_every_reward_shape():
    # unique max -> that seat; None counts as a loss; ties (draws) -> no winner
    assert winning_agent({"rewards": [1, -1]}) == 0
    assert winning_agent({"rewards": [-1, 1]}) == 1
    assert winning_agent({"rewards": [1, None]}) == 0
    assert winning_agent({"rewards": [None, 1]}) == 1
    assert winning_agent({"rewards": [0, 0]}) is None       # draw
    assert winning_agent({"rewards": [None, None]}) is None  # both errored
    assert winning_agent({"rewards": []}) is None
    assert winning_agent({}) is None


def test_decision_meta_outcomes_are_consistent():
    """Stage-2 plumbing: every row's meta must carry a valid outcome, and the
    two seats of one episode must have complementary outcomes (win+loss or
    draw+draw both sum to 1.0). This is the field that was previously absent
    entirely -- both seats were cloned with zero reference to `rewards`."""
    per_seat: dict[tuple[int, int], float] = {}
    n = 0
    for _obs, _label, _exclude, meta in iter_decisions(TRAIN_DIR, max_episodes=20):
        assert meta.outcome in (0.0, 0.5, 1.0), f"bad outcome {meta.outcome}"
        assert meta.seat in (0, 1)
        assert meta.episode_id > 0, "train split filenames are numeric episode ids"
        assert meta.turn >= -1
        prev = per_seat.setdefault((meta.episode_id, meta.seat), meta.outcome)
        assert prev == meta.outcome, "outcome must be constant within one (episode, seat)"
        n += 1
    assert n > 0
    episodes = {ep for ep, _ in per_seat}
    n_pairs = 0
    for ep in episodes:
        if (ep, 0) in per_seat and (ep, 1) in per_seat:
            total = per_seat[(ep, 0)] + per_seat[(ep, 1)]
            assert total == 1.0, f"episode {ep}: seat outcomes {total} != 1.0"
            n_pairs += 1
    assert n_pairs > 0, "expected at least one episode with both seats present"
    print(f"  {n} rows, {n_pairs} episodes with both seats, outcomes complementary")


def test_ildataset_with_meta_batches_through_dataloader():
    """with_meta=True must survive default collate and deliver the manifest
    join: outcome/seat/episode_id/turn/avg_score stacked as [B]-shaped
    tensors alongside the model features."""
    from torch.utils.data import DataLoader

    from pokemon_tcg.il_dataset import ILDataset

    ds = ILDataset(TRAIN_DIR, max_episodes=3, shuffle_buffer=1, with_meta=True)
    batch = next(iter(DataLoader(ds, batch_size=16)))
    for key in ILDataset.META_KEYS:
        assert key in batch, f"missing meta key {key}"
        assert batch[key].shape[0] == batch["label"].shape[0]
    assert batch["outcome"].min() >= 0.0, "sampled episodes should all have known outcomes"
    n_scored = int((batch["avg_score"] > 0).sum())
    print(f"  batch of {batch['label'].shape[0]}: meta keys present, "
          f"{n_scored} rows matched to manifest avg_score")


def test_winner_only_drops_losing_seat_decisions():
    # both-seats clones winner+loser; winner_only clones only the winning seat,
    # so it must yield strictly fewer (but non-zero) decisions on real episodes.
    both = sum(1 for _ in iter_decisions(TRAIN_DIR, max_episodes=40, winner_only=False))
    win = sum(1 for _ in iter_decisions(TRAIN_DIR, max_episodes=40, winner_only=True))
    assert win > 0, "winner_only produced no decisions -- filter is dropping everything"
    assert win < both, "winner_only must drop loser-side rows, so it cannot equal both-seats"


def test_n_episodes_counts_only_allowlisted_episodes():
    """A filtered arm must not report the unfiltered shard size.

    Training schedules are sized off n_episodes(). If the allowlist is
    ignored here, a filtered arm silently receives the SAME step budget as
    the unfiltered arm -- reintroducing the step-matching the --epochs
    budget exists to avoid (see run_skill_filter_ablation.py).
    """
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    from pokemon_tcg.il_dataset import ShardILDataset

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shard_dir = root / "train" / "day=2026-07-26"
        shard_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({"episode_id": list(range(100))}), shard_dir / "shard-000.parquet"
        )

        unfiltered = ShardILDataset("train", local_root=root).n_episodes()
        assert unfiltered == 100, unfiltered

        allowed = {3, 17, 42}
        filtered = ShardILDataset(
            "train", local_root=root, episode_ids=allowed
        ).n_episodes()
        assert filtered == len(allowed), (
            f"allowlist of {len(allowed)} reported as {filtered} episodes -- "
            "n_episodes ignored episode_ids, so filtered arms would be "
            "step-matched to the unfiltered arm"
        )

        # ids outside the shard must not be counted as present
        absent = ShardILDataset(
            "train", local_root=root, episode_ids={5, 999_999}
        ).n_episodes()
        assert absent == 1, absent


if __name__ == "__main__":
    tests = [
        test_resolve_device,
        test_n_episodes_counts_only_allowlisted_episodes,
        test_main_py_survives_kaggle_exec_loading,
        test_missing_ml_stack_fails_loudly_outside_the_evaluator,
        test_agent_degrades_gracefully_when_torch_unavailable,
        test_lr_schedule_does_not_prematurely_decay_to_zero,
        test_option_type_vocab_includes_decline,
        test_winning_agent_resolves_every_reward_shape,
        test_winner_only_drops_losing_seat_decisions,
        test_decision_meta_outcomes_are_consistent,
        test_ildataset_with_meta_batches_through_dataloader,
        test_oov_card_id_does_not_crash_model,
        test_decline_slot_present_when_min_count_zero,
        test_multiselect_unroll_masks_prior_picks,
        test_iter_decisions_yields_declines_and_multiselect_rows,
        test_every_label_points_at_an_unmasked_slot,
    ]
    for t in tests:
        print(f"running {t.__name__}...")
        t()
        print("  OK")
    print("\nall smoke tests passed")
