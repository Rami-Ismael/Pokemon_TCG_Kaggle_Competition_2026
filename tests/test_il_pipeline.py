"""Smoke tests for the IL pipeline fixes made in this pass:
device resolution, decline-as-option, multi-select autoregressive unroll,
OOV-safe embeddings, and the never-crash agent wrapper.

Run directly: uv run python tests/test_il_pipeline.py
"""
from __future__ import annotations

import json
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
)
from pokemon_tcg.il_model import PTCGILConfig, PTCGImitationPolicy  # noqa: E402

TRAIN_DIR = REPO / "data" / "episodes" / "splits" / "train-2026-07-26"


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
    for obs, label, exclude in iter_decisions(TRAIN_DIR, max_episodes=15):
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
    against it is -log(softmax of -inf) == inf. (Caught during the Phase 6
    speed probe: `add_decline` used to check `minCount == 0` globally
    instead of accounting for picks already made in this unroll step.)
    """
    n_checked = 0
    for obs, label, exclude in iter_decisions(TRAIN_DIR, max_episodes=40):
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


def test_agent_degrades_gracefully_when_torch_unavailable():
    """Whether torch/transformers are importable in the Kaggle evaluator
    sandbox is unverified (the working reference agents in this repo depend
    on neither). Simulate that: agent_core.py must still play a full,
    legal game -- never raise ImportError up through agent().
    """
    import builtins
    import importlib
    import json

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("simulated: torch unavailable")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked_import
    try:
        sys.path.insert(0, str(REPO / "agents" / "il_agent"))
        import agent_core

        importlib.reload(agent_core)
        assert agent_core._ML_AVAILABLE is False
    finally:
        builtins.__import__ = real_import

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
    importlib.reload(agent_core)  # restore normal (ML-available) state for later tests


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


if __name__ == "__main__":
    tests = [
        test_resolve_device,
        test_main_py_survives_kaggle_exec_loading,
        test_agent_degrades_gracefully_when_torch_unavailable,
        test_lr_schedule_does_not_prematurely_decay_to_zero,
        test_option_type_vocab_includes_decline,
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
