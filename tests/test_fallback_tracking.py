"""Fallback-tracking tests: tracker math, the disabled no-op, and the masking gate.

The load-bearing assertion is `model_action_illegal == 0` on a real smoke
game: when action masking is correct that counter is STRUCTURALLY zero --
the model argmaxes over `logits[:upto]`, which only spans real options (+
the synthetic DECLINE slot). Any nonzero count is a masking bug, not a
fallback, so this converts the diagnostic into a CI gate.

Also pinned: with PTCG_FALLBACK_TRACK unset the tracker is a genuine no-op
(the submission-bundle invariant), and the rate math counts prefixed
reasons (policy_exception:ValueError) under their base while keeping
silent-degradation signals (unknown_card, nan_logits) OUT of the rate.

Requires no pytest: `uv run python tests/test_fallback_tracking.py` works too.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORE = REPO / "agents" / "il_agent" / "agent_core.py"
DRIVER = REPO / "scripts" / "fallback_diagnostic.py"


def _load_core(track: bool, tag: str):
    """Fresh module instance of agent_core with the flag explicitly set/unset."""
    if track:
        os.environ["PTCG_FALLBACK_TRACK"] = "1"
    else:
        os.environ.pop("PTCG_FALLBACK_TRACK", None)
    spec = importlib.util.spec_from_file_location(f"_fbtest_core_{tag}", CORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_explicit_missing_model_dir_fails_loudly():
    """An explicit IL_MODEL_DIR that doesn't exist must raise at import.

    The old behavior silently resolved to the dev checkpoint, so a
    checkpoint sweep (e.g. an s2_arms wrapper in a fresh worktree) would
    benchmark Stage-1 weights under the arm's name -- it defeated this
    diagnostic's own first negative control. The unset-env dev fallback to
    models/il_agent stays allowed (exercised by every other test here).
    """
    os.environ["IL_MODEL_DIR"] = "/nonexistent_model_dir_for_test"
    try:
        spec = importlib.util.spec_from_file_location("_fbtest_core_missdir", CORE)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except FileNotFoundError as e:
            assert "IL_MODEL_DIR" in str(e)
        else:
            raise AssertionError(
                "importing agent_core with a nonexistent explicit IL_MODEL_DIR "
                "should raise FileNotFoundError, not silently redirect"
            )
    finally:
        os.environ.pop("IL_MODEL_DIR", None)


def test_disabled_is_a_genuine_noop():
    """Flag unset (the Kaggle-bundle case): hooks count nothing, keep nothing."""
    mod = _load_core(track=False, tag="off")
    assert mod._TRACK is False
    mod._diag("policy_exception:ValueError", {"traceback": "boom"})
    mod._diag("model_action_illegal", {"choice": 3})
    snap = mod.diag_snapshot()
    assert snap["enabled"] is False
    assert snap.get("decisions", 0) == 0
    assert snap["fallbacks"] == 0
    assert mod.diag_first() == {}


def test_rate_math_and_reason_classification():
    mod = _load_core(track=True, tag="on")
    mod.diag_reset()
    mod._DIAG["decisions"] += 10
    mod._diag("policy_exception:ValueError", {"traceback": "tb-1"})
    mod._diag("policy_exception:KeyError", {"traceback": "tb-2"})
    mod._diag("model_action_illegal", {"choice": 3, "n_opts": 2})
    # Signals: the model still chose -- these must never inflate the rate.
    mod._diag("nan_logits")
    mod._diag("unknown_card")
    mod._diag("unknown_card")

    snap = mod.diag_snapshot()
    assert snap["enabled"] is True
    assert snap["decisions"] == 10
    assert snap["fallbacks"] == 3, "2 policy_exception:* + 1 model_action_illegal"
    assert abs(snap["fallback_rate"] - 0.3) < 1e-9
    assert snap["unknown_card"] == 2 and snap["nan_logits"] == 1

    first = mod.diag_first()
    # First occurrence is kept per BASE reason: tb-1 won, tb-2 only counted.
    assert first["policy_exception"]["reason"] == "policy_exception:ValueError"
    assert first["policy_exception"]["traceback"] == "tb-1"
    assert first["model_action_illegal"]["choice"] == 3

    mod.diag_reset()
    assert mod.diag_snapshot()["fallbacks"] == 0
    assert mod.diag_first() == {}


def test_smoke_game_masking_is_clean():
    """One real mirrored pair vs random_legal: model_action_illegal must be 0.

    random_legal is the stress opponent on purpose -- uniform-random play
    reaches states a rule opponent never produces.
    """
    out = REPO / "reports" / "fallback_diag_smoke.json"
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--agent", "il_agent",
         "--opponents", "random_legal", "--pairs", "1",
         "--assert-clean", "--no-tb", "--out", str(out)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"assert-clean failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    data = json.loads(out.read_text())
    snap = data["fallback"]
    assert snap["enabled"] is True
    assert snap["decisions"] > 0, "no decisions measured -- the smoke game didn't run"
    illegal = sum(v for k, v in snap.items()
                  if isinstance(v, int) and k.split(":", 1)[0] == "model_action_illegal")
    assert illegal == 0, f"masking bug: model_action_illegal={illegal} (see {out})"


if __name__ == "__main__":
    test_explicit_missing_model_dir_fails_loudly()
    print("explicit missing IL_MODEL_DIR fails loudly: OK")
    test_disabled_is_a_genuine_noop()
    print("disabled no-op: OK")
    test_rate_math_and_reason_classification()
    print("rate math: OK")
    test_smoke_game_masking_is_clean()
    print("smoke game masking gate: OK")
