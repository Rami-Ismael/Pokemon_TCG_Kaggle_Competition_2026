"""Promotion-ratchet unit tests: the gate's decision rule and the
frozen-reference retarget mechanics.

evaluate_gate itself (spawned workers playing real cabt games) is exercised
by the train_ppo_puffer.py smoke path, not here -- these tests cover the
pure logic that must not regress: strict-threshold arithmetic, the
decisive-games floor, and that promotion copies weights into the reference
WITHOUT thawing it.

PuffeRLPriorKL is deliberately not imported: it needs pufferlib 3.0's API
(side venv .venv-ppo); the main venv carries pufferlib 4.0. The retarget
test reproduces retarget_prior's exact mechanics (load_state_dict into a
frozen eval-mode policy) on PTCGPufferPolicy directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.promotion import MIN_DECISIVE, gate_verdict  # noqa: E402


def outcomes(wins: int = 0, losses: int = 0, draws: int = 0) -> list[float]:
    return [1.0] * wins + [0.0] * losses + [0.5] * draws


def test_threshold_is_strict():
    # 70 of 100 decisive is NOT strictly more than 0.70 -- no promotion.
    at = gate_verdict(outcomes(wins=70, losses=30), threshold=0.70)
    assert at["win_rate"] == pytest.approx(0.70)
    assert not at["promote"]
    above = gate_verdict(outcomes(wins=71, losses=29), threshold=0.70)
    assert above["promote"]


def test_draws_are_excluded_from_win_rate():
    v = gate_verdict(outcomes(wins=15, losses=5, draws=80), threshold=0.70)
    assert v["games"] == 100
    assert v["decisive"] == 20
    assert v["win_rate"] == pytest.approx(0.75)
    assert v["promote"]  # decisive == MIN_DECISIVE floor exactly, rate above


def test_too_few_decisive_games_blocks_promotion():
    v = gate_verdict(outcomes(wins=MIN_DECISIVE - 1, draws=81), threshold=0.70)
    assert v["win_rate"] == 1.0
    assert not v["promote"]
    all_draws = gate_verdict(outcomes(draws=100), threshold=0.70)
    assert all_draws["win_rate"] == 0.0
    assert not all_draws["promote"]


def test_losing_record_never_promotes():
    v = gate_verdict(outcomes(wins=30, losses=70), threshold=0.70)
    assert not v["promote"]


@pytest.mark.skipif(
    not (config.MODELS_DIR / "il_agent" / "config.json").exists()
    or not (config.MODELS_DIR / "s2" / "e1_seed43" / "config.json").exists(),
    reason="needs il_agent and s2/e1_seed43 checkpoints (symlink from main checkout)",
)
def test_retarget_mechanics_copy_weights_without_thawing():
    """retarget_prior == load_state_dict into the frozen reference: values
    become the live policy's, requires_grad stays False, eval mode stays."""
    from pokemon_tcg.puffer_policy import PTCGPufferPolicy

    live = PTCGPufferPolicy(str(config.MODELS_DIR / "s2" / "e1_seed43"))
    ref = PTCGPufferPolicy(str(config.MODELS_DIR / "il_agent")).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    before = [
        (n, p.clone()) for n, p in ref.named_parameters()
        if not torch.equal(p, dict(live.named_parameters())[n])
    ]
    assert before, "il_agent and the s2 arm should differ somewhere"

    with torch.no_grad():
        ref.load_state_dict(live.state_dict())

    live_params = dict(live.named_parameters())
    for name, p in ref.named_parameters():
        assert torch.equal(p, live_params[name]), f"{name} not copied"
        assert not p.requires_grad, f"{name} thawed by retarget"
    assert not ref.training, "reference left eval mode"
