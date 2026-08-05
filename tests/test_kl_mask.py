"""KL-anchor mask-identity guard — rl_pipeline_v2.md §3.1.

The anchor term β·KL(π_θ‖π_ref) is only meaningful when both distributions
are softmaxed over the IDENTICAL legal-action mask; a mismatch silently
corrupts the anchor gradient with probability mass on actions one side
considers illegal. These tests run against pokemon_tcg.kl_math — the exact
functions PuffeRLPriorKL imports — so the tested code is the shipped code
(kl_math is pufferlib-free precisely so this suite can run in the main venv).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg.kl_math import (  # noqa: E402
    NEG,
    masked_kl,
    masked_kl_per_row,
    masks_agree,
)


def _masked(logits: list[list[float]], mask: list[list[bool]]) -> torch.Tensor:
    t = torch.tensor(logits, dtype=torch.float32)
    m = torch.tensor(mask, dtype=torch.bool)
    return torch.where(m, t, torch.full_like(t, NEG))


def test_kl_of_identical_distributions_is_zero():
    cur = _masked([[1.0, 2.0, 0.5, 0.0]], [[True, True, True, False]])
    kl = masked_kl(cur, cur.clone())
    assert torch.isfinite(kl)
    assert abs(float(kl)) < 1e-6


def test_illegal_slots_contribute_exactly_zero():
    # Same legal logits, same mask; the two tensors are identical after
    # masking regardless of what the illegal slot 'would have been', so the
    # KL must be exactly the legal-slots KL (here: 0).
    mask = [[True, True, False]]
    cur = _masked([[1.0, 2.0, 99.0]], mask)
    prior = _masked([[1.0, 2.0, -99.0]], mask)
    assert abs(float(masked_kl(cur, prior))) < 1e-6


def test_kl_positive_when_legal_distributions_differ():
    mask = [[True, True, False]]
    cur = _masked([[3.0, 0.0, 0.0]], mask)
    prior = _masked([[0.0, 3.0, 0.0]], mask)
    kl = masked_kl(cur, prior)
    assert torch.isfinite(kl)
    assert float(kl) > 0.5


def test_mask_mismatch_is_detected():
    logits = [[1.0, 2.0, 0.5]]
    cur = _masked(logits, [[True, True, False]])
    prior_same = _masked(logits, [[True, True, False]])
    prior_diff = _masked(logits, [[True, True, True]])
    assert masks_agree(cur, prior_same)
    assert not masks_agree(cur, prior_diff)
    # the trainer's guard is `assert masks_agree(...)`: prove the failing
    # branch actually fires for the mismatched pair
    fired = False
    try:
        assert masks_agree(cur, prior_diff), "mask mismatch"
    except AssertionError:
        fired = True
    assert fired


def test_random_batch_kl_finite_nonnegative_and_per_row_consistent():
    g = torch.Generator().manual_seed(42)
    B, N = 64, 48
    mask = torch.rand(B, N, generator=g) < 0.4
    mask[:, 0] = True  # every row keeps >=1 legal slot
    cur = torch.where(mask, torch.randn(B, N, generator=g) * 3,
                      torch.full((B, N), NEG))
    prior = torch.where(mask, torch.randn(B, N, generator=g) * 3,
                        torch.full((B, N), NEG))
    per_row = masked_kl_per_row(cur, prior)
    assert per_row.shape == (B,)
    assert bool(torch.isfinite(per_row).all())
    assert float(per_row.min()) > -1e-5  # KL >= 0 up to float error
    assert abs(float(per_row.mean() - masked_kl(cur, prior))) < 1e-6
