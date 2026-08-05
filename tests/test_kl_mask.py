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
    same_policy_weights,
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


# --- second reference: the never-retargeted IL prior (rl_pipeline_v4 §3.3) ---
#
# `PuffeRLPriorKL` logs kl_to_prior against the anchor (which moves on every
# promotion) and kl_to_il_prior against the original IL checkpoint (which never
# does). It skips the second forward pass while the two references hold the
# same weights, which is true in generation 1 (one checkpoint passed as both)
# and FALSE from generation 2 on (anchor = a promoted teacher, IL reference =
# the human prior — run_selfplay_g3.sh is exactly this shape). Taking the
# shortcut in that case would log the anchor's KL under both names, so the
# decision is made from the weights by `same_policy_weights`.


def _tiny(seed: int, *, bias: float = 0.0) -> torch.nn.Module:
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    if bias:
        with torch.no_grad():
            m[0].bias += bias
    return m


def test_same_weights_true_for_a_load_state_dict_copy():
    a, b = _tiny(0), _tiny(1)
    assert not same_policy_weights(a, b)
    b.load_state_dict(a.state_dict())  # how retarget_prior copies the anchor
    assert same_policy_weights(a, b)


def test_same_weights_false_for_gen2_shape_and_after_a_retarget():
    # gen 2+: anchor and IL reference are different checkpoints from update 1
    anchor, il_prior = _tiny(0), _tiny(1)
    assert not same_policy_weights(anchor, il_prior)

    # gen 1: same checkpoint as both -> equal until the anchor is retargeted
    anchor, il_prior = _tiny(0), _tiny(0)
    assert same_policy_weights(anchor, il_prior)
    live = _tiny(0, bias=0.5)
    anchor.load_state_dict(live.state_dict())  # retarget_prior()
    assert not same_policy_weights(anchor, il_prior), (
        "retargeting the anchor must leave the IL reference behind -- if this "
        "fails the two series measure the same thing and forgetting the human "
        "prior is unmeasurable")


def test_same_weights_false_on_key_or_shape_mismatch():
    assert not same_policy_weights(_tiny(0), torch.nn.Linear(4, 3))
    wider = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Linear(5, 2))
    assert not same_policy_weights(_tiny(0), wider)
