"""Masked-KL primitives for the Stage-3 anchor — rl_pipeline_v2.md §3.1.

Torch-only on purpose: `pufferl_kl.py` (the consumer) imports pufferlib and
can only load in the .venv-ppo side venv, but the anchor's correctness
properties — KL(π‖π)=0, illegal slots contributing exactly 0, and the
mask-identity guard firing on a mismatch — are testable without pufferlib.
tests/test_kl_mask.py runs against THIS module in the main venv; the trainer
imports the same functions, so the tested code is the shipped code.

Masking convention (repo-wide): illegal option slots carry a FINITE -1e9
logit (never -inf — torch's Categorical.entropy NaNs on -inf). Softmax
underflows -1e9 to exactly 0, so masked slots contribute 0 to both the
distribution and the KL sum, with no torch.where gating needed.
"""

from __future__ import annotations

import torch

# Same value as pokemon_tcg.puffer_policy.NEG — duplicated here (a single
# float) rather than imported, because puffer_policy pulls in puffer_env ->
# gymnasium, which the main venv does not carry.
NEG = -1e9


def masked_kl_per_row(
    cur_logits: torch.Tensor, prior_logits: torch.Tensor
) -> torch.Tensor:
    """KL(π_cur ‖ π_prior) per row over the (finite-)masked option logits.

    Only valid when BOTH logit tensors were masked with the IDENTICAL
    legal-action mask — check with `masks_agree` (the trainer asserts it
    once per prior; a mismatch silently corrupts the anchor gradient with
    probability mass on actions one side considers illegal). Illegal slots
    contribute exactly 0: p underflows to 0 through softmax of NEG, and
    logp − logq stays finite because both sides are offset by the same NEG.
    """
    logp = torch.log_softmax(cur_logits, dim=-1)
    logq = torch.log_softmax(prior_logits, dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def masked_kl(cur_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    """Batch-mean KL — the loss term β multiplies."""
    return masked_kl_per_row(cur_logits, prior_logits).mean()


def masks_agree(cur_logits: torch.Tensor, prior_logits: torch.Tensor) -> bool:
    """True iff both logit tensors mark the SAME slots illegal.

    NEG/10 cleanly separates masked slots (-1e9) from any reachable real
    logit; comparing the two boolean maps is the §3.1 mask-identity guard.
    """
    return bool(torch.equal(cur_logits < NEG / 10, prior_logits < NEG / 10))
