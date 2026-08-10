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


def same_policy_weights(a, b) -> bool:
    """True iff two policies produce identical logits by construction.

    Used once at trainer construction to decide whether the never-retargeted
    IL reference and the (retargetable) anchor reference start out as the
    same policy — generation 1 passes one checkpoint as both, generation 2+
    anchors to a promoted teacher while the IL reference stays the human
    prior. Getting this wrong logs one reference's KL under both names
    (rl_pipeline_v4 §3.3), so it is decided from the weights rather than from
    the caller's intent.

    Compares the ``actor`` submodule when there is one. The KL anchor is a
    divergence between action distributions, so only logit-producing weights
    are relevant — and `PTCGPufferPolicy` randomly initializes its value head
    on construction, so two wrappers built from the SAME checkpoint path
    differ in the critic while being identical policies. Comparing whole
    state_dicts reports those as distinct (harmless for the logged numbers,
    which stay equal, but it mislabels `kl_refs_distinct` and pays for a
    forward pass that reproduces a value already in hand).
    """
    a, b = getattr(a, "actor", a), getattr(b, "actor", b)
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys():
        return False
    return all(
        sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k]) for k in sa
    )


def masks_agree(cur_logits: torch.Tensor, prior_logits: torch.Tensor) -> bool:
    """True iff both logit tensors mark the SAME slots illegal.

    NEG/10 cleanly separates masked slots (-1e9) from any reachable real
    logit; comparing the two boolean maps is the §3.1 mask-identity guard.
    """
    return bool(torch.equal(cur_logits < NEG / 10, prior_logits < NEG / 10))
