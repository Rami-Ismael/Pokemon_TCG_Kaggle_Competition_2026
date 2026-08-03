"""Stage-3 PPO components — rl_pipeline_v1.md §3.2.

Design constraints inherited from the plan:
- Terminal-only reward, remapped {0, 0.5, 1} (never −1: bc_pipeline_v2 §8.4's
  delay-the-loss pathology). With gamma = 1 and no intermediate rewards, every
  step's return in an episode IS the episode outcome, so the advantage is
  simply `outcome − V(s)` — no GAE machinery needed until shaping exists.
- KL penalty to the frozen BC/REWEIGHT prior instead of a naive entropy bonus
  (§8.2): this is the named guard against on-policy forgetting.
- The critic is train-time only and lives in its own network (trunk initialized
  from the actor checkpoint, fresh linear head): value gradients never touch
  the shipped actor's trunk, and the submission bundle stays actor-only.
- Action masking stays in the loss (Huang & Ontanon): logits are masked -inf
  by the model itself, so log_softmax here is already the legal-set softmax.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .il_model import PTCGImitationPolicy
from .selfplay import FEATURE_KEYS


class ValueModel(nn.Module):
    """Separate critic: PTCGImitationPolicy trunk (init from a checkpoint) +
    a fresh scalar head on the CLS hidden state."""

    def __init__(self, ckpt_dir: str) -> None:
        super().__init__()
        self.trunk = PTCGImitationPolicy.from_pretrained(ckpt_dir)
        h = self.trunk.config.hidden_size
        self.head = nn.Linear(h, 1)

    def forward(self, **feats) -> torch.Tensor:
        out = self.trunk(**feats)
        return self.head(out["cls_hidden"]).squeeze(-1)


def stack_rows(rows: list[dict], outcomes: list[float], device: str) -> dict:
    """Rollout rows (numpy, one per pick) -> one batched tensor dict.

    `outcomes` is per-row (already broadcast from its game's terminal result).
    """
    batch = {}
    for k in FEATURE_KEYS:
        arr = np.stack([r[k] for r in rows])
        batch[k] = torch.from_numpy(arr).to(device)
    batch["action"] = torch.tensor([r["action"] for r in rows], dtype=torch.long, device=device)
    batch["logprob_old"] = torch.tensor([r["logprob"] for r in rows], dtype=torch.float32, device=device)
    batch["n_legal"] = torch.tensor([r["n_legal"] for r in rows], dtype=torch.float32, device=device)
    batch["ret"] = torch.tensor(outcomes, dtype=torch.float32, device=device)
    return batch


def _feats(batch: dict) -> dict:
    return {k: batch[k] for k in FEATURE_KEYS}


@torch.no_grad()
def prior_logprobs(prior: PTCGImitationPolicy, batch: dict) -> torch.Tensor:
    """Frozen-prior log-distribution over the masked option set, [B, MAX_OPTIONS]."""
    logits = prior(**_feats(batch))["logits"]
    return torch.log_softmax(logits, dim=-1)


def ppo_losses(
    actor: PTCGImitationPolicy,
    critic: ValueModel,
    batch: dict,
    prior_logp: torch.Tensor,
    clip: float = 0.2,
    kl_coef: float = 0.05,
    value_coef: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """One PPO surrogate evaluation on (a minibatch of) the rollout buffer."""
    out = actor(**_feats(batch))
    logp_all = torch.log_softmax(out["logits"], dim=-1)
    logp = logp_all.gather(1, batch["action"].unsqueeze(1)).squeeze(1)

    values = critic(**_feats(batch))
    adv = batch["ret"] - values.detach()
    # Normalize within the minibatch: with terminal-only {0,0.5,1} returns the
    # raw advantage scale drifts with the critic's calibration; normalizing
    # keeps the surrogate's scale stable across updates.
    adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)

    ratio = (logp - batch["logprob_old"]).exp()
    surr = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    policy_loss = -surr.mean()

    value_loss = nn.functional.mse_loss(values, batch["ret"])

    # KL(pi_theta || pi_prior) over the legal set. Masked slots are -inf in
    # BOTH distributions, and 0 * (-inf - -inf) = NaN. Cleaning that with
    # nan_to_num on the FORWARD is not enough: autograd still backprops
    # through the NaN-producing subtraction and poisons every gradient
    # (verified: 120/122 params non-finite before this torch.where gating).
    # torch.where routes gradients only through the selected branch, so the
    # -inf slots never touch the graph.
    mask = batch["opt_mask"].bool()
    zeros = torch.zeros_like(logp_all)
    safe_logp = torch.where(mask, logp_all, zeros)
    safe_prior = torch.where(mask, prior_logp, zeros)
    p = torch.where(mask, logp_all.exp(), zeros)
    kl_prior = (p * (safe_logp - safe_prior)).sum(dim=-1).mean()

    # Normalized entropy over legal options (>1 option rows), for logging and
    # the entropy-collapse stop signal -- not an optimization bonus.
    ent = -(p * safe_logp).sum(dim=-1)
    multi = batch["n_legal"] > 1
    ent_norm = (ent[multi] / batch["n_legal"][multi].log()).mean() if multi.any() else torch.tensor(0.0)

    loss = policy_loss + value_coef * value_loss + kl_coef * kl_prior
    metrics = {
        "loss/policy": policy_loss.item(),
        "loss/value": value_loss.item(),
        "kl_to_prior": kl_prior.item(),
        "entropy_normalized": ent_norm.item(),
        "ratio/mean": ratio.mean().item(),
        "ratio/clipped_frac": ((ratio < 1 - clip) | (ratio > 1 + clip)).float().mean().item(),
        "value/mean": values.mean().item(),
        "return/mean": batch["ret"].mean().item(),
    }
    return loss, metrics
