"""PufferLib 3.0 policy wrapper around PTCGImitationPolicy (side-venv only).

Contract (pufferlib.models.Default): `forward_eval(obs, state) -> (logits,
values)` and `forward` aliasing it; sampling/entropy handled by
pufferlib.pytorch.sample_logits on the returned logits.

Masking: PTCGImitationPolicy itself masks illegal slots to -inf, but torch's
Categorical.entropy() NaNs on -inf logits (0 * -inf), so illegal slots are
re-masked to a finite -1e9 here — numerically identical through softmax
(underflows to exactly 0) and NaN-free through entropy.

The value head is a fresh Linear on the actor's CLS hidden state. This is the
pufferl-standard shared-trunk critic — a deliberate difference from the
custom-loop's separate critic; value gradients DO touch the shared trunk here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .il_model import PTCGImitationPolicy
from .puffer_env import split_obs

NEG = -1e9


class PTCGPufferPolicy(nn.Module):
    def __init__(self, ckpt_dir: str) -> None:
        super().__init__()
        self.actor = PTCGImitationPolicy.from_pretrained(str(ckpt_dir))
        h = self.actor.config.hidden_size
        self.value_head = nn.Linear(h, 1)

    def forward_eval(self, observations: torch.Tensor, state=None):
        feats = split_obs(observations)
        out = self.actor(**feats)
        logits = torch.where(feats["opt_mask"], out["logits"],
                             torch.full_like(out["logits"], NEG))
        value = self.value_head(out["cls_hidden"])
        return logits, value

    def forward(self, observations: torch.Tensor, state=None):
        return self.forward_eval(observations, state)
