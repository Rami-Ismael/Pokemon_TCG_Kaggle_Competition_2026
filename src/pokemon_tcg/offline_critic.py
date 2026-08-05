"""S2-E4 critic-advantage components — rl_pipeline_v1.md §2.1, row E4.

The offline analog of Metamon's advantage-based objectives (Grigsby et al.,
arXiv:2504.04395, Table 1): a state-value critic V(s) trained on the remapped
terminal outcome {0, 0.5, 1}, consumed at BC-training time as a per-row weight

    adv-exp:    w = exp(beta * (outcome - V(s))), clipped
    adv-binary: w = 1[outcome - V(s) > 0]

With V frozen at the constant 0.5 these collapse exactly to the already-trained
arms (S2-E2 outcome-weighted and S2-E1 winners-only respectively) — that
equivalence is asserted in tests/test_e4_critic.py and is the correctness
anchor for this module.

Deliberately import-light: ppo.py has an identical ValueModel but pulls in
selfplay/kaggle_environments, which the offline path must not depend on. The
critic is TRAIN-TIME ONLY — it never ships in a submission bundle (CPU +
~197.7 MiB envelope; same rule as Stage 3's critic).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .il_model import PTCGILConfig, PTCGImitationPolicy

ADV_ARMS = ("adv-exp", "adv-binary")
_STATE_FILE = "critic_state.pt"
_CONFIG_FILE = "config.json"


class OfflineValueModel(nn.Module):
    """Critic: a PTCGImitationPolicy trunk + fresh scalar head on cls_hidden.

    The trunk is a *copy* (its own parameters), so value gradients never touch
    a live actor. Construct via `from_actor_checkpoint` (warm-start, the real
    E4 path) or `from_config` (fresh, for smoke tests).
    """

    def __init__(self, trunk: PTCGImitationPolicy) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = nn.Linear(trunk.config.hidden_size, 1)

    @classmethod
    def from_actor_checkpoint(cls, ckpt_dir: str | Path) -> "OfflineValueModel":
        return cls(PTCGImitationPolicy.from_pretrained(str(ckpt_dir)))

    @classmethod
    def from_config(cls, config: PTCGILConfig) -> "OfflineValueModel":
        return cls(PTCGImitationPolicy(config))

    def forward(self, **feats) -> torch.Tensor:
        out = self.trunk(**feats)
        return self.head(out["cls_hidden"]).squeeze(-1)


def save_critic(model: OfflineValueModel, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / _CONFIG_FILE).write_text(model.trunk.config.to_json_string())
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state, out / _STATE_FILE)


def load_critic(critic_dir: str | Path, device: str = "cpu") -> OfflineValueModel:
    d = Path(critic_dir)
    cfg = PTCGILConfig(**json.loads((d / _CONFIG_FILE).read_text()))
    model = OfflineValueModel.from_config(cfg)
    model.load_state_dict(torch.load(d / _STATE_FILE, map_location="cpu"))
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def advantage_weights(
    arm: str,
    outcome: torch.Tensor,
    value: torch.Tensor,
    beta: float = 1.0,
    clip: float = 20.0,
) -> torch.Tensor:
    """Per-row weights from advantage = outcome − V(s).

    Rows with unknown outcome (the −1 sentinel) get weight exactly 0 in every
    arm — excluded, never treated as "a bit worse than a loss" (same rule the
    existing outcome arm enforces). adv-exp is clipped because exp(beta*adv)
    on a badly calibrated critic can otherwise hand a single row the whole
    batch's gradient.
    """
    if arm not in ADV_ARMS:
        raise ValueError(f"unknown advantage arm {arm!r}; expected one of {ADV_ARMS}")
    known = (outcome >= 0.0).float()
    # V outside [0,1] is definitionally invalid for a {0, 0.5, 1} outcome —
    # the 2026-08-04 critic audits found the extreme-|Â| tail was exactly
    # such out-of-range regression artifacts (V ≈ −0.4 on even midgames).
    # Clamping at consumption bounds every adv-exp weight to e^beta by
    # construction (advantage ∈ [−1, 1]), so no artifact row can grab the
    # batch's gradient.
    adv = outcome - value.clamp(0.0, 1.0)
    if arm == "adv-exp":
        w = torch.exp(beta * adv).clamp(max=clip)
    else:  # adv-binary
        w = (adv > 0).float()
    return w * known
