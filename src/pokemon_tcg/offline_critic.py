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

ADV_ARMS = ("adv-exp", "adv-binary", "adv-td-binary")
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
    if arm == "adv-td-binary":
        raise ValueError(
            "adv-td-binary needs the successor state; call td_advantage_weights()"
        )
    adv = outcome - value.clamp(0.0, 1.0)
    if arm == "adv-exp":
        w = torch.exp(beta * adv).clamp(max=clip)
    else:  # adv-binary
        w = (adv > 0).float()
    return w * known


def load_calibration(critic_dir: str | Path) -> tuple[float, float]:
    """(platt_a, platt_b) from the critic's mandatory calibration.json.

    Raises if absent: the uncalibrated critic is measured worse than a constant
    (memory `shipped-mcts-critic-worse-than-constant`), so silently running
    without it is the failure mode this load exists to prevent.
    """
    p = Path(critic_dir) / "calibration.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing -- fit it with scripts/fit_critic_calibration.py before "
            "using this critic as a training-time filter"
        )
    cal = json.loads(p.read_text())
    return float(cal["platt_a"]), float(cal["platt_b"])


def _platt(value: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """sigmoid(a * logit(clip(v,0,1)) + b) -- scripts/fit_critic_calibration.py.

    Matches that module's apply_platt exactly, including the [0,1] clip, so the
    probabilities here are the ones the calibration was fitted to produce.
    """
    v = value.clamp(1e-6, 1.0 - 1e-6)
    return torch.sigmoid(a * torch.log(v / (1.0 - v)) + b)


def td_advantage_weights(
    outcome: torch.Tensor,
    value: torch.Tensor,
    value_next: torch.Tensor,
    has_next: torch.Tensor,
    platt_a: float,
    platt_b: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary weights from a ONE-STEP TD advantage. Returns (weights, adv).

    Why this arm exists: `adv-binary`'s advantage is `outcome - V(s)` with
    outcome in {0, 0.5, 1} and V clamped to [0,1], so a win is ALWAYS positive
    and a loss ALWAYS negative -- the critic can never flip the sign, and
    binary weighting consumes only the sign. That arm is therefore winners-only
    BC with the critic as dead code (audited 2026-08-15). The paper's filter
    needs an advantage the critic can actually decide.

    Non-terminal rows: `A = V(s') - V(s)` on the RAW critic outputs. Raw, not
    calibrated, on purpose -- Platt is monotone so it cannot change the sign of
    a difference, while its [0,1] clip WOULD tie every pair of out-of-range
    values to A = 0 and silently drop those rows (the raw critic is on record
    emitting out-of-range values on even midgames). gamma = 1: reward is
    terminal-only, so there is nothing to discount against.

    Terminal rows: `A = outcome - platt(V(s))`. Here the comparison is against
    an absolute {0, 0.5, 1} outcome, so the critic's SCALE matters and the
    calibration is load-bearing. These are ~1 row per seat-episode.
    """
    known = (outcome >= 0.0).float()
    term_adv = outcome - _platt(value, platt_a, platt_b)
    adv = torch.where(has_next.bool(), value_next - value, term_adv)
    return (adv > 0).float() * known, adv
