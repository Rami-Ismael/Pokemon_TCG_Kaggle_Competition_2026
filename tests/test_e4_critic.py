"""Tests for the S2-E4 critic-advantage build (notes/test_plan_e4_critic.md).

Covers: advantage weight math (U1-U4), critic isolation and persistence
(U5, U6, U8), weighted-loss normalization (U7), and the three integration
smokes (I1-I3: train_critic dry-run, train_il adv arm dry-run, train_il
default-arm regression -- the queued approved runs use that same file).

Run directly: uv run python tests/test_e4_critic.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pokemon_tcg.il_dataset import ILDataset  # noqa: E402
from pokemon_tcg.il_model import PTCGILConfig, PTCGImitationPolicy  # noqa: E402
from pokemon_tcg.offline_critic import (  # noqa: E402
    OfflineValueModel,
    advantage_weights,
    load_critic,
    save_critic,
)

TRAIN_DIR = REPO / "data" / "episodes" / "splits" / "train-2026-07-26"
TINY = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2)


def _tiny_critic() -> OfflineValueModel:
    torch.manual_seed(0)
    return OfflineValueModel.from_config(PTCGILConfig(**TINY))


def _real_batch(batch_size: int = 16) -> dict:
    ds = ILDataset(TRAIN_DIR, max_episodes=3, shuffle_buffer=1, with_meta=True)
    return next(iter(DataLoader(ds, batch_size=batch_size)))


def _feats(batch: dict) -> dict:
    drop = set(ILDataset.META_KEYS) | {"label"}
    return {k: v for k, v in batch.items() if k not in drop}


# --- U1/U4: weight math -----------------------------------------------------

def test_advantage_weight_math():
    outcome = torch.tensor([1.0, 0.0, 0.5, 1.0])
    value = torch.tensor([0.2, 0.6, 0.5, 0.9])
    beta = 2.0
    w = advantage_weights("adv-exp", outcome, value, beta=beta, clip=20.0)
    expected = torch.exp(beta * (outcome - value))
    assert torch.allclose(w, expected), f"{w} != {expected}"
    # clip engages
    w_clipped = advantage_weights("adv-exp", torch.tensor([1.0]), torch.tensor([-3.0]),
                                  beta=2.0, clip=5.0)
    assert float(w_clipped) == 5.0
    wb = advantage_weights("adv-binary", outcome, value)
    assert wb.tolist() == [1.0, 0.0, 0.0, 1.0]  # A=0 (draw at V=0.5) is NOT > 0
    try:
        advantage_weights("bogus", outcome, value)
        raise AssertionError("expected ValueError for unknown arm")
    except ValueError:
        pass
    print("  adv-exp matches exp(beta*A) with clip; adv-binary is strict 1[A>0]")


# --- U2: the correctness anchor --------------------------------------------

def test_constant_baseline_equivalence():
    """With V forced to 0.5, the new arms must collapse EXACTLY to the two
    already-trained-and-benchmarked arms: adv-exp == the existing 'outcome'
    weights, adv-binary == the winners-only row selection."""
    outcome = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0, -1.0])
    v_const = torch.full_like(outcome, 0.5)
    beta = 1.0
    w_adv = advantage_weights("adv-exp", outcome, v_const, beta=beta, clip=20.0)
    w_outcome_arm = torch.exp(beta * (outcome - 0.5)) * (outcome >= 0.0)  # train_il.py's E2
    assert torch.allclose(w_adv, w_outcome_arm), f"{w_adv} != {w_outcome_arm}"
    w_bin = advantage_weights("adv-binary", outcome, v_const)
    winners_only = (outcome == 1.0).float()  # E1's row selection
    assert torch.equal(w_bin, winners_only), f"{w_bin} != {winners_only}"
    print("  V=0.5 collapses adv-exp -> E2 weights and adv-binary -> E1 selection")


# --- U3: unknown outcome ----------------------------------------------------

def test_unknown_outcome_gets_zero_weight():
    outcome = torch.tensor([-1.0, -1.0])
    for arm in ("adv-exp", "adv-binary"):
        for v in (torch.tensor([0.0, 0.0]), torch.tensor([-5.0, 5.0])):
            w = advantage_weights(arm, outcome, v)
            assert torch.equal(w, torch.zeros(2)), f"{arm} with V={v}: {w}"
    print("  -1 sentinel rows get weight 0 in both arms regardless of V")


# --- U5: shape + no parameter aliasing with a live actor --------------------

def test_value_model_isolated_from_actor():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        actor = PTCGImitationPolicy(PTCGILConfig(**TINY))
        actor.save_pretrained(td)
        critic = OfflineValueModel.from_actor_checkpoint(td)
        actor2 = PTCGImitationPolicy.from_pretrained(td)
        actor_ids = {id(p) for p in actor2.parameters()}
        shared = [n for n, p in critic.named_parameters() if id(p) in actor_ids]
        assert not shared, f"critic shares tensors with actor: {shared}"
    batch = _real_batch(8)
    v = critic(**_feats(batch))
    assert v.shape == (batch["label"].shape[0],), f"expected [B], got {tuple(v.shape)}"
    print(f"  scalar per row ({tuple(v.shape)}), zero shared tensors with the actor")


# --- U6: save/load round-trip ----------------------------------------------

def test_critic_save_load_roundtrip():
    critic = _tiny_critic().eval()
    batch = _feats(_real_batch(8))
    with torch.no_grad():
        before = critic(**batch)
    with tempfile.TemporaryDirectory() as td:
        save_critic(critic, td)
        loaded = load_critic(td)
        with torch.no_grad():
            after = loaded(**batch)
        assert not any(p.requires_grad for p in loaded.parameters()), "loaded critic not frozen"
    assert torch.allclose(before, after, atol=1e-6), "round-trip changed outputs"
    print("  save -> load reproduces outputs exactly; loaded critic is frozen")


# --- U7: normalization ------------------------------------------------------

def test_equal_weights_reduce_to_unweighted_mean():
    torch.manual_seed(0)
    per_row = torch.rand(32)
    w = torch.full((32,), 3.7)
    weighted = (w * per_row).sum() / w.sum().clamp_min(1e-8)
    assert torch.allclose(weighted, per_row.mean(), atol=1e-6)
    print("  sum(w*ce)/sum(w) with equal weights == plain mean CE")


# --- U8: meta keys cannot reach the critic ----------------------------------

def test_critic_rejects_meta_keys():
    critic = _tiny_critic()
    batch = _real_batch(4)
    try:
        critic(**_feats(batch), outcome=batch["outcome"])
        raise AssertionError("critic accepted a meta key -- structural guard lost")
    except TypeError:
        pass
    print("  passing outcome= to the critic raises TypeError (encoder feats only)")


# --- I1-I3: integration smokes (subprocess, CPU-forced) ---------------------

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, timeout=900,
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
             "PTCG_DEVICE": "cpu", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"},
    )


def test_train_critic_dry_run_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "critic"
        r = _run([sys.executable, "scripts/train_critic.py", "--dry-run",
                  "--out", str(out), "--device", "cpu"])
        assert r.returncode == 0, f"train_critic failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        meta = json.loads((out / "train_metadata.json").read_text())
        assert (out / "critic_state.pt").exists() and (out / "config.json").exists()
        loaded = load_critic(out)
        v = loaded(**_feats(_real_batch(4)))
        assert torch.isfinite(v).all()
        print(f"  dry-run critic trained ({meta['step']} steps), saved, reloadable; "
              f"eval_mse={meta['eval_mse']:.4f} vs const {meta['constant_baseline_mse']:.4f}")


def test_train_il_adv_arm_dry_run():
    with tempfile.TemporaryDirectory() as td:
        critic_dir = Path(td) / "critic"
        save_critic(_tiny_critic(), critic_dir)
        out = Path(td) / "policy"
        r = _run([sys.executable, "scripts/train_il.py", "--dry-run",
                  "--weight-arm", "adv-exp", "--critic-dir", str(critic_dir),
                  "--out", str(out), "--run-dir", str(Path(td) / "run"),
                  "--device", "cpu"])
        assert r.returncode == 0, f"adv-exp arm failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        assert "critic loaded" in r.stdout
        assert (out / "model.safetensors").exists(), "no checkpoint saved"
        print("  train_il --weight-arm adv-exp ran end-to-end and checkpointed")


def test_train_il_default_arm_regression():
    """The queued approved runs execute this exact file with the default arm --
    the E4 edit must not have disturbed that path."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "policy"
        r = _run([sys.executable, "scripts/train_il.py", "--dry-run",
                  "--out", str(out), "--run-dir", str(Path(td) / "run"),
                  "--device", "cpu"])
        assert r.returncode == 0, f"default arm broke:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        assert (out / "model.safetensors").exists()
        print("  default (plain BC) path unchanged and green")


if __name__ == "__main__":
    tests = [
        test_advantage_weight_math,
        test_constant_baseline_equivalence,
        test_unknown_outcome_gets_zero_weight,
        test_value_model_isolated_from_actor,
        test_critic_save_load_roundtrip,
        test_equal_weights_reduce_to_unweighted_mean,
        test_critic_rejects_meta_keys,
        test_train_critic_dry_run_end_to_end,
        test_train_il_adv_arm_dry_run,
        test_train_il_default_arm_regression,
    ]
    for t in tests:
        print(f"running {t.__name__}...")
        t()
        print("  OK")
    print("\nall E4 tests passed")
