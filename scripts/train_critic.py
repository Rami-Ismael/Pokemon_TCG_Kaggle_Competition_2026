"""Train the Stage-2 state-value critic V(s) — rl_pipeline_v2.md §2.B2a.

The critic behind the advantage arms (E1 binary, E2 exp): warm-start the
trunk from the frozen PRIOR actor, put a scalar head on cls_hidden, and fit
the remapped terminal outcome {0, 0.5, 1}. TRAIN-TIME ONLY — never ships in
a bundle. Consumed by scripts/train_il.py --weight-arm adv-binary|adv-exp
and audited by scripts/audit_critic.py before any arm may use it.

Two targets (--target):

* ``td`` (default, the v2-registered method): one-step TD backups at the
  decision level — for each seat's decision sequence s_0..s_{T-1},
  V(s_t) <- td_gamma * V(s_{t+1}) for t < T-1 (bootstrap, detached) and
  V(s_{T-1}) <- outcome. Episodes with unknown outcome are skipped whole
  (their terminal target would be garbage). ``--td-gamma`` defaults to 1.0:
  the arms consume advantage = outcome - V(s), so V must live on the
  outcome's undiscounted scale — a td_gamma < 1 would systematically
  shrink early-game values and masquerade as advantage.
* ``mc`` (ablation, the original method): regress V(s_t) = outcome for
  every t. With td_gamma = 1 the two share a fixed point in expectation
  (TD trades variance for bootstrap bias); a material disagreement on the
  audit is itself a finding.

Data comes from the ADR-001 Hub shards by default (the raw local corpus was
deleted 2026-08-03); --data-source local remains for smoke tests against
the surviving files.

Quality gate printed at the end (a number without a control is not a
result): eval MSE vs the constant-0.5 predictor, and outcome-classification
accuracy at the 0.5 threshold vs the majority base rate. The full three-part
audit (base rate, advantage distribution, hand-read extremes) lives in
scripts/audit_critic.py.

Usage:
  uv run python scripts/train_critic.py --out models/critic_td \
      --hub-days 2026-07-01,2026-07-26 --num-workers 4
  uv run python scripts/train_critic.py --target mc --out models/critic_mc
  uv run python scripts/train_critic.py --dry-run   # CPU-sized smoke
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import resolve_device  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    ILDataset,
    ShardILDataset,
    _decisions_to_features,
    _resolve_hub_shards,
    _shuffled,
    iter_episode_decisions,
    resolve_split_dir,
)
from pokemon_tcg.il_model import PTCGILConfig  # noqa: E402
from pokemon_tcg.offline_critic import OfflineValueModel, save_critic  # noqa: E402

ROWS_PER_EPISODE = 181.3  # same measured constant train_il.py schedules with

# Model-input feature keys (everything encode_observation emits except the
# n_real_options bookkeeping field, which _decisions_to_features pops).
FEAT_KEYS = (
    "slot_card_id", "slot_scalar", "global_scalar", "select_type",
    "select_context", "opt_type", "opt_attack", "opt_special", "opt_scalar",
    "opt_ref_card_id", "opt_ref_scalar", "opt_mask",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TDPairDataset(IterableDataset):
    """Streams (s_t, s_{t+1}, terminal, outcome) pairs per seat-trajectory.

    Rows carry the s_t features under their own keys, the successor features
    under ``next_<key>`` (zeros when terminal — masked out of the target),
    plus ``terminal`` and ``outcome``. Shard/worker splitting mirrors
    ShardILDataset. Episodes with unknown outcome are skipped whole.
    """

    def __init__(self, split: str, days: list[str] | None = None,
                 local_dir: Path | None = None, max_episodes: int | None = None,
                 shuffle_buffer: int = 2000, seed: int = 0) -> None:
        self.split = split
        self.local_dir = local_dir
        self.max_episodes = max_episodes
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self._epoch = 0
        if local_dir is None:
            self.files = _resolve_hub_shards(config.HF_EPISODES_REPO, split, days)
            if not self.files:
                raise FileNotFoundError(f"no Hub shards for {split!r} days {days}")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _episodes(self, files):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        n = 0
        for rel in files:
            pf = pq.ParquetFile(hf_hub_download(config.HF_EPISODES_REPO, rel,
                                                repo_type="dataset"))
            for rb in pf.iter_batches(batch_size=8,
                                      columns=["episode_id", "episode_json"]):
                for eid, raw in zip(rb.column(0).to_pylist(),
                                    rb.column(1).to_pylist(), strict=True):
                    if self.max_episodes is not None and n >= self.max_episodes:
                        return
                    n += 1
                    yield int(eid), json.loads(raw)

    def _local_episodes(self):
        n = 0
        for path in sorted(self.local_dir.glob("*.json")):
            if not path.exists():
                continue  # dangling symlink from the 08-03 cleanup
            if self.max_episodes is not None and n >= self.max_episodes:
                return
            n += 1
            try:
                episode = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            yield (int(path.stem) if path.stem.isdigit() else -1), episode

    def _pairs(self, episodes):
        for eid, episode in episodes:
            # Group this episode's rows per seat (trajectory order is exactly
            # iteration order: iter_episode_decisions walks seat 0 fully,
            # then seat 1). ~181 rows/episode total -- tiny to buffer.
            by_seat: dict[int, list] = {}
            outcome_by_seat: dict[int, float] = {}
            for feats in _decisions_to_features(
                iter_episode_decisions(episode, episode_id=eid), with_meta=True
            ):
                seat = int(feats["seat"])
                outcome_by_seat[seat] = float(feats["outcome"])
                by_seat.setdefault(seat, []).append(
                    {k: feats[k] for k in FEAT_KEYS}
                )
            for seat, rows in by_seat.items():
                outcome = outcome_by_seat[seat]
                if outcome < 0.0:
                    continue  # unknown outcome: terminal target undefined
                zeros = {k: torch.zeros_like(v) for k, v in rows[0].items()}
                for t, row in enumerate(rows):
                    terminal = t == len(rows) - 1
                    nxt = rows[t + 1] if not terminal else zeros
                    out = dict(row)
                    out.update({f"next_{k}": v for k, v in nxt.items()})
                    out["terminal"] = torch.tensor(terminal)
                    out["outcome"] = torch.tensor(outcome, dtype=torch.float32)
                    yield out

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid, n_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = random.Random((self.seed * 1_000_003 + self._epoch) * 64 + wid)
        if self.local_dir is not None:
            assert n_workers == 1, "local TD source has no worker sharding"
            episodes = self._local_episodes()
        else:
            files = list(self.files)
            if self.shuffle_buffer > 1:
                random.Random(self.seed * 1_000_003 + self._epoch).shuffle(files)
            episodes = self._episodes(files[wid::n_workers])
        yield from _shuffled(self._pairs(episodes), self.shuffle_buffer, rng)


def _feats_of(batch: dict, device: str, prefix: str = "") -> dict:
    return {k: batch[prefix + k].to(device) for k in FEAT_KEYS}


@torch.no_grad()
def evaluate(critic, eval_loader, device, max_batches: int) -> dict:
    """Held-out MSE + accuracy vs the two no-skill baselines."""
    critic.eval()
    se_model, se_const, n_rows = 0.0, 0.0, 0
    correct, n_decisive, n_wins = 0, 0, 0
    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        outcome = batch["outcome"].to(device)
        known = outcome >= 0.0
        if not bool(known.any()):
            continue
        pred = critic(**_feats_of(batch, device))[known]
        oc = outcome[known]
        se_model += float(((pred - oc) ** 2).sum())
        se_const += float(((0.5 - oc) ** 2).sum())
        n_rows += int(known.sum())
        decisive = oc != 0.5
        n_decisive += int(decisive.sum())
        n_wins += int((oc[decisive] == 1.0).sum())
        correct += int(((pred[decisive] > 0.5) == (oc[decisive] == 1.0)).sum())
    critic.train()
    base_rate = max(n_wins, n_decisive - n_wins) / max(n_decisive, 1)
    return {
        "eval_mse": se_model / max(n_rows, 1),
        "constant_baseline_mse": se_const / max(n_rows, 1),
        "accuracy_at_0.5": correct / max(n_decisive, 1),
        "majority_base_rate": base_rate,
        "n_rows": n_rows,
        "n_decisive": n_decisive,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-from", type=Path, default=config.MODELS_DIR / "il_agent",
                    help="actor checkpoint to warm-start the trunk from (the PRIOR)")
    ap.add_argument("--target", choices=["td", "mc"], default="td",
                    help="td = one-step TD backups (v2-registered); mc = direct "
                         "outcome regression (ablation, the original method)")
    ap.add_argument("--td-gamma", type=float, default=1.0,
                    help="bootstrap discount for --target td. Keep 1.0: the "
                         "arms consume outcome - V(s), so V must be on the "
                         "outcome's undiscounted scale")
    ap.add_argument("--data-source", choices=["auto", "local", "hub"], default="auto")
    ap.add_argument("--hub-days", default="2026-07-01,2026-07-26",
                    help="comma-separated train days (hub source)")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--max-train-episodes", type=int, default=None)
    ap.add_argument("--max-eval-episodes", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers over the shard stream (hub only)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", type=Path, default=config.MODELS_DIR / "critic_td")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="fresh tiny trunk, 40 train / 15 eval episodes -- smoke test")
    args = ap.parse_args()

    if args.dry_run:
        args.max_train_episodes = args.max_train_episodes or 40
        args.max_eval_episodes = args.max_eval_episodes or 15
        args.epochs = 1
        args.eval_batches = 20
        args.log_every = 20
        args.num_workers = 0

    set_seed(args.seed)
    device = resolve_device(args.device)
    hub_days = args.hub_days.split(",") if args.hub_days else None

    source = args.data_source
    if source == "auto":
        try:
            d = resolve_split_dir(args.train_split)
            source = "local" if any(p.exists() for p in d.glob("*.json")) and (
                args.max_train_episodes or 0) <= sum(
                1 for p in d.glob("*.json") if p.exists()) else "hub"
        except (FileNotFoundError, KeyError):
            source = "hub"

    if args.dry_run:
        critic = OfflineValueModel.from_config(
            PTCGILConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2)
        )
        init_desc = "fresh tiny trunk (dry run)"
    else:
        critic = OfflineValueModel.from_actor_checkpoint(args.init_from)
        init_desc = str(args.init_from)
    critic.to(device).train()
    n_params = sum(p.numel() for p in critic.parameters())
    print(f"critic target: {args.target} (td_gamma {args.td_gamma})  trunk init: "
          f"{init_desc}  params: {n_params / 1e6:.2f}M  device: {device}  "
          f"source: {source}", flush=True)

    # ---- train stream -----------------------------------------------------
    if args.target == "td":
        if source == "hub":
            train_ds = TDPairDataset("train", days=hub_days, seed=args.seed,
                                     max_episodes=args.max_train_episodes)
            n_episodes = ShardILDataset("train", days=hub_days).n_episodes()
        else:
            train_ds = TDPairDataset("train", local_dir=resolve_split_dir(args.train_split),
                                     seed=args.seed, max_episodes=args.max_train_episodes)
            n_episodes = sum(1 for p in resolve_split_dir(args.train_split).glob("*.json")
                             if p.exists())
            args.num_workers = 0
    else:
        if source == "hub":
            train_ds = ShardILDataset("train", days=hub_days, seed=args.seed,
                                      max_episodes=args.max_train_episodes, with_meta=True)
            n_episodes = train_ds.n_episodes()
        else:
            train_ds = ILDataset(resolve_split_dir(args.train_split),
                                 max_episodes=args.max_train_episodes,
                                 seed=args.seed, with_meta=True)
            n_episodes = sum(1 for p in resolve_split_dir(args.train_split).glob("*.json")
                             if p.exists())
            args.num_workers = 0

    # ---- eval stream (deterministic, both-seats, meta for outcome) --------
    if source == "hub":
        eval_ds = ShardILDataset("eval", max_episodes=args.max_eval_episodes,
                                 shuffle_buffer=1, with_meta=True)
    else:
        eval_ds = ILDataset(resolve_split_dir(args.eval_split),
                            max_episodes=args.max_eval_episodes,
                            shuffle_buffer=1, with_meta=True)

    if args.max_train_episodes:
        n_episodes = min(n_episodes, args.max_train_episodes)
    total_steps = args.total_steps or max(
        int(n_episodes * ROWS_PER_EPISODE / args.batch_size * args.epochs), 1
    )
    print(f"schedule: {total_steps} steps ({n_episodes} episodes x "
          f"{ROWS_PER_EPISODE} rows/ep / batch {args.batch_size} x {args.epochs})",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              num_workers=args.num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr)
    step, t0 = 0, time.time()
    running, running_n = 0.0, 0
    pass_idx = 0
    while step < total_steps:
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(pass_idx)
        pass_idx += 1
        for batch in train_loader:
            if step >= total_steps:
                break
            if args.target == "td":
                v = critic(**_feats_of(batch, device))
                with torch.no_grad():
                    vn = critic(**_feats_of(batch, device, prefix="next_"))
                terminal = batch["terminal"].to(device)
                outcome = batch["outcome"].to(device)
                target = torch.where(terminal, outcome, args.td_gamma * vn)
                loss = ((v - target) ** 2).mean()
            else:
                outcome = batch.pop("outcome").to(device)
                known = outcome >= 0.0
                if not bool(known.any()):
                    continue
                v = critic(**_feats_of(batch, device))
                loss = ((v[known] - outcome[known]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item()
            running_n += 1
            step += 1
            if step % args.log_every == 0:
                print(f"step {step}/{total_steps} train_mse={running / max(running_n, 1):.4f} "
                      f"({step / (time.time() - t0):.2f} steps/s)", flush=True)
                running, running_n = 0.0, 0

    res = evaluate(critic, eval_loader, device, args.eval_batches)
    beats_mse = res["eval_mse"] < res["constant_baseline_mse"]
    beats_acc = res["accuracy_at_0.5"] > res["majority_base_rate"]
    print(f"eval -- critic_mse={res['eval_mse']:.4f} vs constant-0.5 "
          f"{res['constant_baseline_mse']:.4f} ({res['n_rows']} rows) "
          f"{'PASS' if beats_mse else 'FAIL'}")
    print(f"eval -- accuracy@0.5={res['accuracy_at_0.5']:.4f} vs majority base rate "
          f"{res['majority_base_rate']:.4f} ({res['n_decisive']} decisive rows) "
          f"{'PASS' if beats_acc else 'FAIL'}")

    save_critic(critic, args.out)
    (args.out / "train_metadata.json").write_text(json.dumps({
        "step": step,
        "target": args.target,
        "td_gamma": args.td_gamma,
        "init_from": init_desc,
        "data_source": source,
        "hub_days": hub_days,
        "seed": args.seed,
        **res,
        "beats_constant_baseline": beats_mse,
        "beats_majority_accuracy": beats_acc,
    }, indent=2))
    print(f"critic saved to {args.out}")


if __name__ == "__main__":
    main()
