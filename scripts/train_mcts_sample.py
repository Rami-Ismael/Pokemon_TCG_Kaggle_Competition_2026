"""Run kiyotah's AlphaZero-miniature training loop (faithful port).

Per iteration: (1) save current weights, (2) evaluate MCTS agent vs a
uniform-random agent, (3) self-play games collecting LearnSamples with
TD(lambda=0.9) value labels and child-minus-root policy labels, (4) one pass
of Huber-loss training. Hyperparameters are the notebook's exactly
(MyModel(128,2,256,1,1), AdamW lr 3e-4, BATCH_SIZE 128, 50 eval / 100
self-play games, SEARCH_COUNT 10).

Parity gate for Phase 1: eval win rate vs random clearly above 50% after a
few iterations. This is a TUTORIAL-SCALE agent (10 sims, one fixed deck,
random opponent) -- its value is API mastery, not ladder strength.

Usage:
    uv run python scripts/train_mcts_sample.py --iterations 5
    PTCG_DEVICE=cpu uv run python scripts/train_mcts_sample.py  # evaluator rehearsal
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import log_device_info, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=5, help="train/eval iterations (notebook: 5)")
    p.add_argument("--eval-games", type=int, default=50, help="eval games vs random per iteration (notebook: 50)")
    p.add_argument("--selfplay-games", type=int, default=100, help="self-play games per iteration (notebook: 100)")
    p.add_argument("--search-count", type=int, default=10, help="MCTS simulations per move (notebook: 10)")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--device", type=str, default=None, help="mps|cpu; default resolve_device()")
    p.add_argument("--out-dir", type=Path, default=config.MODELS_DIR / "mcts_sample")
    p.add_argument("--tag", type=str, default="", help="suffix for out dir / report (e.g. seed sweep)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_str = resolve_device(args.device)
    device = torch.device(device_str)
    if device_str == "cpu":
        torch.set_num_threads(1)  # evaluator envelope: set BEFORE any heavy torch work

    # search_api imports cg (ctypes lib load + card table) at import time;
    # do it after thread setup.
    from pokemon_tcg import search_api as sa

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = args.out_dir if not args.tag else Path(str(args.out_dir) + "_" + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_device_info(out_dir, device_str)

    model = sa.MyModel(128, 2, 256, 1, 1)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device_str} params={n_params:,} out={out_dir}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function

    deck = sa.sample_deck
    history: list[dict] = []
    report_path = out_dir / "train_report.json"

    for counter in range(args.iterations):
        torch.save(model.state_dict(), out_dir / f"model{counter}.pth")
        sample_list: list[sa.LearnSample] = []

        model.eval()
        iter_t0 = time.time()
        with torch.inference_mode():
            # Evaluation vs uniform-random agent
            results = [0, 0, 0]  # win, lose, draw
            for i in range(args.eval_games):
                obs, start_data = sa.battle_start(deck, deck)
                if start_data.errorPlayer >= 0:
                    raise ValueError(f"Deck error type {start_data.errorType}")
                your_index = i % 2
                while obs["current"]["result"] < 0:
                    if obs["current"]["yourIndex"] == your_index:
                        selected, _ = sa.mcts_agent(obs, deck, model, args.search_count)
                    else:
                        selected = sa.random_agent(obs)
                    obs = sa.battle_select(selected)
                sa.battle_finish()

                if obs["current"]["result"] == 2:
                    results[2] += 1
                elif obs["current"]["result"] == your_index:
                    results[0] += 1
                else:
                    results[1] += 1
            decided = results[0] + results[1]
            win_rate = results[0] / decided if decided else float("nan")
            # Binomial sigma on the win rate (decided games only).
            sigma = (win_rate * (1 - win_rate) / decided) ** 0.5 if decided else float("nan")
            eval_secs = time.time() - iter_t0
            print(
                f"iter {counter}: eval {results[0]}W-{results[1]}L-{results[2]}D "
                f"win_rate={win_rate:.1%} sigma={sigma:.1%} ({eval_secs:.0f}s)",
                flush=True,
            )

            # Self Play
            sp_t0 = time.time()
            for _ in range(args.selfplay_games):
                obs, _ = sa.battle_start(deck, deck)
                samples: list[list[sa.LearnSample]] = [[], []]
                while obs["current"]["result"] < 0:
                    selected, sample = sa.mcts_agent(obs, deck, model, args.search_count)
                    samples[obs["current"]["yourIndex"]].append(sample)
                    obs = sa.battle_select(selected)
                sa.battle_finish()

                for i in range(2):
                    LAMBDA = 0.9
                    value = 1.0 if i == obs["current"]["result"] else -1.0
                    for sample in reversed(samples[i]):
                        label = (value + sample.value) * 0.5
                        value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
                        sample.value = label
                        sample_list.append(sample)
            selfplay_secs = time.time() - sp_t0
            print(
                f"iter {counter}: self-play {args.selfplay_games} games -> "
                f"{len(sample_list)} samples ({selfplay_secs:.0f}s)",
                flush=True,
            )

        # Train on collected self-play data.
        train_t0 = time.time()
        model.train()
        random.shuffle(sample_list)
        BATCH_SIZE = 128
        batch_count = len(sample_list) // BATCH_SIZE
        last_loss = float("nan")
        for i in range(batch_count):
            input_enc = sa.LearnInput()
            input_dec = sa.LearnInput()
            mask = []
            label_enc = []
            label_dec = []
            start = BATCH_SIZE * i
            for j in range(start, start + BATCH_SIZE):
                sample = sample_list[j]
                input_enc.add(sample.sv_enc)
                input_dec.add(sample.sv_dec)
                label_enc.append(sample.value)
                label_dec.extend(sample.policy)
                for _ in range(len(sample.policy)):
                    mask.append(1.0)
                for _ in range(sa.MAX_ACTION_COMBOS - len(sample.policy)):
                    mask.append(0.0)
                    label_dec.append(0.0)
                    input_dec.offset.append(len(input_dec.index))

            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)
            label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)
            label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)

            optimizer.zero_grad()
            out_enc, out_dec = model(
                torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
            )
            loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
            loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
            loss_dec = loss_dec * mask_tensor
            loss_dec = loss_dec.sum() / float(BATCH_SIZE)
            loss = loss_enc + loss_dec
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
        train_secs = time.time() - train_t0
        print(f"iter {counter}: trained {batch_count} batches, last loss {last_loss:.4f} ({train_secs:.0f}s)", flush=True)

        history.append(
            {
                "iteration": counter,
                "eval": {"win": results[0], "lose": results[1], "draw": results[2],
                         "win_rate": win_rate, "sigma": sigma, "secs": eval_secs},
                "selfplay": {"games": args.selfplay_games, "samples": len(sample_list),
                             "secs": selfplay_secs},
                "train": {"batches": batch_count, "last_loss": last_loss, "secs": train_secs},
            }
        )
        report_path.write_text(json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()},
             "device": device_str, "params": n_params, "history": history}, indent=2))

    torch.save(model.state_dict(), out_dir / "model_final.pth")
    print(f"done; report at {report_path}", flush=True)


if __name__ == "__main__":
    main()
