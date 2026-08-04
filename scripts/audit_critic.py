"""Three-part critic audit gate — rl_pipeline_v2.md §2.B2a.

A critic must pass ALL of this before any advantage arm (E1/E2/E3/E4) may
consume its advantages; a failing critic blocks those arms and the pipeline
falls back to the registered outcome-weighted arm, loudly.

(i)  V(s) predicts held-out outcomes better than the no-skill baselines:
     MSE vs the constant-0.5 predictor AND classification accuracy at the
     0.5 threshold vs the majority base rate.
(ii) Advantage distribution Â = outcome − V(s) on held-out rows: roughly
     centered at 0 with sane spread (mean/σ/quantiles reported; histogram
     figure split by win/loss).
(iii) The top-K decisions by |Â| dumped human-readably (state, options,
     chosen action, outcome, V, Â) — the sign must be defensible to a human
     reader. The script PRESENTS; the human judgment gets recorded in the
     report by whoever reads it.

Writes reports/critic_audit_<name>.md + reports/figures/critic_advantage_
hist_<name>.png and prints the (i)/(ii) verdicts.

Usage:
  uv run python scripts/audit_critic.py --critic-dir models/critic_td
  uv run python scripts/audit_critic.py --critic-dir models/critic_td \
      --max-episodes 800   # faster pass
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    _resolve_hub_shards,
    encode_observation,
    iter_episode_decisions,
)
from pokemon_tcg.offline_critic import load_critic  # noqa: E402

try:  # enum names make the hand-read section readable; ids still shown
    from cg.api import OptionType, SelectType  # noqa: E402

    def _opt_name(t: int) -> str:
        try:
            return OptionType(t).name
        except ValueError:
            return "DECLINE" if t == 17 else f"type{t}"

    def _sel_name(t: int) -> str:
        try:
            return SelectType(t).name
        except ValueError:
            return f"select{t}"
except Exception:  # pragma: no cover - cg-lib missing
    _opt_name = _sel_name = lambda t: str(t)  # noqa: E731

FEAT_KEYS = (
    "slot_card_id", "slot_scalar", "global_scalar", "select_type",
    "select_context", "opt_type", "opt_attack", "opt_special", "opt_scalar",
    "opt_ref_card_id", "opt_ref_scalar", "opt_mask",
)


def iter_hub_eval(days, max_episodes):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    files = _resolve_hub_shards(config.HF_EPISODES_REPO, "eval", days)
    n = 0
    for rel in files:
        pf = pq.ParquetFile(hf_hub_download(config.HF_EPISODES_REPO, rel, repo_type="dataset"))
        for rb in pf.iter_batches(batch_size=8, columns=["episode_id", "episode_json"]):
            for eid, raw in zip(rb.column(0).to_pylist(), rb.column(1).to_pylist(), strict=True):
                if max_episodes is not None and n >= max_episodes:
                    return
                n += 1
                yield int(eid), json.loads(raw)


def _describe_option(o: dict) -> str:
    bits = [_opt_name(o.get("type", -1))]
    for k in ("area", "index", "inPlayArea", "inPlayIndex", "playerIndex"):
        if o.get(k) is not None:
            bits.append(f"{k}={o[k]}")
    if o.get("attackId"):
        bits.append(f"attack={o['attackId']}")
    if o.get("number") is not None:
        bits.append(f"n={o['number']}")
    if o.get("specialConditionType") is not None:
        bits.append(f"special={o['specialConditionType']}")
    return "(" + " ".join(str(b) for b in bits) + ")"


def _describe_decision(obs: dict, label: int, meta, v: float, adv: float) -> str:
    sel = obs["select"]
    cur = obs.get("current") or {}
    me = (cur.get("players") or [{}, {}])[cur.get("yourIndex", 0)]
    opp = (cur.get("players") or [{}, {}])[1 - cur.get("yourIndex", 0)]
    opts = sel.get("option") or []
    n = len(opts)
    chosen = _describe_option(opts[label]) if label < n else "DECLINE (empty pick)"
    lines = [
        f"episode {meta.episode_id} seat {meta.seat} turn {meta.turn} | "
        f"outcome {meta.outcome:.1f}  V(s) {v:.3f}  Â {adv:+.3f}",
        f"  select {_sel_name(sel.get('type', -1))} ctx {sel.get('context')} "
        f"min {sel.get('minCount')} max {sel.get('maxCount')} | {n} options",
        f"  my prizes left {len(me.get('prize') or [])} deck {me.get('deckCount')} "
        f"hand {me.get('handCount')} | opp prizes {len(opp.get('prize') or [])} "
        f"deck {opp.get('deckCount')} hand {opp.get('handCount')}",
        f"  CHOSE [{label}] {chosen}",
    ]
    prev = "  options: " + " ".join(
        f"[{i}]{_describe_option(o)}" for i, o in enumerate(opts[:10])
    )
    if n > 10:
        prev += f" ... (+{n - 10} more)"
    lines.append(prev)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--critic-dir", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="cap eval episodes (default: full eval day)")
    ap.add_argument("--top-k", type=int, default=10, help="hand-read rows per sign")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from pokemon_tcg.device import resolve_device
    device = resolve_device(args.device)
    torch.set_num_threads(1) if device == "cpu" else None
    critic = load_critic(args.critic_dir, device=device)
    name = args.critic_dir.name

    se_model = se_const = 0.0
    n_rows = n_decisive = n_wins = correct = 0
    advantages: list[float] = []
    adv_win: list[float] = []
    adv_loss: list[float] = []
    # top-K most extreme |Â| rows, kept with printable context. heap of
    # (|adv|, counter, text) so ties never compare dicts.
    top: list = []
    counter = 0

    batch_feats, batch_ctx = [], []

    def flush():
        nonlocal se_model, se_const, n_rows, n_decisive, n_wins, correct, counter
        if not batch_feats:
            return
        batch = {k: torch.stack([f[k] for f in batch_feats]).to(device) for k in FEAT_KEYS}
        with torch.no_grad():
            v = critic(**batch).cpu().numpy()
        for (obs, label, meta), vi in zip(batch_ctx, v, strict=True):
            oc = meta.outcome
            if oc < 0:
                continue
            adv = oc - float(vi)
            advantages.append(adv)
            (adv_win if oc == 1.0 else adv_loss if oc == 0.0 else []).append(adv)
            se_model += (float(vi) - oc) ** 2
            se_const += (0.5 - oc) ** 2
            n_rows += 1
            if oc != 0.5:
                n_decisive += 1
                n_wins += int(oc == 1.0)
                correct += int((float(vi) > 0.5) == (oc == 1.0))
            counter += 1
            item = (abs(adv), counter,
                    _describe_decision(obs, label, meta, float(vi), adv))
            if len(top) < 4 * args.top_k:
                heapq.heappush(top, item)
            else:
                heapq.heappushpop(top, item)
        batch_feats.clear()
        batch_ctx.clear()

    n_eps = 0
    for eid, episode in iter_hub_eval(None, args.max_episodes):
        n_eps += 1
        for obs, label, exclude, meta in iter_episode_decisions(episode, episode_id=eid):
            feats = encode_observation(obs, exclude=exclude)
            if feats is None:
                continue
            feats.pop("n_real_options", None)
            batch_feats.append({k: feats[k] for k in FEAT_KEYS})
            batch_ctx.append((obs, label, meta))
            if len(batch_feats) >= args.batch_size:
                flush()
        if n_eps % 200 == 0:
            print(f"  ...{n_eps} episodes, {n_rows} rows", flush=True)
    flush()

    adv = np.array(advantages)
    eval_mse = se_model / max(n_rows, 1)
    const_mse = se_const / max(n_rows, 1)
    acc = correct / max(n_decisive, 1)
    base = max(n_wins, n_decisive - n_wins) / max(n_decisive, 1)
    q = np.percentile(adv, [1, 25, 50, 75, 99]) if len(adv) else [0] * 5

    pass_i = eval_mse < const_mse and acc > base
    # (ii) sanity: centered near 0 (|mean| under a quarter of σ, and σ neither
    # degenerate nor blown out for a bounded {0,0.5,1}-outcome problem)
    pass_ii = bool(len(adv)) and abs(adv.mean()) < 0.25 * max(adv.std(), 1e-9) \
        and 0.05 < adv.std() < 0.75

    extremes = sorted(top, key=lambda t: -t[0])
    # balance the hand-read set: top-k positive and top-k negative advantages
    pos = [t for t in extremes if "+" in t[2].split("Â ")[1][:1]][: args.top_k]
    neg = [t for t in extremes if "-" in t[2].split("Â ")[1][:1]][: args.top_k]

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(adv_win):
        ax.hist(adv_win, bins=60, alpha=0.6, label=f"wins (n={len(adv_win)})", color="#2a9d8f")
    if len(adv_loss):
        ax.hist(adv_loss, bins=60, alpha=0.6, label=f"losses (n={len(adv_loss)})", color="#e76f51")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("advantage  Â = outcome − V(s)")
    ax.set_title(f"Critic {name}: held-out advantage distribution")
    ax.legend()
    fig.tight_layout()
    fig_path = config.FIGURES_DIR / f"critic_advantage_hist_{name}.png"
    fig.savefig(fig_path, dpi=150)

    lines = [
        f"# Critic audit — {name}",
        "",
        f"Eval rows: {n_rows} ({n_eps} episodes, held-out day 2026-07-27)",
        "",
        "## (i) Outcome prediction vs no-skill baselines",
        f"- MSE {eval_mse:.4f} vs constant-0.5 {const_mse:.4f} -> "
        f"{'PASS' if eval_mse < const_mse else 'FAIL'}",
        f"- accuracy@0.5 {acc:.4f} vs majority base rate {base:.4f} "
        f"({n_decisive} decisive rows) -> {'PASS' if acc > base else 'FAIL'}",
        "",
        "## (ii) Advantage distribution (Â = outcome − V)",
        f"- mean {adv.mean():+.4f}  σ {adv.std():.4f}",
        f"- quantiles p1 {q[0]:+.3f} | p25 {q[1]:+.3f} | p50 {q[2]:+.3f} | "
        f"p75 {q[3]:+.3f} | p99 {q[4]:+.3f}",
        f"- verdict: {'PASS' if pass_ii else 'FAIL'} (centered-at-0 and sane-spread check)",
        f"- figure: {fig_path}",
        "",
        f"## (iii) Hand-read: top {args.top_k} positive and negative |Â| decisions",
        "Human judgment required — the SIGN of each Â below must be defensible.",
        "",
        "### Most positive advantages (critic says: better than expected)",
    ]
    for _, _, txt in pos:
        lines += [txt, ""]
    lines.append("### Most negative advantages (critic says: worse than expected)")
    for _, _, txt in neg:
        lines += [txt, ""]

    report = config.REPORTS_DIR / f"critic_audit_{name}.md"
    report.write_text("\n".join(lines))

    print("\n".join(lines[:16]))
    print(f"\nfull report: {report}")
    print(f"AUDIT (i): {'PASS' if pass_i else 'FAIL'}   (ii): "
          f"{'PASS' if pass_ii else 'FAIL'}   (iii): read {report}")


if __name__ == "__main__":
    main()
