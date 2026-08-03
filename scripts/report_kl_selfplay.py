"""Build the KL-anchored self-play run report (self-contained HTML).

Reads a train_ppo_puffer.py run dir (train_metrics.jsonl + promotion_log.jsonl)
plus the optional anchor-on/off A/B run dirs and a fallback-diagnostic JSON,
and writes one dependency-free HTML dashboard: KPI tiles, the KL-drift line
with promotion-gate markers (the ratchet's sawtooth), per-gate win-rate columns
against the promotion threshold, the throughput A/B, and training-health small
multiples. Charts are inline SVG in the repo's dataviz conventions (validated
categorical slots, hairline grid, crosshair tooltips, table view).

    uv run python scripts/report_kl_selfplay.py --run models/ppo_kl_run1 \
        --ab-off models/ppo_ab_off --ab-on models/ppo_ab_on \
        --fallback reports/fallback_diag_klanchor.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def marginal_sps(rows: list[dict]) -> list[dict]:
    """Per-update steps/s from ts deltas (drops update 1: worker startup)."""
    out = []
    for prev, cur in zip(rows, rows[1:], strict=False):
        dt = cur["ts"] - prev["ts"]
        ds = cur["global_step"] - prev["global_step"]
        if dt > 0 and ds > 0:
            out.append({"epoch": cur["epoch"], "sps": ds / dt})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=config.MODELS_DIR / "ppo_kl_run1")
    ap.add_argument("--ab-off", type=Path, default=None)
    ap.add_argument("--ab-on", type=Path, default=None)
    ap.add_argument("--fallback", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=config.REPORTS_DIR / "kl_selfplay_report.html")
    args = ap.parse_args()

    metrics = read_jsonl(args.run / "train_metrics.jsonl")
    gates = read_jsonl(args.run / "promotion_log.jsonl")
    if not metrics:
        raise SystemExit(f"no train_metrics.jsonl under {args.run}")

    ab = {}
    for name, d in (("off", args.ab_off), ("on", args.ab_on)):
        if d is not None:
            rows = read_jsonl(d / "train_metrics.jsonl")
            m = marginal_sps(rows)
            ab[name] = {
                "marginal": m,
                "mean": sum(r["sps"] for r in m) / len(m) if m else None,
                "total_steps": rows[-1]["global_step"] if rows else 0,
            }

    fallback = None
    if args.fallback is not None and args.fallback.exists():
        fb = json.loads(args.fallback.read_text())
        agg = fb.get("fallback", fb)  # fallback_diagnostic.py layout
        fallback = {
            "decisions": agg.get("decisions"),
            "fallbacks": agg.get("fallbacks"),
        }

    run_tag = metrics[-1].get("run_tag", args.run.name)
    threshold = gates[0]["threshold"] if gates else 0.70
    data = {
        "run_tag": run_tag,
        "run_dir": str(args.run),
        "kl_coef": _run_kl_coef(args.run),
        "threshold": threshold,
        "metrics": [
            {k: r.get(k) for k in ("epoch", "global_step", "kl_to_prior", "sps",
                                    "entropy", "explained_variance", "ref")}
            for r in metrics
        ],
        "gates": [
            {k: g.get(k) for k in ("epoch", "global_step", "wins", "losses",
                                    "draws", "decisive", "win_rate", "promote",
                                    "seconds", "min_decisive", "new_ref")}
            for g in gates
        ],
        "ab": ab,
        "fallback": fallback,
    }

    html = build_html(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"report -> {args.out} ({len(html) / 1024:.0f} KiB; "
          f"{len(metrics)} updates, {len(gates)} gates)")


def _run_kl_coef(run_dir: Path) -> float | None:
    for meta in sorted(run_dir.glob("*/ppo_metadata.json")):
        try:
            return json.loads(meta.read_text()).get("kl_coef")
        except (json.JSONDecodeError, OSError):
            continue
    return None


def build_html(data: dict) -> str:
    payload = json.dumps(data).replace("</", "<\\/")
    # The CSS/JS below follows the dataviz skill's reference palette and mark
    # specs: categorical slots 1-2 (validated both modes), hairline grid,
    # 2px lines, >=8px ring-carrying markers, crosshair tooltips, table view.
    return HTML_TEMPLATE.replace("__PAYLOAD__", payload)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KL-anchored self-play report</title>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --ink-1: #0b0b0b; --ink-2: #52514e; --ink-3: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7;
    --series-1: #2a78d6; --series-2: #eb6834;
    --good-text: #006300;
    --border: rgba(11,11,11,0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d;
      --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-3: #898781;
      --grid: #2c2c2a; --baseline: #383835;
      --series-1: #3987e5; --series-2: #d95926;
      --good-text: #0ca30c;
      --border: rgba(255,255,255,0.10);
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-3: #898781;
    --grid: #2c2c2a; --baseline: #383835;
    --series-1: #3987e5; --series-2: #d95926;
    --good-text: #0ca30c;
    --border: rgba(255,255,255,0.10);
  }
  .viz-root {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page); color: var(--ink-1);
    margin: 0; padding: 24px 16px 48px;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--ink-2); font-size: 13px; margin: 0 0 20px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 12px; margin-bottom: 20px; }
  .tile { background: var(--surface-1); border: 1px solid var(--border);
          border-radius: 8px; padding: 12px 14px; }
  .tile .label { font-size: 12px; color: var(--ink-2); }
  .tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
  .tile .note { font-size: 11px; color: var(--ink-3); margin-top: 2px; }
  .panel { background: var(--surface-1); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px 16px 8px; margin-bottom: 16px; }
  .panel h2 { font-size: 14px; margin: 0 0 2px; }
  .panel .psub { font-size: 12px; color: var(--ink-2); margin: 0 0 10px; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 760px) { .row2 { grid-template-columns: 1fr; } }
  svg { display: block; width: 100%; height: auto; }
  svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .tooltip {
    position: fixed; pointer-events: none; z-index: 10; display: none;
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px; font-size: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12); color: var(--ink-1);
    min-width: 130px;
  }
  .tooltip .tt-title { color: var(--ink-2); font-size: 11px; margin-bottom: 4px; }
  .tooltip .tt-row { display: flex; align-items: center; gap: 6px; margin-top: 2px; }
  .tooltip .tt-key { width: 12px; height: 0; border-top: 2px solid; }
  .tooltip .tt-val { font-weight: 600; }
  .tooltip .tt-name { color: var(--ink-2); }
  .legend { display: flex; gap: 16px; font-size: 12px; color: var(--ink-2);
            margin: 2px 0 8px; flex-wrap: wrap; }
  .legend .key { display: inline-flex; align-items: center; gap: 6px; }
  .legend .swatch-line { width: 14px; height: 0; border-top: 2px solid; }
  details { margin-top: 8px; }
  summary { cursor: pointer; font-size: 13px; color: var(--ink-2); }
  table { border-collapse: collapse; font-size: 12px; margin-top: 8px; width: 100%; }
  th, td { text-align: right; padding: 4px 10px; border-bottom: 1px solid var(--grid);
           font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--ink-2); font-weight: 600; }
  .empty { color: var(--ink-3); font-size: 13px; padding: 24px 0 16px; }
  .scrollbox { max-height: 260px; overflow-y: auto; }
</style>
</head>
<body class="viz-root">
<div class="wrap">
  <h1>KL-anchored self-play — live run report</h1>
  <p class="sub" id="subtitle"></p>
  <div class="tiles" id="tiles"></div>
  <div class="panel">
    <h2>Anchor drift KL(&pi;<sub>&theta;</sub> &Vert; &pi;<sub>ref</sub>) per update</h2>
    <p class="psub">Vertical markers are promotion gates; after a
      <span style="font-weight:600">&#10003; promoted</span> gate the reference
      becomes a copy of the live policy, so the KL re-anchors toward zero
      (the ratchet's sawtooth). Held gates keep the old reference.</p>
    <div id="klchart"></div>
  </div>
  <div class="row2">
    <div class="panel">
      <h2>Promotion gates</h2>
      <p class="psub" id="gatesub"></p>
      <div id="gatechart"></div>
    </div>
    <div class="panel">
      <h2>Anchor cost — steps/s per update</h2>
      <p class="psub" id="absub"></p>
      <div class="legend" id="ablegend"></div>
      <div id="abchart"></div>
    </div>
  </div>
  <div class="row2">
    <div class="panel">
      <h2>Policy entropy</h2>
      <p class="psub">Masked-action entropy per update (collapse toward 0 = mode
        collapse; the anchor should keep it near the prior's).</p>
      <div id="entchart"></div>
    </div>
    <div class="panel">
      <h2>Critic explained variance</h2>
      <p class="psub">Fresh value head warming up from the cold start
        (&minus;1 &asymp; uninformative, 1 = perfect).</p>
      <div id="evchart"></div>
    </div>
  </div>
  <div class="panel">
    <details><summary>Table view — per-gate decisions</summary><div id="gatetable"></div></details>
    <details><summary>Table view — per-update metrics</summary><div class="scrollbox" id="metrictable"></div></details>
  </div>
</div>
<div class="tooltip" id="tt"></div>
<script type="application/json" id="data">__PAYLOAD__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
const fmt = {
  int: (v) => v == null ? "–" : v.toLocaleString("en-US"),
  kl: (v) => v == null ? "–" : v.toFixed(4),
  sps: (v) => v == null ? "–" : v.toFixed(1),
  pct: (v) => v == null ? "–" : (100 * v).toFixed(0) + "%",
  f2: (v) => v == null ? "–" : v.toFixed(2),
};
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function niceTicks(lo, hi, n) {
  const span = hi - lo || 1;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  let step = step0;
  for (const m of [1, 2, 2.5, 5, 10]) { if (span / (step0 * m) <= n) { step = step0 * m; break; } }
  const t = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
  return t;
}
const tt = document.getElementById("tt");
function showTT(x, y, title, rows) {
  tt.textContent = "";
  const t = document.createElement("div"); t.className = "tt-title";
  t.textContent = title; tt.appendChild(t);
  for (const r of rows) {
    const div = document.createElement("div"); div.className = "tt-row";
    if (r.color) { const k = document.createElement("span"); k.className = "tt-key";
      k.style.borderTopColor = r.color; div.appendChild(k); }
    const v = document.createElement("span"); v.className = "tt-val";
    v.textContent = r.value; div.appendChild(v);
    const n = document.createElement("span"); n.className = "tt-name";
    n.textContent = r.name; div.appendChild(n);
    tt.appendChild(div);
  }
  tt.style.display = "block";
  const pad = 14, w = tt.offsetWidth, h = tt.offsetHeight;
  tt.style.left = Math.min(x + pad, window.innerWidth - w - 8) + "px";
  tt.style.top = Math.max(8, Math.min(y - h - pad, window.innerHeight - h - 8)) + "px";
}
function hideTT() { tt.style.display = "none"; }

// Generic single/multi line panel with crosshair tooltip.
// series: [{name, color, points: [{x, y, meta}]}]; xs must align by x value.
function linePanel(mount, series, opts) {
  const W = opts.w || 980, H = opts.h || 240,
        M = { t: 14, r: opts.mr || 76, b: 30, l: opts.w ? 44 : 52 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const allX = [...new Set(series.flatMap(s => s.points.map(p => p.x)))].sort((a, b) => a - b);
  const allY = series.flatMap(s => s.points.map(p => p.y)).filter(v => v != null);
  let lo = opts.zero ? 0 : Math.min(...allY), hi = Math.max(...allY);
  if (opts.includeZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (lo === hi) { hi = lo + 1; }
  const padY = (hi - lo) * 0.08; if (!opts.zero || lo !== 0) lo -= padY; hi += padY;
  const X = x => M.l + (allX.length < 2 ? iw / 2 : (x - allX[0]) / (allX[allX.length - 1] - allX[0]) * iw);
  const Y = y => M.t + ih - (y - lo) / (hi - lo) * ih;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
                          "aria-label": opts.aria || "" });
  mount.appendChild(svg);
  for (const t of niceTicks(lo, hi, 4)) {
    el("line", { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t),
                 stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: M.l - 8, y: Y(t) + 4, "text-anchor": "end", "font-size": 11,
                 fill: "var(--ink-3)" , style: "font-variant-numeric: tabular-nums"}, svg)
      .textContent = opts.yfmt(t);
  }
  el("line", { x1: M.l, x2: W - M.r, y1: M.t + ih, y2: M.t + ih,
               stroke: "var(--baseline)", "stroke-width": 1 }, svg);
  const xt = niceTicks(allX[0], allX[allX.length - 1], opts.w ? 5 : 8)
    .filter(v => Number.isInteger(v) && X(v) < W - M.r - 52);
  for (const t of xt) {
    el("text", { x: X(t), y: H - 8, "text-anchor": "middle", "font-size": 11,
                 fill: "var(--ink-3)", style: "font-variant-numeric: tabular-nums" }, svg)
      .textContent = fmt.int(t);
  }
  el("text", { x: W - M.r, y: H - 8, "text-anchor": "end", "font-size": 11,
               fill: "var(--ink-3)" }, svg).textContent = opts.xlabel || "update";
  // event markers (gates) under the data lines
  for (const ev of opts.events || []) {
    const x = X(ev.x);
    el("line", { x1: x, x2: x, y1: M.t, y2: M.t + ih, stroke: "var(--baseline)",
                 "stroke-width": 1 }, svg);
    const lab = el("text", { x: x, y: M.t + 2, "font-size": 10,
                             "text-anchor": "middle", fill: ev.strong ? "var(--ink-1)" : "var(--ink-3)",
                             "font-weight": ev.strong ? 600 : 400 }, svg);
    lab.textContent = ev.label;
    lab.setAttribute("transform", `translate(0,-4)`);
  }
  for (const s of series) {
    const pts = s.points.filter(p => p.y != null);
    const dAttr = pts.map((p, i) => (i ? "L" : "M") + X(p.x).toFixed(1) + " " + Y(p.y).toFixed(1)).join(" ");
    el("path", { d: dAttr, fill: "none", stroke: s.color, "stroke-width": 2,
                 "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    const last = pts[pts.length - 1];
    if (last) {
      el("circle", { cx: X(last.x), cy: Y(last.y), r: 4.5, fill: s.color,
                     stroke: "var(--surface-1)", "stroke-width": 2 }, svg);
      if (opts.endLabel) {
        el("text", { x: X(last.x) + 10, y: Y(last.y) + 4, "font-size": 11,
                     fill: "var(--ink-1)", "font-weight": 600 }, svg)
          .textContent = opts.endLabel(s, last);
      }
    }
  }
  // crosshair + tooltip
  const cross = el("line", { y1: M.t, y2: M.t + ih, stroke: "var(--baseline)",
                             "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = series.map(s => el("circle", { r: 4.5, fill: s.color,
    stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" }, svg));
  const hit = el("rect", { x: M.l, y: M.t, width: iw, height: ih,
                           fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) * (W / r.width);
    let best = allX[0], bd = Infinity;
    for (const x of allX) { const d = Math.abs(X(x) - px); if (d < bd) { bd = d; best = x; } }
    cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best));
    cross.setAttribute("visibility", "visible");
    const rows = [];
    series.forEach((s, i) => {
      const p = s.points.find(q => q.x === best);
      if (p && p.y != null) {
        dots[i].setAttribute("cx", X(best)); dots[i].setAttribute("cy", Y(p.y));
        dots[i].setAttribute("visibility", "visible");
        rows.push({ color: s.color, value: opts.yfmt(p.y), name: s.name });
      } else dots[i].setAttribute("visibility", "hidden");
    });
    const meta = (series[0].points.find(q => q.x === best) || {}).meta;
    showTT(e.clientX, e.clientY, (opts.xname || "update") + " " + fmt.int(best) +
           (meta ? " · " + meta : ""), rows);
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    dots.forEach(d => d.setAttribute("visibility", "hidden"));
    hideTT();
  });
}

// ---- KPI tiles ----------------------------------------------------------
const m = D.metrics, g = D.gates;
const last = m[m.length - 1];
const promoted = g.filter(x => x.promote);
const tiles = [
  { label: "Steps trained", value: fmt.int(last.global_step), note: fmt.int(m.length) + " updates" },
  { label: "Anchor β", value: D.kl_coef == null ? "0.05" : String(D.kl_coef), note: "KL(πθ ‖ πref), legal actions" },
  { label: "Final KL to ref", value: fmt.kl(last.kl_to_prior), note: "per-update mean" },
  { label: "Promotion gates", value: fmt.int(g.length), note: g.length ? ((100 * D.threshold).toFixed(0) + "% rule, " + g[0].decisive + "+ games") : "none fired yet" },
  { label: "Promotions", value: fmt.int(promoted.length), note: promoted.length ? ("last at update " + promoted[promoted.length - 1].epoch) : "reference held" },
];
if (D.ab.off && D.ab.on && D.ab.off.mean && D.ab.on.mean) {
  const drop = 100 * (1 - D.ab.on.mean / D.ab.off.mean);
  tiles.push({ label: "Anchor cost", value: drop.toFixed(1) + "%",
               note: fmt.sps(D.ab.off.mean) + " → " + fmt.sps(D.ab.on.mean) + " steps/s" });
}
if (D.fallback && D.fallback.decisions) {
  tiles.push({ label: "Model-driven decisions", value:
    fmt.int(D.fallback.decisions - D.fallback.fallbacks) + "/" + fmt.int(D.fallback.decisions),
    note: "fallback diagnostic, 0 = clean" });
}
const tilesEl = document.getElementById("tiles");
for (const t of tiles) {
  const d = document.createElement("div"); d.className = "tile";
  const l = document.createElement("div"); l.className = "label"; l.textContent = t.label;
  const v = document.createElement("div"); v.className = "value"; v.textContent = t.value;
  d.appendChild(l); d.appendChild(v);
  if (t.note) { const n = document.createElement("div"); n.className = "note";
    n.textContent = t.note; d.appendChild(n); }
  tilesEl.appendChild(d);
}
document.getElementById("subtitle").textContent =
  "run " + D.run_tag + " · " + D.run_dir + " · 8×8 self-play (mirror/league/pool), MPS learner" +
  " · reference starts at the IL-lineage prior and ratchets forward on gate wins";

// ---- KL drift panel -----------------------------------------------------
linePanel(document.getElementById("klchart"), [{
  name: "KL to reference", color: css("--series-1"),
  points: m.map(r => ({ x: r.epoch, y: r.kl_to_prior,
                        meta: "step " + fmt.int(r.global_step) })),
}], {
  h: 260, zero: true, yfmt: fmt.kl, aria: "KL to reference per update",
  events: g.map(x => ({ x: x.epoch, strong: !!x.promote,
                        label: x.promote ? "✓ promoted" : "held" })),
});

// ---- Gates panel --------------------------------------------------------
(function gates() {
  const mount = document.getElementById("gatechart");
  document.getElementById("gatesub").textContent =
    "Mirrored pairs, live vs frozen reference; win rate over decisive games. " +
    "Promote iff strictly above " + (100 * D.threshold).toFixed(0) + "% with ≥" +
    (g[0] ? g[0].min_decisive : 20) + " decisive.";
  if (!g.length) {
    const d = document.createElement("div"); d.className = "empty";
    d.textContent = "No gate has fired yet in this run.";
    mount.appendChild(d); return;
  }
  const W = 480, H = 240, M = { t: 18, r: 12, b: 34, l: 44 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
                          "aria-label": "promotion gate win rates" });
  mount.appendChild(svg);
  const Y = v => M.t + ih - v * ih; // 0..1
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), stroke: "var(--grid)",
                 "stroke-width": 1 }, svg);
    el("text", { x: M.l - 6, y: Y(t) + 4, "text-anchor": "end", "font-size": 11,
                 fill: "var(--ink-3)" }, svg).textContent = (100 * t) + "%";
  }
  el("line", { x1: M.l, x2: W - M.r, y1: Y(0), y2: Y(0),
               stroke: "var(--baseline)", "stroke-width": 1 }, svg);
  const band = iw / g.length, bw = Math.min(24, band * 0.5);
  g.forEach((x, i) => {
    const cx = M.l + band * (i + 0.5);
    const h = Math.max(1, x.win_rate * ih);
    const y = Y(x.win_rate);
    const bar = el("path", { d: roundedTopBar(cx - bw / 2, y, bw, h + (Y(0) - y - h), 4),
                             fill: "var(--series-1)" }, svg);
    const capY = y - 6;
    el("text", { x: cx, y: capY, "text-anchor": "middle", "font-size": 11,
                 "font-weight": 600, fill: "var(--ink-1)" }, svg)
      .textContent = fmt.pct(x.win_rate);
    el("text", { x: cx, y: H - 20, "text-anchor": "middle", "font-size": 11,
                 fill: "var(--ink-2)" }, svg).textContent = "u" + fmt.int(x.epoch);
    el("text", { x: cx, y: H - 7, "text-anchor": "middle", "font-size": 10,
                 "font-weight": x.promote ? 600 : 400,
                 fill: x.promote ? "var(--good-text)" : "var(--ink-3)" }, svg)
      .textContent = x.promote ? "✓ promoted" : "held";
    const hit = el("rect", { x: cx - band / 2, y: M.t, width: band, height: ih,
                             fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (e) => {
      bar.setAttribute("opacity", "0.85");
      showTT(e.clientX, e.clientY,
             "gate at update " + x.epoch + " (step " + fmt.int(x.global_step) + ")", [
        { color: css("--series-1"), value: fmt.pct(x.win_rate), name: "win rate (decisive)" },
        { value: x.wins + "W / " + x.losses + "L / " + x.draws + "D", name: "of " + (x.wins + x.losses + x.draws) + " games" },
        { value: String(x.decisive), name: "decisive (need ≥" + x.min_decisive + ")" },
        { value: x.promote ? "promoted" : "held", name: "decision" },
        { value: (x.seconds || 0).toFixed(0) + "s", name: "gate cost" },
      ]);
    });
    hit.addEventListener("pointerleave", () => { bar.removeAttribute("opacity"); hideTT(); });
  });
  // threshold rule on top of bars
  el("line", { x1: M.l, x2: W - M.r, y1: Y(D.threshold), y2: Y(D.threshold),
               stroke: "var(--ink-2)", "stroke-width": 1,
               "stroke-dasharray": "none" }, svg);
  el("text", { x: W - M.r, y: Y(D.threshold) - 5, "text-anchor": "end",
               "font-size": 10, fill: "var(--ink-2)" }, svg)
    .textContent = "promote above " + (100 * D.threshold).toFixed(0) + "%";
})();
function roundedTopBar(x, y, w, h, r) {
  r = Math.min(r, w / 2, h);
  return `M ${x} ${y + h} L ${x} ${y + r} Q ${x} ${y} ${x + r} ${y}` +
         ` L ${x + w - r} ${y} Q ${x + w} ${y} ${x + w} ${y + r} L ${x + w} ${y + h} Z`;
}

// ---- Throughput A/B panel ----------------------------------------------
(function ab() {
  const mount = document.getElementById("abchart");
  const sub = document.getElementById("absub");
  if (!(D.ab.off && D.ab.on)) {
    sub.textContent = "A/B runs not provided.";
    return;
  }
  sub.textContent = "Same config (8×8, " + fmt.int(D.ab.off.total_steps) +
    " steps each), chained runs; marginal steps/s per update, worker-startup update dropped.";
  const sOff = { name: "anchor off (β=0)", color: css("--series-1"),
                 points: D.ab.off.marginal.map(r => ({ x: r.epoch, y: r.sps })) };
  const sOn = { name: "anchor on (β=0.05)", color: css("--series-2"),
                points: D.ab.on.marginal.map(r => ({ x: r.epoch, y: r.sps })) };
  const leg = document.getElementById("ablegend");
  for (const s of [sOff, sOn]) {
    const k = document.createElement("span"); k.className = "key";
    const sw = document.createElement("span"); sw.className = "swatch-line";
    sw.style.borderTopColor = s.color;
    const t = document.createElement("span");
    t.textContent = s.name + " · mean " +
      fmt.sps(s === sOff ? D.ab.off.mean : D.ab.on.mean);
    k.appendChild(sw); k.appendChild(t); leg.appendChild(k);
  }
  linePanel(mount, [sOff, sOn], { h: 210, w: 480, zero: true, yfmt: fmt.sps,
    aria: "throughput anchor on vs off", mr: 16, xname: "update" });
})();

// ---- Health small multiples --------------------------------------------
linePanel(document.getElementById("entchart"), [{
  name: "entropy", color: css("--series-1"),
  points: m.map(r => ({ x: r.epoch, y: r.entropy })),
}], { h: 190, w: 480, zero: true, yfmt: fmt.f2, mr: 16,
      aria: "policy entropy per update" });
linePanel(document.getElementById("evchart"), [{
  name: "explained variance", color: css("--series-1"),
  points: m.map(r => ({ x: r.epoch, y: r.explained_variance })),
}], { h: 190, w: 480, includeZero: true, yfmt: fmt.f2, mr: 16,
      aria: "critic explained variance per update" });

// ---- Tables -------------------------------------------------------------
function table(mount, cols, rows) {
  const t = document.createElement("table");
  const tr = document.createElement("tr");
  for (const c of cols) { const th = document.createElement("th");
    th.textContent = c; tr.appendChild(th); }
  t.appendChild(tr);
  for (const r of rows) {
    const tr2 = document.createElement("tr");
    for (const v of r) { const td = document.createElement("td");
      td.textContent = v; tr2.appendChild(td); }
    t.appendChild(tr2);
  }
  mount.appendChild(t);
}
table(document.getElementById("gatetable"),
  ["gate (update)", "step", "W", "L", "D", "decisive", "win rate", "decision", "cost (s)"],
  g.map(x => [x.epoch, fmt.int(x.global_step), x.wins, x.losses, x.draws,
              x.decisive, fmt.pct(x.win_rate), x.promote ? "promoted" : "held",
              (x.seconds || 0).toFixed(0)]));
table(document.getElementById("metrictable"),
  ["update", "step", "KL to ref", "steps/s (cum)", "entropy", "explained var"],
  m.map(r => [r.epoch, fmt.int(r.global_step), fmt.kl(r.kl_to_prior),
              fmt.sps(r.sps), fmt.f2(r.entropy), fmt.f2(r.explained_variance)]));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
