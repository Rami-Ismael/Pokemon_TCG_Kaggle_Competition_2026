---
name: expert-play-analysis
description: Measure which slice of the legal action space strong players actually use, by filtering episodes with the manifest's min_score/avg_score rating fields. Use when asked what strong/expert/top players do, to find macro-patterns (attach timing, retreat thresholds, never-pass-when-attack), to shrink or mask the effective decision space, to mine feature candidates from the episode corpus, or to filter training data by player skill.
---

The transferable lesson this skill encodes: the episode corpus and its
manifest ratings have value **independent of whether IL survives into the
final agent**. Filtering to high-rated players and asking "of the legal
options the cabt engine offers, which slice do strong players actually
use?" yields action-space masks and feature candidates that survive any
pivot to self-play RL. The harness is
[scripts/expert_play_analysis.py](../../../scripts/expert_play_analysis.py);
this file is its method card. All paths relative to repo root.

## Run

```bash
uv run python scripts/expert_play_analysis.py                      # eval/2026-07-27, 150+150 eps
```

```bash
uv run python scripts/expert_play_analysis.py --split train --day 2026-07-01 --top 40 --bottom 40
```

Reads ADR-001 Hub shards (`Rami/ptcg-episodes`) via the huggingface_hub
cache: warm-cache runs measured 300 eps / 57,745 decisions in ~21s and
80 eps in ~8s (training sessions keep the cache warm; a cold first pass
pays one shard download per day). Output: printed tables + a ~10KB JSON
under `reports/expert_play_analysis-{split}-{day}.json`.

## Method (why it's shaped this way)

- **Stratify episodes, not seats.** Neither the manifest nor
  `iter_episode_decisions` attributes decisions to a rated seat, so:
  strong = top-N by `min_score` (BOTH players above the floor → every
  decision is a strong player's); weak = bottom-N by
  `max_score = 2*avg_score − min_score` (both below the ceiling).
  Filtering by `avg_score` alone admits lopsided matches whose weak seat
  pollutes the stratum.
- **Within one day only.** Ladder scores drift across days (07-27 tops
  out ~1216, 07-01 ~1344) — cross-day strata would confound skill with
  calendar. The script refuses days with no manifest scores
  (train/2026-07-26 has none; 2026-08-03 is still ingesting).
- **Forced decisions excluded.** ~5.5% of decisions offer exactly one
  legal option; including them inflates every rate (phase0 §"exactly one
  legal option").
- **Every rate gets a Wilson 95% CI**; strong-vs-weak differences are
  flagged `≠` only when intervals are disjoint (repo standing rule: no
  uncertainty, no claim).

## How to read the output

Findings describe **offline ladder behavior, not our agent's strength** —
never promote one to a "better agent" claim without the
`leaderboard-check` skill. The three sections map to three uses:

1. **Pick share / pick-given-legal by OptionType** → is the effective
   decision space smaller than the legal one? (mask candidates)
2. **Macro-pattern probes** → conditional rules worth encoding as
   features (attach timing × `turnActionCount`, retreat × active-HP
   fraction, END × ATTACK-legal).
3. **Per-context volume** → where the decisions actually are (MAIN is
   only ~40% of them) and where declines are ever used.

## Verified findings — eval/2026-07-27, 150+150 eps (2026-08-03 run)

One day, one meta snapshot; re-run before relying on these.

- **Nobody passes when they can attack**: END picked with ATTACK legal =
  0.0% [0.0,0.1] (strong, n=7303) and 0.1% (weak). The strongest mask
  candidate found — pruning END-when-ATTACK-legal costs ~0 expert
  agreement.
- **5 of ~11 MAIN option types cover ≥95% of picks** (PLAY, ABILITY,
  ATTACH, ATTACK, EVOLVE); RETREAT+END ≈ 5% combined. MAIN offers 7.9
  options on average — the used space is much smaller.
- Strong players use **ABILITY more** (23.9% vs 21.1% share, ≠) and
  **EVOLVE less/later** (44.5% vs 49.7% when legal, ≠) — eager evolution
  is a weak-player tell, evolution *timing* is signal.
- Strong players **retreat more** (4.2% vs 3.2% when legal, ≠) and
  HP-conditionally: 6.1% at hp≤⅓ vs 3.2% at hp>⅔ (within-stratum CIs
  disjoint) — "retreat-only-under-X" exists as a gradient.
- **"Always attach energy first" is false as stated** — attach rate is
  *higher* later in the turn (act2+ 29.6% vs act0-1 20.3%) — but strong
  players attach earlier than weak (20.3% vs 15.5% at act0-1, ≠).
- Declines are near-dead except `ATTACH_TO` (17.7%) and `TO_BENCH`
  (13.6%) — decline modeling matters only in those contexts.

## Gotchas

- **Score coverage is per-day and partial**: 07-27 (4430 eps) and 07-01
  (5266) are scored; 07-26 has NO manifest rows (phase0-era gap);
  08-03 grows while the ingest loop runs — don't analyze it mid-ingest.
- **`ATTACH` conflates energy and tool attachment** (OptionType has no
  energy-only member); the energy-first probe uses `energyAttached=False`
  as the guard, which is the once-per-turn *manual energy* flag — close
  but not exact. Say so when quoting.
- **Multi-select decisions are counted per unrolled pick** (same
  convention as training rows, `iter_decisions`) — decision counts are
  "training-row" counts, not raw engine selects.
- **Needs `data/episodes/manifest.csv` locally** (worktrees: the
  episodes symlink — `run-ptcg-battle` doctor prints the fix) and Hub
  access for shards (`--local-root data/episodes_packed` works where a
  local pack exists).
- The raw split dirs now hold only ~a dozen sample JSONs — the corpus
  lives in the shards; don't "fix" the script to read raw folders.

## Extending

Per-card analysis (which card to PLAY, retreat targets) needs a card-id →
name/type DB join — not built. Seat attribution is impossible with
today's manifest. New scored days appear automatically as
`scripts/ingest_episodes.py` runs; re-running on a fresh day is the
cheapest replication check.
