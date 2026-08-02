# Phase 1 — Benchmark Honesty Pass + Archetype Coverage (in progress)

Date: 2026-08-02. Session also fixed a worktree gap: `submissions/` wasn't
symlinked from the main checkout (only `data/`/`models/` were, from the
Phase 0 session), which made `rule_baseline` crash with `NameError: name
'my_deck' is not defined` — it fell back to a bare-module load with no
sibling `deck.csv`. Fixed by symlinking `submissions/{il_agent,
improved_probabilistic,mega_lucario,mega_lucario_improved}` the same way.
Not a code bug; flagging in case another fresh worktree hits the same thing.

## 1. Benchmark harness honesty pass — DONE

`scripts/benchmark_agents.py`:
- Wilson 95% CI (via new `scripts/glicko1.py::wilson_ci`) on every overall
  win rate and every per-pairing win rate, printed and written to
  `reports/agent_benchmark.json` (`overall_win_pct_wilson_ci`).
- Wall clock per pairing and total run wall clock, written to
  `wall_clock_sec` / `total_wall_clock_sec` in the same JSON — previously
  printed to stdout per-pairing and then discarded.
- `scripts/ablation_a0_summary.py` refactored to import the same
  `wilson_ci` instead of a duplicate copy.

**Common random seeds: confirmed NOT POSSIBLE, not just unimplemented.**
Checked the actual engine, not assumed: `cabt.json`'s configuration schema
has no seed field, `kaggle_environments.make()`'s generic `configuration`
dict has no seed convention this env reads, and `cg.game.battle_start(deck0,
deck1)` — the actual call that starts a match — takes no seed parameter.
Deck shuffling and opening hands are decided inside the compiled native
engine (`libcg-arm64.so`/`cg.dll`/`libcg.dylib`), which exposes no RNG
control from Python at all. This is now documented in
`benchmark_agents.py`'s module docstring and recorded as `"seeded": false`
with a reason string in every `agent_benchmark.json` run, so it can't be
silently assumed later.

**Validated on a real run**: full current 14-agent pool, `--games 8`
(105 pairings, 1,568 games), 454.7s wall clock. All CIs, wall-clock fields,
and the seed-note populated correctly. Also surfaced and fixed a pre-existing
harness crash (`TypeError: '>' not supported between NoneType and int` when
a game ends `ERROR`/reward `None`) — root cause was the missing
`submissions/` symlink above, not a play_match logic bug; not touched
further since it stopped reproducing once the symlink was in place, but
flagging that `play_match` still has no explicit handling for a genuine
`ERROR`-status game if one ever occurs for a different reason — it would
crash the whole run and lose every result gathered so far, since
`agent_benchmark.json` is only written at the very end. Worth a follow-up
if this turns out not to be purely the symlink issue.

## 2. Archetype coverage — measured, one new archetype recruited

**Before** (14-agent pool, per `deck.csv`/hardcoded-`DECK` audit against
`cg.api.all_card_data()`):

| Archetype | Pilots | Count |
|---|---|---:|
| Mega Lucario ex (the frozen submitted deck) | rule_baseline, improved_prob_main, agent_core_improved, proto, il_agent, grunt, random_legal, dedquoc_rule_engine, makthanithin_improved_prob | 9 |
| Alakazam (Psychic) | ryotasueyoshi_alakazam, mechi22_alakazam | 2 |
| Dragapult ex (Dragon) | kiyotah_dragapult | 1 |
| Iono's Bellibolt ex (Electric) | kiyotah_iono | 1 |
| Mega Abomasnow ex (Grass/Ice) | kiyotah_abomasnow | 1 |

**9/14 = 64.3% of the pool pilots the exact same deck** — worse than Phase
0's "8/13" estimate, because the agents added since then (`grunt`,
`random_legal`) correctly ride the frozen deck (they're testing algorithm
quality on the actual submitted deck, which is right for THEM) without
adding pool diversity. 5 distinct archetypes total.

**Recruited**: `plamen06/pokemon-steel` → `agents/plamen06_steel/` —
Archaludon ex / Duraludon / Cinderace, Metal-type "metal-tempo" deck. First
non-Fighting/Psychic/Dragon/Electric/Grass-Ice archetype in the pool.
Sourced from `makimakiai/ptcg-public-23-plus-sample-4-roster-update`'s
referenced-notebooks list (pulled via `kaggle kernels pull`, not the
browser — Kaggle notebook pages are client-rendered JS and the browser tool
couldn't reliably extract cell/image content; the CLI pull approach that
built the rest of `notebooks/reference/` was far more reliable). Adapted
(not copied verbatim), safety-scanned, and card-ID-verified before wiring
in — see `notebooks/reference/INDEX.md`'s new section for the full
provenance and the one real bug fixed in porting (a relative `deck.csv`
read that would have silently loaded the WRONG deck under this repo's
harness, since the repo root has its own `deck.csv` for a different
agent's frozen deck).

**Investigated and rejected**:
- `aman5153684/a-crustle-aware-fighting-agent` — same exact Mega Lucario ex
  deck IDs (673-678) as everything else, just Crustle-matchup-tuned. Adds
  no archetype diversity.
- `seokjeongeum/strong-start-psychic-anti-meta-v8-lb-1100` — 403 Forbidden
  on `kaggle kernels pull` (private/deleted), same pattern as the
  already-noted `seokjeongeum` 1208 notebook from the Phase 0 crawl.
- `ahmedabdelhmed-bayesian-heuristic` (the original crawl's seed notebook) —
  re-examined while auditing `notebooks/reference/` for un-wired
  candidates. References observation fields that don't exist in the real
  API (`obs.active`, `obs.initial_deck_composition`,
  `obs.estimated_opp_damage`, etc. — the real schema is
  `obs_dict['select']`/`obs_dict['current']`) and never imports `cg.api`.
  Its "78% win rate" claim is very likely fabricated/decorative, not
  measured against the real engine. Not wired in.
- `avikdas567-heuristic-agent`, `aristophanivan`, `romanrozen-strong-start-agent`,
  `makimakiai-tiny-rl-baseline` (all already in `notebooks/reference/` from
  the Phase 0 crawl, not yet wired into `benchmark_agents.py`) — checked
  their deck sources: all either read the official sample-submission deck
  directly or hardcode the same card IDs (673-678). All Mega Lucario ex.
  Not recruited (would just be more mirror-match volume).

**After recruiting `plamen06_steel`**: 15-agent pool, 9/15 = 60.0% mirror
share. A small improvement (64.3% → 60.0%), not the halving that would
actually fix Gate 1's "archetype coverage is the problem" diagnosis —
finding genuinely new, functional, safety-clean archetypes among public
notebooks is harder than the prompt's source list suggested (most
candidates traced back to the same deck, one likely-fabricated notebook
found along the way, one source author has gone largely private).

## Gate 1 status: NOT YET MET, in progress

Per the prompt's own gate condition ("if >=50% of games are still one
archetype, the pool still is not a ladder proxy and you say so rather than
proceeding as if it were") — **60.0% is still well above 50%.** This gate
has not passed. Options, not yet decided:
1. Continue recruiting from the rest of `makimakiai`'s referenced list not
   yet checked (`kokinnwakashuu`, `penguin069`, `alycemiki`,
   `yaroslavkholmirzayev`, `kacchanwriting`, `biohack44`,
   `makthanithin_1084_5_baseline`/`ptcg-mega-lucario-ex-v62` (likely more
   Lucario), `pixiux`, `pilkwang`, `rauffauzanrambe`, `rv1922`,
   `seokjeongeum/pure-dragapult-ex-deck` (another Dragapult, not new),
   `seokjeongeum/mega-pokemon-reinforcement-ai-battle`, `yaminh`,
   `biohack44/meta-snapshot-07-july`) — several unlabeled/generic names,
   real chance most are also Lucario given the pattern so far.
2. Accept 60% and re-run Phase 2/3 analysis with archetype as an explicit
   confound rather than chasing a 50% threshold that may not be reachable
   from public notebooks alone (the frozen deck IS the dominant meta pick
   among public submitters, which is itself informative).
3. Down-weight or cap per-archetype contribution to Glicko/benchmark
   aggregates instead of trying to physically balance the roster.

Re-running the full-pool benchmark now (15 agents, `--games 6`) as the
"after" data point; standings + updated mirror-match fraction to follow.
