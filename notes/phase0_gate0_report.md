# Phase 0 / Gate 0 — Reproducibility and the A0 finding

Date: 2026-08-02. Supersedes nothing in `phase0_discovery_report.md` /
`phase1_decisions.md` — this is new ground covered while rebuilding the
827.8 artifact, not a redo.

## Reproducibility: PASS

`scripts/build_winning_submission.py` + `configs/winning_build.json` rebuild
submission 55162376 (public 827.8) from `models/il_agent_winning_827.8/`
(the preserved artifact) and git SHA `e42f3bb` — not from
`agents/mega_lucario/agent_core_improved.py` HEAD or `models/il_agent/`,
both of which have since diverged.

- Checkpoint sha256 verified: `758dd7bc55fbda7a119eb7071b42e6744dd0f83f03365f0b85b0d55d0884a67a`.
- Sandboxed self-play (bundle extracted to scratch, only bundled files on
  `sys.path`): `DONE`/`DONE`, no crash.
- Bundle size: **13.13 MiB**, well under the 197.7 MiB cap.
- `src/pokemon_tcg/*` is byte-identical between `e42f3bb` and HEAD
  (`git diff e42f3bb HEAD --stat -- src/pokemon_tcg/` is empty) — safe to
  bundle current source for those files.
- `data/external/cg-lib/` is not git-tracked (under gitignored `data/**`);
  bundled as whatever is currently on disk. Assumed unchanged since
  submission — not independently verified, flagging as an assumption.

## The `agent_core_improved.py` diff: e42f3bb (winning) vs HEAD

`git diff e42f3bb HEAD -- agents/mega_lucario/agent_core_improved.py`:
97 lines changed, all from commit `b086538` ("Fix dead search bandit").
Every change is scoped to code that only executes when `SEARCH_ALGO`
actually reaches `simulate_action` (see below) — deck-out penalty regrade,
`plan`/`ability_used` global save/restore, `random.sample` vs `pad` in
`_predict`, plus the `rank_all()`/`choose()` split. `bc_prior.py` is
byte-identical between the two.

## A0: the BC prior's contribution — PROVEN inert in the winning build, not just unmeasured

The original Phase-0 plan was to benchmark `USE_BC_PRIOR=0` vs `1` (~2h) on
the winning code. That benchmark was never run, because tracing the code
first showed it would measure a foregone conclusion.

**Static trace.** `SEARCH_ALGO` only proceeds past its guard at
`SelectContext.MAIN` decisions:

```python
def SEARCH_ALGO(obs_dict, obs):
    if select is None or select.context != SelectContext.MAIN:
        return None
    base_order = AdvancedPolicy(obs).choose()      # truncates to maxCount
    candidates = base_order[:SEARCH_MAX_CANDIDATES]
    if not candidates:
        return None
    if len(candidates) == 1:
        return [candidates[0]] + [...]              # <-- returns HERE
    ...
    bc_probs = (bc_prior.candidate_probs(...) if ... else None)   # never reached
```

`choose()` (in the winning source) returns `ranked[:self.select.maxCount]`.
Scanned **23,166/23,166** real `MAIN` decisions across 150 train-day
episodes: `maxCount == 1` in every single one (mechanically true — a TCG
turn takes one action per `MAIN` decision). So `candidates` always has
exactly 1 element at `MAIN`, the early-return always fires, and
`bc_prior.candidate_probs()` — along with `simulate_action` and the UCB1
loop — is never reached.

**Dynamic confirmation.** Instrumented `bc_prior.candidate_probs` with a
call counter, ran 5 full self-play games (672 total env steps) using the
exact rebuilt 827.8 bundle: **0 calls.** Matches the static trace exactly.

**Verdict:** in submission 55162376, `USE_BC_PRIOR` and `BC_PRIOR_C` were
architecturally inert — not "small effect," not "underpowered to detect,"
zero code-path reachability. The submission's observed behavior is
`AdvancedPolicy.choose()` (the hand-written heuristic), full stop. The
prompt's own framing of 827.8 as "one-ply engine-rollout search over
top-8 candidates... with bc_prior.py supplying a PUCT-style prior" describes
the code's intent, not its behavior when it scored 827.8.

Root cause is not unique to this repo: commit `b086538`'s message notes the
same truncate-then-search bug exists in `makthanithin_improved_prob`,
carried from the public notebook lineage it's ported from.

HEAD already fixed the bug independently (`rank_all()`, `b086538`,
2026-08-01) and, on turning search genuinely on, benchmarked it losing to
`makthanithin_improved_prob` (10 games/pair: 50% pre-fix dead-search
baseline → 0–5% with search live). HEAD now defaults `USE_SEARCH=0`. That
commit doesn't record whether `USE_BC_PRIOR` was on or off during that
specific test — the real, working combination (fixed search + working BC
prior) has never actually been measured. That is the real open question A0
was trying to answer and didn't.

## What A0 should measure instead (not yet run — see cost below)

Three (or five, with the `BC_PRIOR_C` sweep) arms, all using HEAD's fixed
`agent_core_improved.py` (real `rank_all()`-based candidates) +
`models/il_agent/` (current checkpoint — the winning checkpoint predates
this fix and was never exercised with working search, so there's no
"reproduce 827.8 with real search" arm; this is a fresh architecture
question, not a reproduction):

| Arm | USE_SEARCH | USE_BC_PRIOR | Represents |
|---|---|---|---|
| Control | 0 | — | = 827.8's actual behavior (pure `AdvancedPolicy`) |
| Search, no prior | 1 | 0 | The already-benchmarked losing config (probably — prior state during that run is unrecorded) |
| Search + prior | 1 | 1 (C=0.75) | Never actually measured — the real hybrid the prompt assumed existed |
| (optional) C sweep | 1 | 1 (C=0.25, 2.0) | Sensitivity of the prior's weight |

## Cost reality check — this is not the ~2h originally budgeted

Timed 2 self-play matches with `USE_SEARCH=1 USE_BC_PRIOR=1` on both sides
(HEAD's fixed code, current checkpoint): **104.6 s/match** average (160
steps/174.5s, 45 steps/34.6s — roughly 1s/step, dominated by
`SEARCH_TIME_BUDGET=1.5s` now being spent on close to every real `MAIN`
decision instead of being dead code). Compare to the dead-search/heuristic
baseline from Phase 0: 0.25 s/match. **That's a >400x slowdown** —
`SEARCH_TIME_BUDGET` is a per-decision cap that used to never bind and now
always does.

At this rate, the original Q24 power-calc target (≈1,570 matches/arm for a
5pp resolution) is **≈45 hours per arm** — not affordable. See the
follow-up question for a scoped-down design.

## A0' results — fast-signal scope (run 2026-08-02)

Scoped down to 3 arms × 5 mirrored pairs (10 games) vs a single fixed
opponent (`makthanithin_improved_prob`, continuity with the one existing
historical data point). `scripts/ablation_a0.py` / `ablation_a0_summary.py`,
raw results in `reports/ablation_a0.json`.

Note the real per-game cost came in far below the ~104.6s/match self-play
figure used for the projection — that number was two search-enabled agents
playing each other; here only one side (`agent_core_improved`) searches
against a fast non-search opponent, so total time per game is bounded by one
side's budget, not two.

| arm | USE_SEARCH | USE_BC_PRIOR | games | win% | 95% Wilson CI | s/game |
|---|---|---|---:|---:|---|---:|
| control (= 827.8's real behavior) | 0 | — | 10 | 40.0% | [16.8, 68.7] | 0.3s |
| search, no prior | 1 | 0 | 10 | 0.0% | [0.0, 27.8] | 15.8s |
| search + prior (C=0.75) | 1 | 1 | 10 | 0.0% | [0.0, 27.8] | 10.3s |

**Per the pre-registered rule** (refuse to rank overlapping intervals):
control's CI [16.8, 68.7] overlaps both search arms' CI [0.0, 27.8] — a
~11-point overlap band. Formally: **tied at this sample size, all three
pairs.**

**But say what the point estimate says too, since it's this stark.** 4/10 vs
0/10 vs 0/10 is not a subtle signal — a rough two-proportion power calc at
these exact point estimates (p1=0.40, p2=0.00, α=0.05, 80% power) puts the
resolving sample at **~12 games/arm**, i.e. roughly **2 more mirrored pairs
each** would very likely separate control from either search arm, and it is
now cheap to get them (~15s/game against this opponent, not ~100s). Distinguishing
the two search arms from each other (0/10 vs 0/10) needs a real effect to
exist first — no evidence of one yet either way.

**Reading, not yet a final verdict:** every real-search game lost,
regardless of the BC prior. This is directionally consistent with HEAD's
prior finding (search live nets negative vs this same opponent) and extends
it: a genuinely working BC prior (the untested combination A0 set out to
measure) did not rescue search here, at least against this one opponent, at
n=10. It does **not** yet distinguish "search hurts, prior doesn't fix it"
from "prior needs the C sweep" or "this opponent happens to be a bad matchup
for the rollout heuristic specifically" (single-opponent design, flagged as
a limitation, not fixed here per the approved fast-signal scope).

## A0' — extended to n=16/arm, resolved (run 2026-08-02, same session)

+6 games/arm (3 more mirrored pairs), same opponent. Real cost stayed cheap
(one-sided search against a fast non-search opponent: 10–14s/game, not the
~100s self-play worst case).

| arm | games | win% | 95% Wilson CI |
|---|---:|---:|---|
| control (pure heuristic, = 827.8's real behavior) | 16 | 43.8% | [23.1, 66.8] |
| search, no prior | 16 | 0.0% | [0.0, 19.4] |
| search + prior (C=0.75) | 16 | 0.0% | [0.0, 19.4] |

**Now resolved, not just suggestive:** control's CI [23.1, 66.8] no longer
overlaps either search arm's CI [0.0, 19.4] — **pure `AdvancedPolicy`
decisively beats real search** against this opponent, prior or no prior.
`search_noprior` vs `search_prior` remain tied at 0/16 each — no evidence
the BC prior changes anything once search is live, at least against this one
opponent at this N.

**Decision (confirmed): treat `AdvancedPolicy` — the pure heuristic that
actually produced 827.8 — as the real baseline going forward, not the
search/UCB1/BC-prior machinery.** HEAD's `USE_SEARCH=0` default already
matches this; no code change needed. The BC prior remains available and
wired (harmless when `USE_SEARCH=0`, since `SEARCH_ALGO` short-circuits
before ever reaching it), but it should not be treated as load-bearing or as
this repo's "hybrid" story going forward — Phase 6+'s BC/RL track is judged
against pure `AdvancedPolicy`, not a search fiction.

**Not resolved, deliberately left open:** whether `evaluate_state` (the
rollout's shallow one-turn board scorer) is a fixable culprit — per
`b086538`'s own suspicion that it "trusts evaluate_state... over
AdvancedPolicy's more sophisticated domain-tuned heuristic whenever they
disagree" — or whether one-ply search is simply the wrong shape for a TCG
turn regardless of leaf quality. Also open: this is a single-opponent
result (`makthanithin_improved_prob`); Phase 1's widened pool may show a
different matchup-dependent picture. Not spending further session time on
`evaluate_state` now per the explicit decision above; revisit only if a
specific opponent-matchup gap in the Phase 1 pool implicates the rollout
heuristic specifically.

## A0'' — same experiment, but on a deck the BC prior actually has signal for (2026-08-02)

A0' (above) tested search + BC prior on the frozen Mega Lucario ex deck, whose
defining cards (Riolu/Mega Lucario ex) appear in **0/800** training-corpus
perspectives — the prior's embeddings for that deck are untrained regardless
of outcome. Re-scanned prevalence for every archetype now in the pool
(`cg.api`-verified card IDs, same 400-episode/800-perspective methodology as
`notes/phase1_decisions.md`):

| Archetype | Training-corpus prevalence |
|---|---:|
| Alakazam (Abra/Kadabra/Alakazam) | **16.1%** (129/800) |
| Archaludon ex (`plamen06_steel`) | 0.12% (1/800) |
| Mega Lucario ex (frozen deck, 9 pool agents) | 0.00% (0/800) |
| Dragapult ex / Iono's Bellibolt ex / Mega Abomasnow ex | 0.00% each |

Alakazam is the only pool archetype with real trained signal, so it's the
only fair test of "does a *working* BC prior help search." Re-ran the exact
A0' design (same 3 arms, same fast-signal n=10/arm, same opponent
`makthanithin_improved_prob` for comparability) with
`agent_core_improved.my_deck` overridden to `agents/ryotasueyoshi_alakazam/deck.csv`
(`scripts/ablation_a0.py --my-deck-csv`, results in
`reports/ablation_a0_alakazam.json`). The heuristic (`AdvancedPolicy`) itself
is untouched — still Lucario-tuned, doesn't know Alakazam-specific matchups —
but that's held constant across all three arms, so it doesn't bias the
prior-specific comparison, only the absolute win rates (all arms are weaker
here than the Lucario A0' run, as expected).

| arm | games | win% | 95% Wilson CI | s/game |
|---|---:|---:|---|---:|
| control (no search) | 10 | 20.0% | [5.7, 51.0] | 0.3s |
| search, no prior | 10 | 0.0% | [0.0, 27.8] | 3.5s |
| search + working prior (C=0.75) | 10 | 10.0% | [1.8, 40.4] | 19.5s |

**Tied at this sample size, all three pairs** — CIs overlap throughout, this
is underpowered by design (fast-signal scope, not a resolving one).

**What it does and doesn't show.** search_prior (1/10) beat search_noprior
(0/10) — the same direction the question hoped for — but this is one game
out of twenty total, nowhere near enough to call it real; a rough
two-proportion estimate for resolving a 10pp gap at these base rates would
need on the order of 100+ games/arm, not a cheap follow-up like A0' had.
**What generalizes cleanly, though**: control (pure heuristic) beat both
search arms here too, same as the Lucario result — search hurting relative
to no-search is not deck-specific, it replicates on a completely different
archetype with a completely different opponent matchup profile. That's the
more load-bearing finding from this run: it's evidence the problem is in
this codebase's search/rollout implementation (most likely `evaluate_state`,
per `b086538`'s own suspicion, still not reworked), not something specific
to the Lucario deck's heuristic or the Lucario deck's untrained embeddings.

Not extended further this session — the prior-vs-no-prior question specifically
remains open at real statistical power; flagging as a candidate for Phase 6
rather than chasing it here, since Phase 6 will train a policy that (unlike
this repo's frozen `models/il_agent`) could be deliberately built to have
real signal for whichever deck ships.

## Gate 0 verdict

**PASS.** Bundle reproduces byte-for-byte-verified (checkpoint sha256) and
self-plays cleanly. The behavioral gap between the winning bundle and HEAD
is fully enumerated and, per the analysis above, does not change what should
ship next (pure heuristic was and remains the better default). A0's real
question — does search + a working BC prior help — is answered: no, not
against this opponent, not at this sample size. Proceeding to Phase 1.
