# mcts_il_agent v2 — improve the IL-prior search agent

Paste the block below into Claude Code as the opening message. Everything
above the `---BEGIN PROMPT---` marker is notes for the human.

Provenance of the facts baked into the prompt:
`reports/search_prior_phase3.md`, `notes/experiments/2026-08-05-critic-calibration-sparse-reward.md`,
`reports/submission_ledger.jsonl` (submission 55248781), and the session
memory files `il-prior-mcts-beats-il-agent-locally`, `critic-calibration-measured`,
`cabt-time-budget-600s-overage-bank`, `local-glicko-vs-real-leaderboard-diverge`.

Skills the prompt asks Claude Code to load, and why:

| skill | why |
|---|---|
| `ptcg-repo-context` | hard constraints, data layout, agent/deck/episode interfaces — load before writing code, not after |
| `idea-to-experiment` | Rami's standing instruction: design the experiment first, implement second |
| `leaderboard-check` | v1's local 67% did NOT transfer (ladder 294.7); required before any "better" claim and after any submit |
| `run-fallback-diagnostic` | the ≥48-option OOB bug in v1 was caught ONLY by fallback instrumentation |
| `run-ptcg-battle` | live battles + worktree setup (symlinks for `models/`, `data/episodes`, cg-lib) |
| `deck-selection` | only if the deck axis is touched — the v1 comparison deliberately held deck fixed |
| `expert-play-analysis` | optional, for the "shrink the search's action space to what strong players use" arm |

---BEGIN PROMPT---

Improve `mcts_il_agent` past v1. Treat this as a research task, not a coding
task: I want experiment designs with pre-registered gates first, implementation
second, and honest reporting of negative results.

Load these skills before you plan: `ptcg-repo-context`, `idea-to-experiment`,
`leaderboard-check`, `run-fallback-diagnostic`, `run-ptcg-battle`.

## What v1 is

- Agent: `agents/mcts_il_agent/agent_core.py`; search: `src/pokemon_tcg/search_prior_mcts.py`.
- piKL-shaped (Jacob et al., arXiv:2112.07544): the engine's **official Search
  API** with `PTCGImitationPolicy` (`models/il_agent`, 3.32M params) supplying
  child priors; `SEARCH_COUNT=30`; leaf value 0 in the shipped prior-only mode.
- Forced CPU + `torch.set_num_threads(1)` at import; never-raise contract with
  `_safe_choice` fallback and `diag_snapshot()` counters.
- Deck held fixed to `models/il_agent`'s bundled list — **verify which deck
  that actually is** before writing about it; `reports/search_prior_phase3.md`
  calls it Grimmsnarl, my memory says the bundled list is Lucario. Whichever it
  is, it stays fixed unless an arm's whole point is the deck axis.

## What is already measured — do not re-derive these

1. **Prior-only search beats the policy it wraps, locally**: 67.2%
   [58.4, 75.0] pooled over two independent runs, n=119 decided, 0 fallbacks
   (`reports/search_prior_phase3.md`).
2. **The local gain did not transfer.** Submitted 2026-08-04 (ref 55248781):
   first reading 600.0, converged to **294.7** within ~30 min — below the
   il-family band (397–400). Third local-vs-ladder inversion. Any v2 claim of
   "better" is unproven until the ladder says so.
3. **The v1 critic arm's collapse to 52.5% is explained**: that critic FAILED
   calibration (eval MSE 0.2660 vs 0.2500 constant-0.5). Do not cite it as
   "critics don't help in search".
4. **A newer critic passes.** 2026-08-05 audit on 128k held-out rows: AUC
   0.759, MSE +17.6% over base rate, phase AUC 0.61→0.84, shuffled-outcome
   control chance-flat. ECE 0.062 vs a 0.05 bar → **fit temperature/isotonic
   before using V(s) as a probability**. Checkpoints `models/critic_trainday*`,
   rerun with `scripts/eval_critic_calibration.py`. Single seed — provisional.
5. **Timing at N=30**: 209 ms/decision mean, 358 ms p95, 441 ms max, forced-CPU.
   Budget is a 600 s/match overage bank with no per-step timeout; size any
   search increase against that bank with a 2–3× CPU safety factor.
6. **Bug class to keep instrumented**: decisions with ≥48 options have no
   DECLINE slot; v1's prior mapping indexed out of bounds and silently ate
   12.76% of decisions before the fix.

## What I want from you

**Step 1 — diagnose before you improve.** Read the v1 search code and produce
a short written account of *where the 67% local gain comes from and why it
plausibly failed to transfer*. Candidate explanations to discriminate between,
not just list: (a) the local benchmark's opponent pool is the policy the search
is anchored to, so search is exploiting a single known opponent rather than
playing well; (b) think-time distortion — search agents get an advantage in
local pairings that the ladder does not reproduce; (c) the ladder pool contains
strong non-ML heuristics the prior has never modeled; (d) a submission-bundle
difference (model/critic dir resolution, missing checkpoint → silent degraded
mode). Where a hypothesis is checkable with existing artifacts (ledger, replay
review, fallback counters, the anchored pool), check it — don't speculate.

**Step 2 — propose 3–5 v2 arms as experiments**, ranked by expected value per
GPU/CPU-hour, each with: hypothesis, the single variable that changes, control
arm, sample size for the effect size you expect to detect, and a
**pre-registered pass/fail gate written before any run**. Arms I consider live
(you may replace any of these with something better-argued):

- **Calibrated critic as leaf evaluator** — the v1 critic arm rerun with the
  passing `models/critic_trainday` checkpoint *after* a temperature fit. This
  is the arm the 08-05 audit green-lit; it directly tests whether leaf value
  adds to terminal lookahead or fights it.
- **Search budget scaling** — N ∈ {30, 60, 120} against the 600 s bank, with a
  measured timing rehearsal per point. Is the local gain monotone in N? Does it
  flatten (search saturating) or keep climbing (v1 was budget-starved)?
- **Prior temperature / piKL λ** — how hard search is allowed to deviate from
  the human-cloned prior. v1 used a plain softmax; sweep the KL anchor strength.
- **Opponent-diverse evaluation** — re-benchmark v1-as-is against the
  ladder-anchored pool rather than only vs `il_agent`, to test hypothesis (a)
  above. If v1's gain shrinks toward parity here, that is the whole finding and
  the other arms wait.
- **Action-space pruning** — restrict expansion to the option classes strong
  players actually use (`expert-play-analysis`), so a fixed sim budget goes
  further.

**Step 3 — implement and run only what survives Step 2**, one variable at a
time, deck fixed, mirrored pairs, Wilson 95% CIs, `--no-glicko-persist`.

## Non-negotiables

- **Never claim "better", "beats", or "improves" without `leaderboard-check`.**
  Local Glicko has inverted against the real ladder three times, this agent
  included. Evaluate against the anchored pool, not only against `il_agent`.
- **Fallback rate is a gate, not a footnote.** Every benchmark run reports
  `diag_snapshot()`; a nonzero unexplained fallback rate voids the run.
- **Timing rehearsal before any submission** at the final N, forced-CPU, quiet
  machine, p95 and max reported against the 600 s bank.
- **Never raise.** The `_safe_choice` never-crash contract holds in every new
  code path; INVALID is a loss.
- **Preflight before training runs** (disk, RAM, competing MPS jobs) and don't
  overlap MPS jobs. Training runs are pre-authorized; report projections rather
  than stopping for approval, keep the laptop responsive.
- **Write the negative results down.** An arm that fails its pre-registered
  gate goes in `notes/experiments/` with the same care as a win. Do not quietly
  drop it or re-cut the analysis until it passes.
- Before implementing, `git fetch` and `gh pr list` — concurrent sessions merge
  work into main mid-session, and refresh `reports/submission_ledger.jsonl`
  before any submit.

Start with Step 1. Show me the diagnosis and the ranked experiment table before
you write any agent code.

---END PROMPT---
