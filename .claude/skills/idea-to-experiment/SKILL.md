---
name: idea-to-experiment
description: Design a structured research experiment — ablation, scaling study, controlled comparison, generalization test — for an idea Rami wants tested. SUSPENDED as a default (2026-08-16) — use ONLY when Rami explicitly invokes it (/idea-to-experiment) or explicitly asks for an experiment design ("design an experiment for this", "how would we test this"). Do NOT auto-apply when he merely proposes an idea, improvement, or hunch — implement what he asks. When it does run, it is pairing on his idea, not persuading him: offer the design once, and if he says the design or analysis is wrong, that is the finding — drop it.
---

# Idea → research experiment

**Status: suspended as a default (2026-08-16).** Rami revoked the 2026-08-03
standing request that ideas be persuaded into experiments. A session used the
scientist framing as authority to argue with him when he said its
experimentation made no sense. This skill now runs only when he explicitly
invokes it or explicitly asks for an experiment design. When he proposes an
idea without asking for a design, implement the idea.

## Scope

Runs only on explicit request. When it runs, it applies to ideas carrying a
performance claim — "this should make the agent stronger / training faster /
data better" — not to bug fixes, refactors, tooling, or chores.

This is pairing, not persuading. Offer the design once. If Rami declines it,
or says the design or analysis is wrong, his call is the finding: drop it,
re-examine, and never cite this skill or a "scientist" role against him.

## The conversation move

When Rami asks for an experiment design, respond in this order:

1. **Name the hypothesis hiding in the idea.** Restate the idea as a testable
   claim: "Adding X will raise win rate against Y by Z because <mechanism>."
   If no mechanism can be stated, say so — that's the first finding.
2. **Ask what it would take to be wrong.** A great researcher pre-registers the
   result that would kill the idea. Get Rami to commit to a decision criterion
   *before* any code: "if the metric doesn't move by ≥ N, we drop it."
3. **Find the cheapest version.** What is the minimal change and smallest run
   that could falsify the hypothesis? (One config flag, one short training run,
   one benchmark sweep — not a rewrite.)
4. **Check prior work first.** Search `notes/` (ADRs, phase reports,
   `study-kiyotah-rl-mcts.md`, `scores.md`) and the Metamon paper before
   designing — the experiment may already have been run, or the grid already
   argued against it (see the rescaled-grid memory).
5. **Then implement and run it** — as the experiment, with the controls below.

If Rami says "just do it, skip the experiment," comply — but say in one sentence
what question will remain unanswered afterward.

## Experiment card (fill before running)

Write this into `notes/experiments/<date>-<slug>.md` before starting the run:

```markdown
# <short title>
- **Hypothesis:** <one falsifiable sentence with a mechanism>
- **Independent variable:** <the ONE thing that changes>
- **Baseline:** <exact agent/checkpoint/config being compared against>
- **Metric & protocol:** <what is measured, how many games/steps, which opponents>
- **Primary metric:** <strength — Glicko or win rate against the stated pool>
- **Guardrail metric:** <fidelity to teacher — e.g. KL to the never-retargeted
  IL prior, or top-1 agreement; a strength gain that torches the teacher is a
  different result than a clean gain>
- **Pre-registered decision:** adopt if <threshold>; drop if <threshold>
- **Cost estimate:** <wall-clock, disk, MPS contention>
- **Prior work checked:** <notes/ files or papers consulted>

## Result (fill after)
- **Observed:** <numbers, with the command that produced them>
- **Decision:** adopted / dropped / inconclusive — <why>
- **What we learned:** <one sentence, even (especially) for negative results>
- **Belief update:** <one line — ask Rami what the result changed about his
  beliefs, and log his answer, not a paraphrase of the numbers>
```

Negative results get written up too — an empty `## Result` section or a deleted
card means the experiment didn't happen.

## Experiment design menu — match the design to the question

Ablations are one design, not the only one. Part of becoming an ML researcher
is picking the design that actually answers the question. Name the design on
the experiment card; if the idea fits none of these, say so and design from
first principles.

| The question sounds like… | Design |
|---|---|
| "Is X better than what we have?" | **Controlled comparison** — X vs baseline, everything else identical. The atom of all designs below. |
| "Which of our n components matter?" | **Ablation** (leave-one-out / add-one-in) — detailed below |
| "Does more X (params, data, steps) help?" | **Scaling study** — sweep X, read the curve — detailed below |
| "How sensitive is this to its knobs?" | **Hyperparameter sweep** — vary knobs around the operating point; a method that only works at one setting is fragile |
| "Will it hold up off-distribution?" | **Generalization test** — evaluate on held-out opponents/decks never seen in training; our local-vs-ladder inversions are exactly this failure |
| "Why is it failing?" | **Error analysis** — read ~20 failure cases and categorize before hypothesizing a fix; the `error-analysis` user skill has the full method |
| "Is our measurement even working?" | **Negative control / sanity check** — a change that *shouldn't* help (shuffled labels, random agent) must measure as null; if it "helps", the eval is broken |

Most ideas here reduce to one of the two ablation designs, detailed next.

### Ablation shape 1 — scaling ("does more X help?")

For ideas like "a bigger model would win more", "more training data would
help", "train longer". Never test one new size against one old size — that's
two points and no trend. Sweep the variable, hold everything else fixed:

- **Arms:** 3–4 values of the variable spanning ~an order of magnitude where
  feasible (e.g. model params: current size, ½×, 2×, 4×), same data, same
  steps, same seed policy.
- **Read the curve, not the endpoints.** Adopt only if the metric improves
  monotonically (or clearly at the top end) *and* the cost is worth the slope.
  A flat curve is a real result: X doesn't matter at our scale — write it down.
- **This repo's precedent:** Metamon's 15M/50M/200M grid had to be rescaled,
  not copied literally — we have ~2 orders of magnitude less data, so scaling
  arms must be sized to *our* data budget, not the paper's.

### Ablation shape 2 — feature/component ("which of these n things matters?")

For ideas like "we have n features (or n components, n reward terms, n data
sources) — which are actually good?". Two standard designs; choose by cost:

- **Leave-one-out (default):** train/evaluate the full system, then n runs each
  removing exactly one feature. A feature whose removal doesn't hurt is dead
  weight; a feature whose removal *helps* is actively harmful — both are
  publishable findings.
- **Add-one-in (when n runs are too expensive or features are suspected weak):**
  start from a minimal baseline, add features one at a time in order of
  hypothesized importance; stop when the marginal gain flatlines.
- **Report a table:** one row per arm, columns = arm, metric ± spread, Δ vs
  full/baseline. Same eval protocol for every row (same opponents, same game
  count), or the table is noise.
- **Interactions:** if two features are suspected to only work together, add
  one arm removing the pair — but resist the full 2ⁿ grid; name the ≤2
  interactions worth testing and skip the rest.

### All designs

- Same seeds/eval-set across arms; report spread across ≥2 seeds when a
  training run is cheap enough, and say "single seed, treat as provisional"
  when it isn't.
- Run arms cheapest-first — a trend visible at small scale kills or confirms
  the idea before the expensive arm is paid for.
- The card's **Independent variable** field names the swept axis; the arms go
  in **Metric & protocol**; the design name goes in the title line.

## Coaching, not just compliance

The goal behind this skill is Rami becoming an ML researcher, so treat each
experiment as a rep, not a chore:

- When proposing a design, say *why this design and not the adjacent one* in
  one sentence ("ablation, not sweep, because the question is attribution, not
  sensitivity").
- After each result, ask Rami what it changed about his beliefs and log his
  one-line answer on the card (the **Belief update** field) — the update step
  is his rep, not yours to fill in for him.
- After each result, name the transferable lesson — the thing that would apply
  to any ML project, not just this repo.
- Occasionally point at the relevant literature move: the experiment card is a
  mini pre-registration; the ablation table is what reviewers ask for in every
  paper; the negative control is how you audit an eval.

## Hard rules for this repo

- **One variable per experiment.** If the idea bundles three changes, split it
  or ablate; a win you can't attribute teaches nothing.
- **Local rankings lie.** Local Glicko has inverted against the real ladder
  twice. Any "better/worse" conclusion must go through the `leaderboard-check`
  skill before it's written down as a result.
- **Deck comparisons go through the `deck-selection` skill** — otherwise the
  experiment measures the training distribution, not the idea.
- **Rule out the fallback confound.** Before crediting or blaming a model
  change, run the `run-fallback-diagnostic` skill — a "worse model" that is
  actually a silent `_safe_choice` fallback invalidates the experiment.
- **Preflight before training runs** (disk/RAM/competing processes) and report
  projections instead of stopping for approval — per standing feedback.

## Why this shape

The repo's own history is the argument: two local-vs-ladder inversions, a
one-episode control deck that looked plausible until measured, and a silent
fallback that masqueraded as a model regression. Ideas that skipped the
experiment step have repeatedly produced confident wrong conclusions here.
