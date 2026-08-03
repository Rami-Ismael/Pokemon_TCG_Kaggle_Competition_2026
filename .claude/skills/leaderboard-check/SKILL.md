---
name: leaderboard-check
description: Read the real Kaggle leaderboard and our submission scores before evaluating, comparing, or reporting on agents in the Pokémon TCG AI Battle Challenge. Use at the START of any task that benchmarks agents, interprets Glicko or win-rate results, decides what to submit, or writes up results; IMMEDIATELY before writing "better", "best", "beats", "improves", or "top-ranked" about any agent or method; and right after any kaggle submission. Local rankings have inverted on the real ladder twice — never repeat a local claim the ladder contradicts.
---

# Leaderboard check

Assumes `ptcg-repo-context`. The ladder is the quantity being optimized; everything
local is a proxy for it. This skill exists because the proxy has been wrong.

## The failure this skill exists to prevent

Two confirmed inversions, one day apart:

| Date | Local claim at submit time | Ladder verdict |
|---|---|---|
| 2026-08-02 | `improved_prob_main` "top-ranked among our agents, Glicko 1720.2, 64.3% WR, ahead of agent_core_improved (1665.7)" | **701.6**, while the agent it "beat" sat at **804.0** |
| 2026-08-03 | `ppo_u120832` "beats s2_e1_s43 head-to-head 87.5% [69.0, 95.7], Glicko 1478±71 vs 1300±71, non-overlapping" | **232.1–265.5**, while s2_e1_s43 sat at **395.0** |

The local round-robin measures strength *within our 13-agent pool*; the ladder
measures strength against ~6,200 heterogeneous teams. Different quantities. Note
the one time local ordering did hold (55190924 vs 55190932) it was a same-checkpoint
ablation — trust local orderings less the more the two arms differ in kind.

## ① Read the ladder before saying anything about quality

```
uv run python scripts/check_leaderboard.py
```

Prints rank / field size / team score / top-8 cutoff, best-ever vs newest
submission, flags divergences, and appends a snapshot to
`reports/leaderboard_history.jsonl` (so residuals are trackable over time).

Fallback if the script is unavailable:

```
uv run kaggle competitions submissions -c pokemon-tcg-ai-battle
uv run kaggle competitions leaderboard -c pokemon-tcg-ai-battle --show
```

- Auth is OAuth: `uv run kaggle auth login`. kaggle.json is not the mechanism.
- ⚠️ Slug trap: `pokemon-tcg-ai-battle-challenge-strategy` is the separate writeup
  track with an empty leaderboard. The ladder lives at `pokemon-tcg-ai-battle`.

## ② Rules the numbers are subject to

1. **Ladder is ground truth; local numbers are hypotheses about it.** Until a
   submission's score is read back, the only honest phrasing is *"wins X% ± σ
   locally; unverified on the ladder."* "Better", "best", "beats", "improved",
   "top-ranked" require a ladder read that supports them. If the ladder
   contradicts a local result, report the contradiction — never restate the
   local claim as if the ladder read hadn't happened.

2. **Ladder scores are live ratings, not fixed grades.** A same-build resubmit
   of the 804.0 bundle scored 699.0; one submission's score moved 232.1 → 265.5
   between two reads minutes apart (2026-08-03); an earlier pair read 252.8/437.3
   intra-day and 190.3/397.3 later. Observed same-build spread is ~±100. Do not
   call a ladder delta inside that band an improvement or a regression without
   corroboration (repeated reads, or a margin well beyond it).

3. **The team score rests on the ACTIVE set, not best-ever.** The leaderboard
   counted 2 submissions for the team while we had 9 completed — recent
   submissions displace older ones. This is how the team fell from 804.0
   (rank ~865) to 395.0 (rank ~5408) on 2026-08-03: two experimental submissions
   pushed the 804.0 agent out of the active set. **Before submitting, state what
   the team score currently rests on and what this submission displaces.** An
   experiment that "can't hurt" can cost 4,500 ranks.

4. **Every read-back updates `notes/scores.md`** (append-only per its Gate 5
   protocol: fill a PENDING score once, never edit it after; append a new row
   for rebuilds; note the read-back date since ratings drift). Record the
   local-vs-ladder residual alongside — that residual is the running answer to
   "is our benchmark pool predictive?", and right now the answer is no.

## Done means

Quality claims in reports and submit messages carry this shape:

> *On the ladder we are rank **R/N** at **S** (active set: **A**; best-ever **B**).
> Locally, X beats Y at **Z% ± σ** — [consistent with / contradicted by] the
> ladder, which reads **Sx** vs **Sy**.*

If the ladder half of that sentence is missing, the claim is not ready to make.

## Skill routing

| Need | Skill |
|---|---|
| repo constraints, submission build scripts | `ptcg-repo-context` |
| which deck to pilot (separate confound) | `deck-selection` |
| designing a local benchmark that predicts the ladder | `engineering:testing-strategy` |
