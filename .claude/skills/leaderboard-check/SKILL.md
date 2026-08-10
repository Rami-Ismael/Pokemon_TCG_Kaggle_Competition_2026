---
name: leaderboard-check
description: Read the real Kaggle leaderboard, and record/track our submissions in the per-submission ledger, for the Pokémon TCG AI Battle Challenge. Use at the START of any task that benchmarks agents, interprets Glicko or win-rate results, decides what to submit, or writes up results; IMMEDIATELY before writing "better", "best", "beats", "improves", or "top-ranked" about any agent or method; right after any kaggle submission (log its approach/config/rationale — Kaggle's submit message caps at 500 chars, the ledger doesn't); and whenever asked what a past submission was, or how a submission's score moved over time. Local rankings have inverted on the real ladder twice — never repeat a local claim the ladder contradicts.
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

## ② Per-submission ledger — record what a submission IS, and watch its score move

Kaggle's submit-message field caps at 500 characters and each submission shows
ONE live score that drifts for hours (55215267 read 232.1 → 265.5 → 290.8 →
299.1 → 319.0 in one day). `reports/submission_ledger.jsonl` (append-only,
driver `scripts/submission_ledger.py`) is where the full story lives: unlimited
metadata per ref + a timestamped score timeline.

After every submission — and periodically while scores are settling:

```
uv run python scripts/submission_ledger.py refresh
```

One cheap submissions-list API call. Appends a reading for every ref whose
score/status moved (deduped — re-running immediately appends nothing), and
auto-stubs refs it has never seen, preserving their 500-char Kaggle
description verbatim so nothing is lost even for un-logged submissions.

Right after submitting, log the detail the submit message couldn't hold:

```
uv run python scripts/submission_ledger.py log --ref <ref> \
  --deck "..." --git-sha <sha> --checkpoint-sha256 <sha> \
  --config "env flags / hyperparams, no length limit" \
  --local-result "local evidence at submit time" \
  --displaces "what this pushes out of the 2-slot active set" \
  --expects "falsifiable score prediction" \
  --approach "full approach description"      # or --approach-file <path> / '-' for stdin
```

A later `log` for the same ref amends per field (append-only file; later-wins
on fold) — use that to correct a field, never edit the jsonl by hand.

Read it back:

```
uv run python scripts/submission_ledger.py show              # table: first→latest score, drift, #reads
uv run python scripts/submission_ledger.py show --ref <ref>  # full metadata + score timeline
```

Division of labor: this ledger is per-REF and machine-readable;
`reports/leaderboard_history.jsonl` (written by check_leaderboard.py) tracks
the TEAM (rank, active set). These two jsonl files are the ONLY submission
records — the old `notes/scores.md` table was purged 2026-08-06; do not
recreate it.

Ledger gotchas (all hit for real on 2026-08-04):

- An errored submission (e.g. 55149689, `SubmissionStatus.ERROR`) never gets a
  score; `show` prints ERROR for it, not pending.
- A ref submitted seconds ago records as PENDING/score-null — that's correct;
  `refresh` again later and the scored reading lands as a second timeline entry.
- "0 new score readings" from `refresh` means no score moved since last run —
  a data point, not a failure.
- Trust the stub's `kaggle_description` over hand-typed `log` fields when they
  disagree — it's what was actually said at submit time (this caught a wrong
  USE_SEARCH claim on 55228113's first `log`).

## ③ Rules the numbers are subject to

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

4. **Every read-back goes through `submission_ledger.py refresh`** (never
   hand-edit the jsonl). Record the local-vs-ladder residual via a `log`
   amendment on the ref — that residual is the running answer to
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
