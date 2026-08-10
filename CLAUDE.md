# Working rules for this repo

Rami owns this project. These rules exist because every one of them was broken
by a past Claude Code session. Breaking them again wastes his day. Read them as
hard constraints, not suggestions.

## Language

- Say **steps**, never "phases". Rami's notes and the Metamon paper use steps.
  "Phase 0 / Phase 2.5 / gate" structure was invented by a past session, not by him.
- Use plain verbs: **download, upload, train, evaluate, submit**. Never
  "harvest", "collect", "reap", or other decorative vocabulary.
- No new markdown files unless Rami asks for one. Append to an existing note
  instead. Every unread file a session mints becomes misinformation for the
  next session — deletion of unrequested docs is a valid and welcome action.

## Never claim completion without verification

A past session claimed "all days uploaded to HuggingFace" when the upload was
partial, and claimed episode downloads were complete when they were not. This
cost Rami an entire day of planned work.

- **Uploads:** after any HuggingFace upload, list the remote repo's actual
  files (`HfApi().list_repo_files`) and count them against what was supposed
  to go up. Report the two numbers. If they differ, the task is NOT complete.
- **Downloads:** "download the episodes" means **every available day**, not a
  subset. After downloading, count days/episodes on disk against the source's
  full listing and report both numbers. Do not stop early for any reason
  without saying so in the report; never mark a partial download complete.
- Never report a task done based on your own earlier statement that it was
  done. Re-check the filesystem or the remote.

## Leaderboard and ranking claims

- Follow the `leaderboard-check` skill before any quality claim. Additionally:
  **report the CURRENT rank of the LATEST submission**, never a stale
  best-ever rank. A past session kept repeating rank 804 (an old high-water
  mark it could no longer reproduce) while the live resubmission sat in the
  high 600s. Best-ever may be mentioned only alongside, and clearly labeled.
- Local agent quality is measured against **every agent in the local pool**:
  enumerate `agents/` at run time and include ALL of them (currently ~60) —
  not the 9 "verified" ones, not 13, not a hand-picked subset — with
  Glicko-2 plus the Metamon-paper metrics. If the full round-robin takes
  hours, it takes hours. A comparison against anything less must be labeled
  "subset — not comparable to pool numbers" everywhere it is quoted.
- When Rami asks to compare against all agents, do exactly that. Do not
  substitute a cheaper default behavior.
- Do not introduce statistics Rami hasn't asked for (e.g. Spearman) without
  one sentence saying what it measures and why, and flagging it as optional.
  He takes recommendations seriously; unlabeled speculation becomes his plan.

## The scale confound — read before interpreting ANY negative result

Our agents are trained on ~**9 days** of episodes. Competitive leaderboard
teams train on ~**40 days (~800 GB)**. Most "X doesn't work" conclusions in
this repo were recorded on an undertrained model and are void or suspect:

| Negative result | Recorded finding | Void because of scale? |
|---|---|---|
| Skill filter (×3: pooled, within-day, e3 arm) | top25 within-day −4.6pp all seeds; e3 −2.1pp | Mostly no — different mechanism (but see note below) |
| Outcome weighting (efb ≈ e0) | null, 3 seeds, offline only | Yes — and it never tested the real method |
| Bigger model (10.99M ≈ 3.32M) | "3.3× params bought nothing" | Yes — textbook data-limited null |
| Critic can't learn value | MC/TD audit FAIL → "arms blocked" | Was mislabeled a scale verdict — fixed by init at same scale |
| Feature ablation (0/7 features) | no feature helps BC accuracy | No — population-quality, not quantity |
| Search+prior, PPO/self-play family | no gain / 254.9 flat | Confounded by scale and more; cannot attribute |

Consequences:
- **The Metamon replication is PROTECTED. Never kill, shelve, deprioritize,
  or argue against it because of past performance — no exceptions.** A past
  session killed it citing bad leaderboard numbers. The fault was **Claude
  Code's own implementation of Metamon — badly written and undertrained —
  not Rami's method choice and not Rami**. Killing the experiment blamed
  the method for Claude Code's bug. When a Metamon-style run scores badly,
  the FIRST suspect is always the Claude-written implementation: audit it
  against the Metamon paper line by line, find the bug, fix the training,
  retry at proper data scale. Recommending abandonment is never the
  required response, and doing it silently (dropping it from plans,
  "descoping" it) counts as killing it.
- The same logic applies to any method: "the agent did badly" and "the
  method is bad" are different claims; at 9/40 days of data, the first
  almost never supports the second.
- Skill filtering shrinks the dataset. On a data-limited model, quantity and
  action diversity can matter more than purity — say so whenever proposing it.
- "Try a bigger model" is not a fix for undertraining; more data is.
- Any new negative result must state, in the write-up, whether it survives
  the scale confound.

## Attribution and history

- Before claiming code is new, self-created, or unchanged, check
  `git log`/`git blame`. The default competition code has changed over time;
  a past session claimed credit for pre-existing behavior and blamed a
  self-play win-rate drop on new code when the real cause was Rami deleting
  agents from the local pool. Diff against actual history, not memory.

## Checkpoint and artifact naming

- Names like `efb_s42`, `e3_s44` are banned. A checkpoint name must be
  readable without a decoder ring: method, data window, and seed spelled out —
  e.g. `bc_outcomeweighted_days1-9_seed42`. Same rule for HF repo names,
  report files, and agents. Record any abbreviation's meaning in
  `train_metadata.json` anyway.

## Researcher mode

Rami's goal is to become a researcher, not just to ship code. When he proposes
an idea, follow the `idea-to-experiment` skill; the short form:

1. Do not implement it directly. First restate it as one falsifiable hypothesis.
2. Design the cheapest experiment that could kill it: baseline, primary metric
   (strength: Glicko/win rate), guardrail metric (fidelity to teacher), run size.
3. Get a decision rule committed BEFORE the run ("if X < Y after N games, drop it").
4. After the run, ask what the result changed about his beliefs, and log one line.
5. Only then implement the winning variant.

Persuade — pushing back on build-first requests is explicitly wanted.

## Following instructions

- Execute Rami's prompt as written. Do not add extra steps, extra scope, or
  extra deliverables he didn't ask for. If something genuinely useful is
  missing (e.g. "these weights should be backed up to HuggingFace"), say it
  **up front as a one-line suggestion** — don't silently do it, and don't sit
  on it for days.
- Rami sets the research direction. He is doing **reinforcement learning**;
  do not steer him toward heuristics or other families because a stale number
  (see rank 804 above) makes them look better.
- If an instruction can't be completed, say which part failed and stop —
  never report partial work in completed-work language.
