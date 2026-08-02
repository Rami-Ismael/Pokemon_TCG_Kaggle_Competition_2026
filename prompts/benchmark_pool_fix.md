# Benchmark pool — make it predict the ladder

Paste everything below the line into Claude Code from the repo root.

---

Load the `ptcg-repo-context` skill before doing anything else.

**Read `notes/` and the vault note `Benchmark Pool — Does It Predict the Ladder` first.
The diagnosis is already done — do not redo it.** Everything below assumes these findings
and your job is to act on them, not re-derive them:

- Local Glicko orders agents correctly against known LB scores (Spearman ρ = 0.90, n = 5).
  **Ordering is not the problem.**
- 224 games/agent → ±6.5 pt CI on a win rate. Ten of thirteen agents are one
  undifferentiated blob. **Resolution is the problem.**
- 8 of 13 agents pilot Mega Lucario ex, so ~62% of every rating is mirror-match.
  **Archetype coverage is the problem.**
- Pool's strongest agent is LB 1084.5; top-8 cutoff is 1141.0. Recruiting *stronger*
  opponents buys nothing yet — they are already 250 pts above us. **Do not go shopping for
  strong agents. Go shopping for different decks.**

**Goal.** Make `scripts/benchmark_agents.py` produce a number I can act on: intervals
instead of point estimates, and an opponent field that contains the deck classes currently
winning the ladder. Three steps, three gates. At each gate stop, print findings, and state
whether you are proceeding and why.

---

## ① Honesty pass on the existing harness (do this first — it guards the rest)

Cheap, and without it steps ② and ③ produce more numbers I will over-read.

- **Report intervals.** Wilson 95% CI on every overall win rate and every matrix cell.
  Print them. When two agents' intervals overlap, the output must say so rather than
  ordering them — a `~` tie marker in the standings, not a silent rank.
- **Record wall clock** per run and per pairing into `reports/agent_benchmark.json`. It is
  absent today, which makes every "can I afford a deeper run" question unanswerable
  without a rerun. Gate ③ depends on this number.
- **Common random seeds.** Check whether the harness can seed deck shuffles/opening hands
  identically across agents so the same variance is subtracted from every arm. If it can,
  wire it behind a flag and report the variance reduction you measure. If the engine does
  not expose seeding, say so plainly — this is a real possibility, not a formality, and
  it changes the cost arithmetic in ③.

> **🛑 GATE 1 — the Glicko persistence trap.** `reports/glicko_ratings.json` accumulates
> across runs. Adding agents in ② changes the opponent field, which changes what every
> existing rating means, and the persisted history will silently blend two incomparable
> pools. Decide and justify in three sentences before touching anything: reset the file,
> version it per pool composition, or namespace the new pool separately. Do not proceed
> with the old file in place by default. Whatever you choose, record the pool roster hash
> alongside the ratings so a future run can tell whether it is comparable.

## ② Recruit for archetype, not strength

Missing from the pool entirely: **Crustle · Great Tusk · Grimmsnarl · Archaludon ·
Garchomp · Azumarill · Terrakion · Starmie**. Missing as *techniques*: LibraryOut
(deck-out), anti-wall, meta router/portfolio, belief policies, replay clone.

**Priority is Crustle and Great Tusk / LibraryOut.** A wall archetype and a deck-out
archetype. Nothing in the pool wins by running the opponent out of cards, so nothing in
the pool tests whether our aggro-tempo heuristics have an answer to the thing the ladder
leader does.

- Source from the 69-agent roster in the vault note `Public Agent Roster — Makimakiai
  Matchup Matrix` and the Kaggle Code tab. Follow the crawl and wiring conventions already
  in `notebooks/reference/INDEX.md` — mirror the notebook verbatim, strip only deck-load
  and submission-packaging cells, keep the `agent()` body unchanged.
- **Selection criterion is deck archetype and win condition, not stated LB score.** A weak
  Crustle agent is worth more to this pool than a strong Lucario one. Say out loud, per
  candidate, which archetype it fills.
- **Safety-review every foreign notebook before executing it**, and record the review in
  `INDEX.md` the way the existing honesty flags do. `eval`/`exec`/network/subprocess are
  disqualifying unless you can decode and read the payload, as was done for
  `mechi22-alakazam`. Do not run code you have not read.
- Note that `seokjeongeum/max-elo-1208-libraryout-w-crustle-great-tusk` — the obvious first
  target — was **403 / unreachable as of 2026-08-01**. Expect to need a substitute. Do not
  fabricate it.

> **🛑 GATE 2.** If you cannot find a safe, runnable agent for *either* Crustle or a
> LibraryOut/deck-out win condition, STOP and report. Do not backfill with more Lucario or
> Alakazam agents to hit a count — that makes the monoculture worse while looking like
> progress. Tell me the honest options: write a minimal deck-out agent myself, settle for a
> different wall archetype, or accept the gap and label it in the output.

## ③ Buy resolution where it matters

Do not deepen the full round-robin. `random_legal`, `dedquoc_rule_engine` and `il_agent`
are already resolved — their gaps are outside the noise floor. Spend the games on the blob.

- Define a **top sub-pool** (roughly the six agents whose intervals mutually overlap, plus
  the new ② recruits) and raise games/pair there only.
- Targets from the diagnosis: **44 mirrored pairs → ±3 pts**, 100 pairs → ±2 pts. Current
  is 8. Pick the target the wall-clock number from ① actually supports.
- Keep `random_legal` in as the floor check regardless of depth.

> **🛑 GATE 3.** Before launching anything long, print the projected wall clock from ①'s
> measurement and the depth it buys. If ±3 pts costs more than a few hours, stop and tell
> me — say whether common random seeds from ① close the gap, and whether a smaller
> sub-pool at full depth beats a larger one at half depth. I would rather have four agents
> separated with confidence than ten ranked by noise.

---

## Deliverables

1. The diff to `scripts/benchmark_agents.py`, explained in a few sentences before you run
   anything long.
2. New agents under `agents/`, with `INDEX.md` updated including the safety review.
3. A rerun of the benchmark with intervals, and a standings table that refuses to order
   overlapping agents.
4. Three sentences in `notes/` on whether the recruits changed the picture — specifically
   whether any existing agent's ranking moved once non-Lucario opponents were weighted in.
   That is the actual test of whether the monoculture was distorting things.

## Do not

- Do not touch the pinned winning checkpoint in `models/il_agent_winning_827.8/`. Its
  sha256 is recorded in `PROVENANCE.md` and it must stay byte-identical.
- Do not recruit agents on the strength of a slug-claimed LB score. Two of the five paired
  points in the diagnosis are unverified slug claims and they are the weakest part of it.
- Do not report a rank without an interval next to it.
