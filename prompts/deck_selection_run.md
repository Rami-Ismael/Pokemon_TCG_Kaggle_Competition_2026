# Deck selection — full run prompt

Paste everything below the line into Claude Code from the repo root.

---

Load the `ptcg-repo-context` and `deck-selection` skills before doing anything else, and
follow `deck-selection`'s ①→②→③→④ ordering exactly. The ordering is the point of the
task: run ③ before ② and the result measures my training distribution instead of the
decks.

**Why this is happening now (context from the session that queued this task, 2026-08-02).**
A local 15-agent benchmark round-robin (`--games 6`) had `plamen06_steel` (a newly
recruited Archaludon ex / Metal-type agent, see `notebooks/reference/INDEX.md`) finish
#1 by Glicko (1762.7, GXE 73.3%), ahead of `agent_core_improved`. Before reading that as
"switch decks," I found a confound: `plamen06_steel`'s own code defaults `USE_SEARCH=1`
(real, working multi-candidate search), while `agent_core_improved` defaults
`USE_SEARCH=0` in that same benchmark run (per Gate 0's finding that real search loses to
pure heuristic on the Lucario deck). So that #1 finish is at least partly "agent with
search on vs. agent with search off," not "deck A vs deck B" — and separately, the two
agents' Wilson 95% CIs overlap ([66.2,78.7] vs [59.7,73.0] for `proto`), so it wasn't even
statistically resolved. This task exists to actually disentangle deck from
implementation/policy-familiarity, which is exactly the trap `plamen06_steel`'s result
fell into. Full writeup: search this session's transcript / `notes/` for
"plamen06_steel" if more detail is needed; not re-derived here.

**Checkpoint to hold fixed across all deck arms (per ③'s explicit requirement).** Use
whatever `models/il_agent/` is at task-start time UNLESS a Phase 2 checkpoint sweep
(`models/il_agent_sweep/{0.5,1,2,4,8}ep/`, see `notes/phase2_report.md`) has since
concluded with an explicit winner recorded in `notes/` — if so, use that one instead and
say which. Do not silently pick a different checkpoint without checking.

**Deck.csv count is now 11, not the older "9" some earlier notes may reference** — see
`find agents -name deck.csv` for the live count before trusting any stale number in a
note.

**Goal.** Decide which deck my agent pilots on the ladder, and produce the evidence that
justifies it. Not a vibe, not a markdown table — a defensible sentence backed by a matrix
with uncertainty on it.

**You are running all four steps in one session, so you are also responsible for the
review I would otherwise do between them.** Three hard gates below. At each one, stop,
print your findings, and state explicitly whether you are proceeding and why. If a gate
fails, do not continue to the next step — write up what you found and stop. A partial
answer with a clear reason is worth more to me than a complete one built on a broken
foundation.

---

## ① Enumerate

Define a **deck identity key** first and write it down: exact 60-card multiset, ace/carry
Pokémon, or a clustering. Justify the choice in two sentences. Every count downstream
depends on it, so this is not a detail.

Then build two separate tables:

- **Corpus decks** — parse `splits/train-2026-07-26` (resolve via
  `il_dataset.resolve_split_dir`, not the config path constant) and count episodes per
  deck identity. Note whether decks are directly recoverable from episode JSON or must be
  reconstructed from played cards; if reconstructed, state the reconstruction rule and its
  failure modes.
- **Field decks** — the `agents/*/deck.csv` files (count them fresh, don't trust a cached
  number — 11 as of 2026-08-02) plus the repo-root `deck.csv`, mapped
  to which pool agent pilots each.

Also answer the 5-minute question: **which deck does the existing rule-based agent
pilot?** It gates the frozen control deck in my BC-vs-baseline comparison.

> **🛑 GATE 1.** If deck identity is not recoverable from the episode JSON at acceptable
> confidence, STOP. Report what *is* recoverable and what the honest alternatives are
> (proxies, partial reconstruction, a smaller well-identified subset). Do not proceed on a
> reconstruction you do not trust — a wrong deck key silently corrupts ② and ③ both.

## ② Familiarity audit

For every deck that will appear in ③, record its training-episode count. Propose an
episode-count floor and justify the number. Decks below it are **unmeasured**, not bad,
and must be labelled that way everywhere they appear.

> **🛑 GATE 2.** If fewer than 3 decks clear the floor, STOP and report. A deck comparison
> across 2 arms where one is thinly represented is not an experiment. Tell me whether the
> fix is a lower floor, more data, or abandoning the cross-deck comparison in favour of
> "play the deck I actually have data for."

## ③ Measure

`scripts/benchmark_agents.py` has **no deck axis** — deck is bound to agent identity
through module-level `my_deck`. Add a deck-override mechanism as an explicit, minimal,
reviewable change. Do not clone agent directories to fake it. Show me the diff and explain
the design in three sentences before running anything long.

Then run the deck × opponent matrix:

- Policy checkpoint held **fixed** across all deck arms — this is the whole point
- Opponent pool = the **public roster** agents, not just our own; `random_legal` included
  as a floor check only
- Mirrored pairs (the harness already does this)
- ≥3 seeds; σ on every cell
- State the total game count and wall-clock estimate **before** launching, and respect the
  single-threaded evaluator envelope
- Say whether this run should write to the persistent `reports/glicko_ratings.json` or be
  isolated, and act on your answer

> **🛑 GATE 3.** Before the full run, do a smoke run at minimum game count and show me the
> matrix shape, one populated cell, and the timing extrapolation. If the full run is over
> ~2 hours, stop and propose a cheaper design instead of launching it.

## ④ Decide

Two forks, both real:

- **One deck vs a portfolio**
- **Best-vs-field vs most-learnable-from-data** — rank decks by how learnable they were
  (action entropy / policy self-consistency on that deck's episodes) as well as by win
  rate, and put both on the chart. If the orderings disagree, that is the finding, and say
  so loudly.

## Deliverables

1. `reports/deck_selection.xlsx` — deck × opponent matrix. Cells = win rate ± σ. A visible
   column for each deck's training-episode count from ②. Use the `xlsx` skill.
2. `reports/deck_selection.md` — the writeup, ending in the one sentence:
   *"The agent pilots X, because against field Y it wins Z% ± σ, and I had N training
   episodes of X."* If any of X/Y/Z/σ/N is missing, say which and why rather than
   inventing it.
3. The deck-override diff, isolated in its own commit.
4. A short **"what would change my mind"** section — the specific result that would
   reverse the recommendation.

## Rules

- Distinguish measured from assumed in every claim. If a number came from a blog, a
  notebook, or my notes rather than a run you did, mark it.
- Never rank two decks whose intervals overlap. Say "not separated at N games" and state
  the N that would separate them.
- If you find that the honest answer is "the data cannot support this comparison," that is
  a valid and useful outcome. Report it rather than manufacturing a ranking.
- Do not touch the training pipeline. This task measures; it does not retrain.
