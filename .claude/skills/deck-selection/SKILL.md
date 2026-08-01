---
name: deck-selection
description: Method for deciding which deck the agent should pilot in the Pokémon TCG AI Battle Challenge. Use for any question of the form "which deck is best", "what's the meta deck", "compare model performance across decks", "should the agent play one deck or several", or when about to benchmark decks against each other. Enforces the ordering that keeps the answer from being a measurement of the training distribution.
---

# Deck selection

Assumes `ptcg-repo-context`. Do not restate repo constraints; load that skill instead.

## The failure this skill exists to prevent

"Which deck is best?" reads like one question. It is three, and running them out of
order yields a confident wrong answer:

1. **Which decks exist** — a count, over two *different* populations
2. **Which deck wins most against the field** — a measurement, and field-relative
3. **Which deck my model plays best** — a **confound**, because BC only learned decks it saw

If deck A is 40% of the training corpus and deck B is 2%, an A-over-B win rate measures
the **data distribution**, not the deck. The naive run — point the IL agent at several
decks, compare win rates, pick the winner — has this baked in and produces a number that
looks clean and means nothing. Order is load-bearing: ① → ② → ③ → ④.

## ① Enumerate — two populations, not one

They are different lists and conflating them is the first mistake:

- **Corpus decks** — what appears in `splits/train-2026-07-26`. This bounds what BC could
  possibly have learned.
- **Field decks** — what the opponent pool actually pilots. This is what must be beaten.

Sources for the field list: the 9 `agents/*/deck.csv` files already in the repo, the
public roster notes, and other Kaggle notebooks not yet mined.

Deliverable: two tables — deck identity → episode count (corpus), and deck identity →
which pool agents pilot it (field). Deck identity needs a **defined key**; decide and
write down whether it is the exact 60-card multiset, the ace/carry Pokémon, or a
clustering, because "the same deck" is otherwise undefined and every later count depends
on it.

⚠️ Cheapest possible first move: determine which deck the existing rule-based agent
pilots (~5 min). It gates the frozen control deck in the BC-vs-baseline comparison — if
the control deck does not match, that comparison is already deck-confounded.

## ② Audit familiarity before measuring anything

For every deck that will appear in ③, record its **training-episode count**. Then:

- Decks below a stated floor are **unmeasured**, not bad. Report them as such.
- Every win-rate cell in ③ carries its deck's episode count alongside it.
- A deck's win rate may only be compared to another deck's when both clear the floor.

This is the same distinction as the open question "does deck-specific data beat
deck-volume data" — cross-reference it rather than re-deriving it.

## ③ Measure — deck × opponent, with σ

**The harness has no deck axis.** Deck is bound to agent identity via module-level
`my_deck`, injected from `<agent_dir>/deck.csv`. Varying deck while holding policy fixed
requires adding that mechanism — a deck override — as an explicit, reviewable change.
Do not fake it by cloning agent directories.

Rules:

- **Opponent pool is the public roster, not our own agents.** Beating our own three
  baselines was never evidence about the ladder. A public rule-based notebook has scored
  **1208**, above the top-8 cutoff of **1122** — that is the bar, and whatever it pilots
  belongs in the pool.
- **Mirrored pairs only** (`play_match` already does this): seats swapped, first-player
  advantage cancelled.
- **σ on every cell.** Refuse to rank two decks whose intervals overlap; say "not
  separated at this sample size" and state how many games would separate them.
- **Hold everything else fixed**: same policy checkpoint, same seeds, same game count.
- Glicko-1 ratings persist and compound across runs in `reports/glicko_ratings.json` —
  if the run should not pollute the standing ratings, say so and isolate it.

Deliverable: a deck × opponent matrix as a real spreadsheet (`xlsx`), cells = win rate ±
σ, with the ② episode count per deck as a visible column. Not a hand-maintained markdown
table.

## ④ Decide — two real forks

Neither is a formality; both change what gets built next.

- **One deck vs a portfolio.** Self-play across meta decks argues portfolio; a short clock
  argues one deck done properly.
- **Best-vs-field vs most-learnable-from-data.** The Orbit War imitability finding —
  cloneability tracks predictability, not score — applied to *decks* rather than players.
  If the most-learnable deck wins, "pick the strongest meta deck" was the wrong axis from
  the start.

Worth an ADR (`engineering:architecture`), because these are decisions with consequences,
not measurements.

## Done means

One defensible sentence:

> *The agent pilots **X**, because against field **Y** it wins **Z% ± σ**, and I had **N**
> training episodes of X — enough to believe the number reflects the deck and not my data.*

If any of X, Y, Z, σ, or N is missing, the question is not answered yet.

## Skill routing

| Step | Skill |
|---|---|
| ②, ③ — experiment design, pool composition, games per cell | `engineering:testing-strategy` |
| ③ — the matrix artifact | `xlsx` |
| ④ — the fork | `engineering:architecture` |
| encoder change for a deck override | `engineering:code-review`, aimed at opponent-private leakage |

`debug` is for when the harness misbehaves, not for this question.
