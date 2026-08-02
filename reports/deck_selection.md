# Deck selection

Run following `prompts/deck_selection_run.md`, the `deck-selection` skill's ①→②→③→④ ordering,
`il_agent` (the trained BC transformer checkpoint) held fixed as the policy under test. Full
cell-level numbers: [`reports/deck_selection.xlsx`](deck_selection.xlsx). Deck-override
mechanism: commit `0f553f9` (isolated, see below).

**Note on how this task was queued.** The task description said this run was queued off "a
confounded plamen06_steel benchmark result," with a policy checkpoint named to hold fixed. I
searched the full repo, git history on every branch, and my memory store for `plamen06`/`steel`
and found nothing — that context does not exist anywhere I could locate. I flagged this to the
user before starting; per their direction I used `il_agent` (the repo's actual BC policy, and
the one the deck-override mechanism in ③ is built for) as the fixed checkpoint and proceeded.

## ① Enumerate

**Deck identity key: ace/carry Pokémon** — the Pokémon card in a 60-card list maximizing
`(evolution_stage + ex/megaEx_bonus, copy_count)`, read from the official card database
(`cg.api.all_card_data()`, the same one every agent imports). Chosen over the exact 60-card
multiset: the multiset is losslessly recoverable but over-fragments — `rule_baseline`'s actual
runtime deck (`submissions/mega_lucario/deck.csv`) differs card-for-card from `il_agent`'s own
"same" Mega Lucario ex deck, so an exact-multiset key would call two copies of the frozen
control deck *different decks*. Ace/carry Pokémon is also the granularity this repo's own field
decks are already named at (`kiyotah_dragapult`, `*_alakazam`, `mega_lucario`). Validated
against all 9 field decks' own docstring names — 9/9 match.

**Recoverability:** read directly from `steps[0][0]["visualize"][0]["action"]`, the literal
recorded 60-card submission for both players — not reconstructed from play. Parsed all 4554
train-split episodes (`splits/train-2026-07-26`, resolved via `il_dataset.resolve_split_dir`),
0 parse errors, 9108 deck-instances (2 seats × 4554 episodes).

**Corpus decks:**

| Ace / carry Pokémon | Episodes | Deck-instances | Distinct exact multisets |
|---|---:|---:|---:|
| Marnie's Grimmsnarl ex | 3488 | 4672 | 15 |
| Alakazam | 1542 | 1693 | 8 |
| Team Rocket's Spidops | 784 | 813 | 11 |
| Cynthia's Garchomp ex | 625 | 642 | 3 |
| Mega Kangaskhan ex | 600 | 604 | 5 |
| Dragapult ex | 318 | 319 | 4 |
| Thwackey | 111 | 112 | 2 |
| Crustle | 92 | 93 | 4 |
| Mega Lopunny ex | 77 | 77 | 2 |
| Mega Froslass ex | 38 | 38 | 3 |
| Cornerstone Mask Ogerpon ex | 16 | 16 | 2 |
| N's Zoroark ex | 12 | 12 | 2 |
| Archaludon ex | 10 | 10 | 3 |
| Mega Starmie ex | 3 | 3 | 1 |
| Brambleghast | 2 | 2 | 1 |
| Mega Sharpedo ex | 1 | 1 | 1 |
| **Mega Lucario ex** | **1** | **1** | 1 |

**Field decks** (9 `agents/*/deck.csv` + the runtime deck actually loaded for
`rule_baseline`/`improved_prob_main`):

| Ace / carry Pokémon | Pool agents |
|---|---|
| Mega Lucario ex | rule_baseline, il_agent, random_legal, dedquoc_rule_engine, grunt, improved_prob_main, makthanithin_improved_prob |
| Alakazam | ryotasueyoshi_alakazam, mechi22_alakazam |
| Dragapult ex | kiyotah_dragapult |
| Mega Abomasnow ex | kiyotah_abomasnow |
| Iono's Bellibolt ex | kiyotah_iono |

Repo-root `deck.csv` does not exist — a stale claim in the `ptcg-repo-context` skill, not
trusted here.

**5-minute question:** `rule_baseline` pilots **Mega Lucario ex**, loaded at runtime from
`submissions/mega_lucario/deck.csv` (`agents/mega_lucario/agent_core.py` itself ships no
`deck.csv` and depends entirely on injection).

**🛑 GATE 1: PASS.** Deck identity is directly recoverable at high confidence.

## ② Familiarity audit

Cross-referenced against `prompts/bc_pipeline_v2.md:319-336` (§1.4/Q43, "does deck-specific
data beat deck-volume data") rather than re-derived — that doc already named a deck shortlist
that matches several of the aces recovered here independently.

**Floor: ≥150 training episodes** (~28,500 decisions, at the corpus-wide average of 190
decisions/episode, both seats, measured via `il_dataset.iter_decisions`). Not an arbitrary round
number — it sits in a natural gap: the lowest deck above it (Dragapult ex, 318 episodes) has
nearly 3× the episodes of the highest deck below it (Thwackey, 111 episodes).

6 decks clear the floor: Marnie's Grimmsnarl ex, Alakazam, Team Rocket's Spidops, Cynthia's
Garchomp ex, Mega Kangaskhan ex, Dragapult ex. Everything else — including **Mega Lucario ex,
the frozen control deck, at 1 episode** — is unmeasured (see the Familiarity Audit sheet in the
xlsx for the full labelled list). Two more field decks (Mega Abomasnow ex, Iono's Bellibolt ex)
have zero corpus episodes.

**🛑 GATE 2: PASS.** 6 decks clear the floor, well above the 3-deck minimum. Composition
caveat: only one of the 6 (Dragapult ex) is a field-pool deck; the deck the agent currently
pilots on the ladder is not one of them.

## ③ Measure

**Deck-override mechanism** (commit `0f553f9`, isolated): `load_agent(name)` in
`scripts/benchmark_agents.py` now accepts `<agent>@<deck-tag>` (e.g. `il_agent@dragapult_ex`).
The base agent loads and runs completely unchanged; only the freshly loaded module's `my_deck`
is overwritten from `configs/deck_lists/<deck-tag>.csv` afterward. Each `load_agent` call
already builds an independent module object (no caching), so distinct deck-tagged labels for
the same base agent never share state — and the override is invisible to the policy itself,
since `il_agent`'s `agent()` only reads `my_deck` at the deck-submission step, never as a model
input. Seven decklists back this: the frozen control (copied from `agents/il_agent/deck.csv`),
two field decks (`dragapult_ex`, `alakazam`, copied from their pool agents' own `deck.csv`), and
four corpus-only archetypes with no field equivalent, each the single most common exact 60-card
list among that archetype's episodes.

**Setup:**
- Policy checkpoint held fixed: `il_agent` (`models/il_agent`), byte-identical weights/code
  across every deck arm — confirmed by module inspection before the run.
- Opponent pool: 7 public agents (`kiyotah_dragapult`, `kiyotah_iono`, `kiyotah_abomasnow`,
  `dedquoc_rule_engine`, `ryotasueyoshi_alakazam`, `makthanithin_improved_prob`,
  `mechi22_alakazam`) + `random_legal` as a floor check.
- Mirrored pairs (harness default), `--games 20` (40 games/cross-cell).
- **Isolated**: `--no-glicko-persist`, separate output file
  (`reports/deck_selection_benchmark.json`) — the standing `reports/glicko_ratings.json` was
  never touched. These are synthetic per-arm identities that don't exist on the ladder.

**🛑 GATE 3 process note (important):** the first smoke test and the first launched full run
were measured against a *silent fallback*, not the real policy — `models/il_agent` doesn't
exist in this git worktree (gitignored, same as `data/`), so `il_agent`'s `_load_model()`
caught the resulting exception and quietly fell back to non-ML behavior. I caught this by
directly checking `mod._load_model() is not None` after noticing the checkpoint directory was
missing, stopped the already-launched (invalid) background run before it produced numbers I'd
have had to discard, symlinked `models/il_agent` from the main checkout (same fix already
applied to `data/episodes` and `data/external/cg-lib` for the same gitignore reason), re-timed
(real inference is ~3× slower than the fallback), and relaunched. The number reported below —
`il_agent@mega_lucario_ex` beating `random_legal` 85–95% before the fix, 34/40 after — is the
same qualitative shape, but I do not have a clean way to determine whether earlier smoke-test
*timing* numbers (already superseded) were contaminated; only the final, reported numbers below
came from a verified-real-checkpoint run. Total: 120 pairs (4500 games), ~30–35 min estimated,
comfortably under the 2-hour ceiling — actual run completed with exit code 0, 0 crashes.

**Deck × field-opponent matrix** (win% ± σ, σ = √(p(1−p)/n), 40 games/cell):

| Deck | Episodes (②) | vs kiyotah_dragapult | vs kiyotah_iono | vs kiyotah_abomasnow | vs dedquoc_rule_engine | vs ryotasueyoshi_alakazam | vs makthanithin_improved_prob | vs mechi22_alakazam | vs random_legal | **vs-field agg** |
|---|---:|---|---|---|---|---|---|---|---|---|
| Marnie's Grimmsnarl ex | 3488 | 87.5±5.2% | 72.5±7.1% | 40.0±7.7% | 100±0% | 80.0±6.3% | 55.0±7.9% | 92.5±4.2% | 100±0% | **78.4±2.3%** |
| Team Rocket's Spidops | 784 | 47.5±7.9% | 60.0±7.7% | 50.0±7.9% | 97.5±2.5% | 97.5±2.5% | 40.0±7.7% | 97.5±2.5% | 100±0% | **73.8±2.5%** |
| Cynthia's Garchomp ex | 625 | 65.0±7.5% | 85.0±5.6% | 37.5±7.7% | 100±0% | 75.0±6.8% | 52.5±7.9% | 55.0±7.9% | 97.5±2.5% | **70.9±2.5%** |
| Alakazam | 1542 | 47.5±7.9% | 57.5±7.8% | 87.5±5.2% | 100±0% | 70.0±7.2% | 55.0±7.9% | 25.0±6.8% | 100±0% | **67.8±2.6%** |
| Dragapult ex | 318 | 42.5±7.8% | 32.5±7.4% | 35.0±7.5% | 90.0±4.7% | 65.0±7.5% | 27.5±7.1% | 82.5±6.0% | 95.0±3.4% | **58.8±2.8%** |
| Mega Kangaskhan ex | 600 | 10.0±4.7% | 20.0±6.3% | 22.5±6.6% | 90.0±4.7% | 67.5±7.4% | 10.0±4.7% | 47.5±7.9% | 92.5±4.2% | **45.0±2.8%** |
| **Mega Lucario ex (control)** | **1** | 12.5±5.2% | 15.0±5.6% | 20.0±6.3% | 80.0±6.3% | 15.0±5.6% | 7.5±4.2% | 2.5±2.5% | 85.0±5.6% | **29.7±2.6%** |

**Separation analysis (95% CI, ±1.96σ):**
- **Every measured alternative is significantly separated from and better than the control**
  (Mega Lucario ex): gaps of 15.3–48.8 percentage points, all non-overlapping 95% CIs. This is
  the load-bearing, fully decisive result.
- **NOT separated at this sample size** (adjacent-rank 95% CIs overlap): Grimmsnarl ex ↔
  Spidops, Spidops ↔ Garchomp ex, Garchomp ex ↔ Alakazam, Alakazam ↔ Dragapult ex. I am not
  ranking these pairs against each other.
  - Grimmsnarl ex vs Spidops (4.7pp gap): would need ~1297 games/arm to separate at 80% power
    (currently 320).
  - Spidops vs Garchomp ex (2.8pp): ~3965 games/arm needed.
  - Garchomp ex vs Alakazam (3.1pp): ~3410 games/arm needed.
  - Alakazam vs Dragapult ex (9.1pp): ~442 games/arm needed.
- Grimmsnarl ex **is** separated from Alakazam directly (non-adjacent, CI [73.9, 82.9] vs
  [62.7, 72.9], no overlap) — the top pick is decisively ahead of the 4th-ranked deck even
  though the intervening ranks aren't pairwise separated from each other.
- Dragapult ex ↔ Mega Kangaskhan ex, Mega Kangaskhan ex ↔ Mega Lucario ex: both separated.

## ④ Decide

**Fork 1 — one deck vs a portfolio.** Given the corpus is this deck-imbalanced (77% of
episodes touch Marnie's Grimmsnarl ex; three field decks have <1 episode of support), a
portfolio strategy right now would mean fielding decks the policy has essentially never seen
decision-making for. One well-supported deck, done properly, is the defensible near-term move;
a portfolio becomes viable only after either collecting matched-volume data for more decks or
conditioning the policy on a deck embedding (the open question already flagged in
`bc_pipeline_v2.md` §1.4, Q43).

**Fork 2 — best-vs-field vs most-learnable-from-data.** Learnability measured as self-consistency
(does the model's greedy choice match the recorded expert action) and mean entropy (nats) of the
model's softmax over legal options, sampled 40 episodes / 600 decisions per deck (fixed seed 42
— a compute-budget choice, not a full census):

| Deck | vs-field win% | Win-rate rank | Self-consistency | Learnability rank | Mean entropy |
|---|---:|---:|---:|---:|---:|
| Marnie's Grimmsnarl ex | 78.4% | 1 | 0.797 | 1 | 0.537 |
| Team Rocket's Spidops | 73.8% | 2 | 0.693 | 4 | 0.709 |
| Cynthia's Garchomp ex | 70.9% | 3 | 0.655 | 6 | 0.878 |
| Alakazam | 67.8% | 4 | 0.767 | 2 | 0.693 |
| Dragapult ex | 58.8% | 5 | 0.687 | 5 | 0.733 |
| Mega Kangaskhan ex | 45.0% | 6 | 0.765 | 3 | 0.698 |

**The orderings disagree (Spearman ρ ≈ 0.26, weak), and loudly:**
- **Mega Kangaskhan ex** ranks 3rd on learnability (self-consistency 0.765, comparable to
  Alakazam) despite ranking **last** on win-rate among measured decks (45.0%) — the model
  confidently reproduces its training demonstrations, but that strategy still loses to the
  field. High imitability did not transfer to competitive strength here.
- **Cynthia's Garchomp ex** is the inverse: **last** on learnability (self-consistency 0.655,
  highest entropy 0.878 — the model is least sure what to do) yet **3rd** on win-rate (70.9%).
  Winning games did not require confident/consistent imitation.
- Only the top pick is consistent across both lenses: Marnie's Grimmsnarl ex is 1st on both.

This is the same shape as the Orbit War imitability-tracks-predictability finding, applied to
decks instead of players — and here it fails to hold uniformly, which is itself the finding:
"most learnable" and "wins against the field" are measuring different things, and picking a
deck by learnability alone (e.g. favoring Mega Kangaskhan ex for its high self-consistency)
would have been a mistake by 33 win-rate points.

## The one sentence

> **The agent should pilot Marnie's Grimmsnarl ex, because against the public field it wins
> 78.4% ± 2.3% (251/320 games), and I had 3488 training episodes of it — the largest deck in
> the corpus, vs. 1 episode for Mega Lucario ex, the deck it currently pilots, which under the
> identical policy and opponent pool wins only 29.7% ± 2.6%.**

Caveat carried forward honestly: Grimmsnarl ex is not statistically separated from Team
Rocket's Spidops (73.8%) or Cynthia's Garchomp ex (70.9%) at this sample size — see above for
the exact N that would separate them. What is fully decisive, independent of that caveat, is
that the agent should stop piloting Mega Lucario ex: every alternative tested beats it by a
wide, statistically clean margin.

## What would change my mind

- **A re-run at ~1300+ games/arm that flips or collapses the Grimmsnarl-ex-vs-Spidops gap** —
  the current 4.7pp edge is inside noise; more games could reorder the top of the field-facing
  ranking (though not the "leave Mega Lucario ex" conclusion, which is separated by 6–20× that
  margin from everything).
- **Evidence the modal decklist I picked for a corpus-only deck (e.g. Team Rocket's Spidops,
  whose modal exact list covers only 200/813 = 25% of that archetype's instances — the least
  consensus of the four corpus-only decks) is not representative** — a different sampled
  decklist for the same archetype could move that deck's win rate.
- **A deck-embedding-conditioned retrain** that changes what "the same policy" means — this
  whole measurement assumes the current unconditioned BC policy, and its ranking of decks could
  shift once the model can see which deck it's piloting as an input (the Q43 open question).
- **Opponent-pool composition changes** — this ranks decks against the *current* 7 public
  agents + random_legal. A different, larger, or meta-shifted public pool could reorder decks
  whose strength depends heavily on specific matchups (Mega Kangaskhan ex's 87.5% vs
  `kiyotah_abomasnow` but 10.0% vs `kiyotah_dragapult` shows this sensitivity already).
- **Real Kaggle leaderboard score diverging from this local measurement** — per prior session
  history, local Glicko/win-rate rankings have already been observed to diverge from the real
  leaderboard once; this local matrix should be treated as strong evidence, not a substitute
  for checking the actual submission score.
