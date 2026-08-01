# Phase 1 — Architecture & Data Decisions

Grounded in the Phase 0 measurements (`notes/phase0_discovery_report.md`).
Each decision states the trade-off and commits.

## 1.1 Architecture

**Attention / Pattern-B over a flat MLP: keep it, and this is not close.**
The observation is a set of variable-length, permutation-invariant zones
(hand 0–32 cards, bench 0–8, prizes) and — more importantly — the action
space itself is a variable-length list of legal options (2 to 42+ observed).
A flat MLP needs a fixed slot layout for outputs; it cannot represent "option
3 is ATTACH this turn, EVOLVE next turn" without a global action vocabulary,
which this game doesn't have (option semantics are defined relative to the
current select, not a fixed enum of moves). Cost, measured: 1.22M params /
1.57 ms per decision at the existing hidden=128/layers=4 config — trivial
against the 600s/2000s budget. There is no version of this problem where a
flat MLP is cheaper in any way that matters here.

**History: none (memoryless on `current`) for v1.** `current` already hides
the opponent's hand and both decks' order (verified, Phase 0 §0.1), so the
policy is fitting a belief-free approximation regardless of how much history
it's given — more ticks of the past don't recover hidden information the
opponent hasn't revealed. The existing encoder already carries the only
history that matters for most decisions as scalars on `current` itself
(`supporterPlayed`, `stadiumPlayed`, `energyAttached`, `retreated` — all
turn-scoped memory, already present). `logs` (turn-level event history) is
real and available but unused; cost of adding an attention-over-logs branch
is a second encoder stack, non-trivial extra params and dev time, for a
benefit I have no evidence for yet. Decision: ship without it, revisit only
if Rung 3 sanity checks (playing real games) surface a specific short-memory
failure that a logs-based feature would plausibly fix.

**HF `transformers.BertModel` vs hand-rolled: keep `transformers`.**
I flagged this as a possible switch in the Phase 0 report on cold-start
grounds; on reflection given the real budget numbers, that flag doesn't
survive its own evidence: cold-start import overhead is 1.4s
(`transformers` on top of `torch`) against a 600s per-match / 2000s total
budget — three orders of magnitude of headroom. Rewriting `BertModel`'s
`inputs_embeds`-only usage as a hand-rolled `nn.TransformerEncoder` would
save nothing that matters and would forfeit `save_pretrained`/
`from_pretrained`/`PretrainedConfig` versioning for free. Reversing my own
earlier caution: this was over-indexing on a number that turned out not to
matter once checked against the actual budget, not on the merits.

## 1.2 Parameter budget

Existing default (hidden=128, layers=4, heads=4): **1.22M params**, 33% of
which is embedding tables (`card_emb` dominates at `1269 × 128`). Measured
CPU fp32 latency: 1.57 ms/decision. Q6/Q19 evidence (Phase 0): ~718K labeled
rows, 33–48% distinct by a loose state signature → roughly 250–350K
effective distinct decision states. That comfortably supports something
larger than 1.22M without obvious overparameterization risk, and CPU latency
headroom is enormous (1.57ms against a budget where even 100ms/decision
would be fine).

**Decision: bump to hidden=192 / layers=6 / heads=6 → 3.32M params.**
Measured CPU fp32 latency at this size: **2.85 ms/decision** — still
trivial. This sits mid-range in the 2–15M budget (nowhere near the 50M
ceiling), justified by the row-count headroom above, not by "bigger is
free" — I am not going to 15M without evidence it helps, since duplication
in the data (Q6) argues for staying conservative.

## 1.3 Action masking

**Pattern B (score-each-option) — already implemented correctly, keep it.**
`il_model.py`'s `score_head` scores every option token independently via the
shared attention encoder, masks padding slots to `-inf`, and shares the card
embedding table between the state encoder and the option-reference encoding
(verified, Q13). Cross-entropy over the legal set with the recorded index as
label — matches the spec exactly.

**`maxCount > 1` (8.8% of decisions, Q9): autoregressive picking with
re-masking.** Combinatorial scoring is out — legal combinations range from a
handful to combinatorially many (`maxCount` up to 28 observed on train day),
so a joint softmax over combinations isn't tractable at any of these sizes.
Autoregressive is simple, reuses the existing single-choice head unchanged
(pick highest-scoring unmasked option → mask it out → repeat until
`maxCount` picks or the model would emit a duplicate), and lets multi-select
decisions share every parameter with single-select ones instead of needing
a second head. Implemented in `il_dataset.py`/`il_model.py` as k
sequential label rows per multi-select decision (masking prior picks),
not a new model.

**`minCount == 0` (7.8% of decisions; 18% of those are genuine declines,
Q10): decline as a first-class scored option, not a special-cased
zero-length response.** When `minCount == 0`, the encoder appends one
virtual "decline" slot to the option sequence (its own learned embedding,
not tied to any card), scored by the same head as every real option;
choosing it means the response is `[]`. This was the single concrete bug
found in the existing v1 pipeline — it silently dropped every decline
example from training (Phase 0 finding). Fixed as part of this build, not
deferred.

**Deck-selection (`select is None`): routed to the fixed 60-ID deck
constant, never the model.** Already correct in `agent_core.py`.

## 1.4 Deck policy

**Frozen deck: the existing `agents/il_agent/deck.csv` — confirmed by name
lookup to be exactly the Mega Lucario ex sample list** (Mega Lucario ex ×4,
Riolu ×4, Makuhita/Hariyama/Lunatone/Solrock support line, Boss's Orders,
Lillie's Determination, Poké Pad, Dusk Ball, Switch, Fighting Gong, Premium
Power Pro, Hero's Cape, Gravity Mountain, Carmine, 14× Basic Fighting
Energy). No change needed — this is what `agent_core.py` already submits at
the deck step.

**Does the model condition on which deck it's piloting? No — and this is
now a measured, not assumed, decision.** I checked how much of the training
corpus actually features the frozen deck's own defining cards: across 400
episodes / 800 player-perspectives, **Mega Lucario ex and Riolu appear
zero times**, and the deck's other archetype-defining cards
(Makuhita/Hariyama/Lunatone/Solrock/Dusk Ball) also appear **zero times**.
Only the deck's generic meta-staple support cards show up commonly (Poké
Pad 88%, Boss's Orders 77%, Lillie's Determination 72%, Basic Fighting
Energy 14% — because these are played across many decks on the ladder, not
because anyone is piloting Mega Lucario ex).

This settles **(a) filter to frozen-deck episodes vs (b) deck embedding**
decisively against filtering: **filtering to the frozen deck would keep
~0% of the corpus.** It is not a matter of "expensive," it is not
available. A deck-ID embedding is also not meaningfully implementable right
now — there is no ground-truth deck/archetype label in the replay, only
cards revealed as the match progresses, and building an archetype
classifier is out of scope for this pass.

**Decision: no explicit deck conditioning — keep the existing card-ID-level,
deck-agnostic encoding.** This is not a compromise; it's the better fit
given the data reality. A policy that memorized "when piloting deck X, do
Y" would have almost nothing to learn from for the deck we actually submit,
since the corpus is ~0% Mega Lucario ex games. A policy that learns
per-card-signature tactics (attach energy when useful, retreat when
behind, attack for lethal, use Boss's Orders to pull a low-HP target) from
whatever cards are visible on a given turn transfers zero-shot to a deck
it's never seen played, because the mechanics it's learning are about the
*cards and board state*, not about "deck X's playbook." The one real risk
this creates, quantified: the card-ID embedding rows for Mega Lucario ex,
Riolu, and the support-line Pokémon will be **untrained / near-random** at
inference (never seen in a single gradient step), so any Lucario-specific
learned association is absent — the policy will lean entirely on the
decoupled scalar features (hp_frac, energy_norm, tool_norm) and generic
option-type/attack embeddings for its own key cards, not on a learned
"this card is Lucario" representation. Flagging this as a known, accepted
limitation rather than a silent gap.

**Ranked alternate-deck shortlist**, evidence = measured prevalence of each
archetype's defining card(s) across the same 400-episode / 800-perspective
sample (a real meta signal, not a guess):

| Rank | Deck | Prevalence | Note |
|---|---|---:|---|
| 1 | Raging Bolt ex | 3.4% | most-played of the candidates by a clear margin |
| 2 | Mega Lopunny ex | 1.9% | second most common |
| 3 | Misty's/Mega Starmie ex | 0.6% | present but rare |
| 4 | N's Zoroark ex | 0.2% | rare; also the deck seen in the very first sample episode I inspected in Phase 0 (N's Darmanitan/N's Zorua) |
| 5 | Archaludon ex | 0.1% | rare |
| — | Dragapult ex, Iono's Bellibolt ex, Mega Abomasnow ex, Greninja ex, Mega Absol ex | 0.0% | not observed in this sample; a 0/800 count upper-bounds true prevalence at roughly ≤0.4% (95% CI), not proven absent |

If deck choice is revisited later, **Raging Bolt ex** is the best-evidenced
alternative — it would also inherit the most training signal of anything on
the shortlist, for the same reason the frozen deck currently inherits none.
