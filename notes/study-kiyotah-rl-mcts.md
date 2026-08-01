# Study: "Reinforcement Learning and MCTS Sample Code" (Kiyotah)

> Source: `notebooks/reference/kiyotah-rl-mcts/reinforcement-learning-and-mcts-sample-code.ipynb`
> Kiyotah is one of the competition hosts. This is the official self-play + neural-MCTS
> example for the Pokémon TCG AI Battle Challenge (Simulation track).
>
> Structure of the notebook: **2 cells only.** Cell 1 = imports + constants. Cell 2 =
> the entire model, search, training loop (~25 KB of code).
>
> Model: `MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1)`
> `SEARCH_COUNT=10`, candidate enumeration cap = **64**, reward blending `LAMBDA=0.9`.

> **Following the line references:** the original `.ipynb` is a *single* 689-line code
> cell with no visible line numbers in Jupyter, so any "read lines N–M" instruction is
> un-followable there. Use the line-numbered copy instead:
> `notebooks/reference/kiyotah-rl-mcts/source_with_linenumbers.py` — the same cell exported
> verbatim as a plain `.py`, so its real editor line numbers are 1:1 with the notebook. The
> walkthroughs `03_kiyotah_rl_mcts_walkthrough.py` ("lines 389-442") and
> `05_kiyotah_rl_mcts_full_walkthrough.py` ("lines 7-36", "lines 196-241", …) cite ranges
> against that exact cell; they map exactly to this file (verified — see the folder `README.md`).

---

## 1. What kind of agent is this?

Not PPO. It's a compact **AlphaZero-style self-play** system:
- A neural net (`MyModel`) learns **two things**: a scalar **value** of a state, and a
  **score per candidate action** (a policy prior).
- **MCTS** does the actual search each move, using the net's scores as priors and its value
  for backprop. The played action is the **most-visited** root child.
- Training targets come from **self-play rollouts** blended with terminal outcomes — no
  external expert, no human data.

This matches the skill's verified reading: *"a compact self-play + neural policy/value +
MCTS example, not PPO."*

---

## 2. The model (`MyModel`)

```python
self.encoder_bag   = EmbeddingBag(encoder_size=22000, d_model=128, mode="sum")
self.encoder       = TransformerEncoder(TransformerEncoderLayer(128, num_heads=2, 256, 0), num_layers=1)
self.encoder_fc    = Linear(128, 1)          # VALUE head  -> 1 scalar
self.decoder_bag   = EmbeddingBag(decoder_size, 128, mode="sum")
self.decoder       = ModuleList([DecoderLayer(128, 2, 256)])   # 1 layer, cross-attn only
self.decoder_fc    = Linear(128, 1)          # ACTION head -> 1 score per candidate
```

**Two output heads, both `Linear(d_model, 1)`:**
- `encoder_fc` → value: `tanh(encoder_out.mean(0))` → one scalar.
- `decoder_fc` → action: one score **per candidate**, then `tanh`. This is a **shared
  per-candidate scorer**, not K fixed action heads (see §Open Questions #3/#5).
- `num_heads=2` is **multi-head attention**, not an action head. Don't conflate (this was
  the earlier "how many action heads" question — answer: **1 action head**, shared).

**Encoder input** = sparse `EmbeddingBag` over board/player/hand/deck/stadium/turn words
(`num_words_encoder = 24`). **Decoder input** = sparse features per *candidate action*
(the decisions CABT offers), hashed by `OptionType` + referenced card id.

---

## 3. How an action is chosen (the core loop)

1. `mcts_agent(obs, deck, model)` opens a `SearchState` via `search_begin` — this is the
   **imperfect-information determinization**: the agent's own hidden deck/prizes are sampled
   from the known deck, but the **opponent's** hidden info is filled with placeholders
   (Snorlax `1072` deck/prize/active, Basic Energy `1` hand/prize — comment: "no deep
   meaning").
2. `create_node` enumerates **up to 64 candidate action combinations** from the variable
   legal `option` list, builds decoder features, and calls the model → `value, policy`.
   `policy[i]` is one score per candidate.
3. Scores become **priors**: `p = exp(policy[i] * 10)` → softmax across candidates.
4. **MCTS expands `SEARCH_COUNT=10` times**, descending by PUCT:
   `value + 0.4 * sqrt(parent_visits) * prior / (1 + child_visits)`.
5. Final action = **most-visited root child** (`max_child.select`). Not highest score, not
   highest value.

So the net is a **prior + value function**; search commits. Exactly AlphaZero.

---

## 4. Training loop

- Each outer cycle (5 total): save checkpoint, **evaluate vs `random_agent`** (only a
  legality smoke test — see skill note), then **self-play 100 games** collecting
  `(value, policy, sv_enc, sv_dec)` samples.
- Labels computed walking backward from terminal reward (±1):
  ```python
  label = (value + sample.value) * 0.5
  value = value * LAMBDA + sample.value * (1.0 - LAMBDA)   # LAMBDA = 0.9
  sample.policy[i] = clip(child_value - root_value, -1, 1)
  ```
- Loss = `HuberLoss(value)` + `HuberLoss(policy) * mask` (mask zeroes the 64-padding
  candidates). Optimizer = AdamW, lr 3e-4, batch 128.

> See **Open Questions #2, #4, #10, #11** for ambiguities in this section.

---

## 5. Open Questions about the notebook itself

These are unresolved points **in the code as written** — ambiguities, edge cases, and
unexplained constants you'd hit if you ran or extended it.

### #1 — Is `num_words_encoder = 24` actually emitted?
Encoder words: `2 players × 8 bench` (16) + `2 active` + `2 player-state` + `1 hand` +
`1 deck` + `1 stadium` + `1 turn/global` = 24. **But** the bench loop uses a `sv.pos` reset
trick (`if j != 7: sv.pos = pos`) so the 8th bench slot may not emit a real word. Does the
loop truly produce exactly 24 `word_start()` calls, or is one slot silently empty → a
shape mismatch / zero-padded word?

### #2 — Why both a `*0.5` blend AND the `LAMBDA=0.9` blend?
The label loop mixes the same two quantities two different ways in the same pass (the `0.5`
average and the `LAMBDA` recurrence). Is the `*0.5` a leftover / redundant with `LAMBDA`?
What happens if you drop one? The notebook gives no justification.

### #3 — The decoder is **cross-attention only** — intended?
`DecoderLayer` calls `attention(x, encoder_out, encoder_out)` with **no self-attention** and
no mask. Each candidate attends only to the encoder, never to other candidates. Deliberate
(independent score queries) or a simplified stand-in for a full decoder? Unexplained.

### #4 — Does the 64-candidate cap truncate the real legal space?
`create_node` enumerates with a fixed `for _ in range(64)` and `break`s when combos run out.
For multi-pick decisions with many options the true combination count can exceed 64 → **some
legal candidates are never scored.** Is 64 always enough, or does the agent silently ignore
legal actions on large option sets? Not measured anywhere.

### #5 — What is `decoder_main_feature = 8` counting?
Comment: "Feature count of SelectContext.Main." Used in
`decoder_size = decoder_card_offset + (1 + decoder_main_feature + RECOVER_SPECIAL_CONDITION) * card_count`.
`decoder_main` only writes slots 0–7. Is the `+1` a sentinel? Is `RECOVER_SPECIAL_CONDITION`
a count or a flag? Exact meanings and whether the vocabulary math is right are undefined.

### #6 — `SparseVector.add_pos` vs `add_single` slot layout
`add` writes at `pos+index`; `add_single` writes at `pos` then advances; `add_pos` advances
without writing. The interleaving defines which card-bucket a feature lands in (the
`card_count` spacing), implicit in helpers `add_card`/`add_cards`/`add_pokemon`. A single
off-by-one shifts every downstream slot. The layout is undocumented — is it verified correct?

### #7 — Value head uses `mean` over encoder words
`v = tanh(encoder_out.mean(0))` averages the sequence to one scalar — giving the Active
Pokémon the same weight as an arbitrary bench slot. Is a mean the right aggregation, or
should there be learned pooling / a CLS token? Unexplained.

### #8 — `max_child` can be unbound
If `root.children` are all `node == None` (no expansion happened), the loop never sets
`max_child` → `max_child.select` raises `UnboundLocalError`. When can the root have only
unexpanded children with `SEARCH_COUNT=10`? Edge case the sample doesn't guard.

### #9 — `sample_deck` looks like **62 cards, not 60**
The literal `sample_deck = [721,721,722,...]` ends with many `3`s and appears to have 62
entries. `battle_start` requires exactly 60 (the skill notes the official visualizer literal
also has 62 and raises "must contain 60 cards"). Does the sample's `sample_deck` actually
pass `battle_start`, or would the training loop crash on the first call? Unverified.

### #10 — Magic PUCT `0.4` and softmax temperature `10`
```python
p = math.exp(policy[i] * 10.0)
c = 0.4 * math.sqrt(current.visit)
```
Both are magic numbers with no comment. Tuned or arbitrary? Sensitivity for this action
space? Not addressed.

### #11 — Decoder loss divides by `BATCH_SIZE`, not unmasked count
`loss_dec.sum() / BATCH_SIZE (128)` rather than the count of real (unmasked) candidates.
Batches with fewer real candidates get under-weighted. Intended (stable gradient scale) or a
subtle bias? Unexplained.

### #12 — Mirror self-play + fake opponents
Both self-play players use the **same** `model` and **same** `sample_deck`; the opponent is
determinized to Snorlax/Energy regardless of who's playing. Does training on identical-mirror
self-play + placeholder opponents transfer to **real asymmetric** opponents? The notebook only
evaluates vs `random_agent`, so this is untested. (Skill context: RL+MCTS basic v5 × Mega
Lucario scored **580.6 μ** vs SearchScorer **660.5 μ** — a real-baseline gap.)

### #13 — `cuda` branch is vestigial
`device = cuda if available else cpu`, but no AMP / `.half()` / `torch.compile`. The `cuda`
path just runs the same float32 forward. Irrelevant at submission (eval sandbox is CPU-only),
but may mislead a reader into thinking GPU matters for this net.

---

## 6. Smallest set to resolve before trusting the notebook

1. **#9** — does `sample_deck` even run? (blocks execution)
2. **#1** — is the encoder shape actually 24? (blocks forward pass)
3. **#4** — does 64 ever drop legal candidates? (selection correctness)
4. **#8** — can `max_child` be unbound? (runtime crash edge case)

---

## 7. Verified competitive context (from skill)
- penguin069's neural `model.pth` agent reached **1117.2 μ** — a learned net *can* be
  competitive on the ladder.
- Cautionary: RL+MCTS basic v5 (same deck) = **580.6 μ** < SearchScorer **660.5 μ**. A learned
  policy must be measured against strong heuristic/search baselines, not `random_agent`.
- The 64-candidate padding mask is **training-target padding**, not PPO action masking — don't
  confuse the two.
