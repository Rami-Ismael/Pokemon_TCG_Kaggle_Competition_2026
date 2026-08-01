import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import json
    import math
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return Path, itertools, json, math, mo, pd


@app.cell
def _(Path, json):
    # Load the creator's ORIGINAL notebook from the local reference mirror.
    # Source: https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code
    repo_root = Path(__file__).resolve().parents[1]
    source_path = (
        repo_root
        / "notebooks/reference/kiyotah-rl-mcts"
        / "reinforcement-learning-and-mcts-sample-code.ipynb"
    )
    _nb = json.loads(source_path.read_text())
    # Cell 'source' may be a str OR a list[str]; "".join handles both safely.
    source_code = "\n".join("".join(_c.get("source", [])) for _c in _nb["cells"])
    src_lines = source_code.splitlines()
    total_lines = len(src_lines)
    return source_path, src_lines, total_lines


@app.cell
def _(src_lines):
    # Shared helper: render an inclusive source line range as a numbered block.
    # Defined ONCE and threaded to later cells as a parameter (one-name-one-cell).
    def show(start, end):
        return "\n".join(
            f"{_n:>4}: {src_lines[_n - 1]}" for _n in range(start, end + 1)
        )

    return (show,)


@app.cell
def _(mo, source_path, total_lines):
    mo.md(
        f"""
    # Reinforcement Learning + MCTS sample — full step-by-step reading

    A complete, top-to-bottom walkthrough of **Kiyotah's official
    "Reinforcement Learning and MCTS sample code"** for the Pokémon TCG AI
    Battle Challenge (written by one of the competition creators).

    - **Original:** <https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code>
    - **Local mirror read here:** `{source_path.name}` — **{total_lines} lines** in one code cell.

    This is a **reading notebook, not a training run.** It does not import `torch`
    or `cg-lib` (both are unavailable off-Kaggle). Instead it shows the creator's
    **exact source** with line numbers, explains it in plain English, and runs
    small **pure-Python demos** that reproduce the parts that do not need the
    game engine (combination enumeration, sparse features, the PUCT rule,
    self-play credit assignment, and the masked training loss).

    ## How to read this notebook
    Every section follows the same shape:

    1. **Action** — what part of the source we read next.
    2. **Thought process** — how to think about it and why it matters.
    3. **Source** — the creator's verbatim code, with real line numbers.
    4. **Demo / table** — a runnable check, where it can run without the engine.

    ## Provenance rule (important)
    Each claim is one of three kinds, and the notebook keeps them separate:

    - **✅ verified from code** — you can point at the exact source lines.
    - **🔎 inference** — a reasonable reading of the structure, labelled as such.
    - **❓ not documented** — the creator gave no reason; left as an open question,
      never back-filled with an invented explanation.

    ## Big picture — the whole pipeline in one breath
    The sample is a **self-play + neural policy/value + MCTS** loop (it is **not**
    PPO). One `MyModel` (a small Transformer) both *values* a state and *scores*
    the legal candidate moves. MCTS uses those scores to search, self-play games
    generate training targets from the search results, and the model is trained to
    predict them. Then it repeats.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Map of the source (so you never lose your place)

    | Lines | What lives there | Section below |
    |------:|------------------|---------------|
    | 1–6   | Author's markdown intro (Search API + Battle APIs) | 0 |
    | 7–36  | Imports + `cg-lib` API surface | 1 |
    | 38–54 | Card/attack tables, vocab sizes, `SEARCH_COUNT` | 2 |
    | 57–120 | `DecoderLayer` + `MyModel` (the Transformer) | 3 |
    | 122–152 | `SparseVector` (EmbeddingBag input) | 4 |
    | 154–241 | Encoder features (state → sparse vector) | 5 |
    | 243–331 | Decoder features (each candidate move → sparse vector) | 6 |
    | 333–344 | `eval_nn` (run the model once) | 7 |
    | 346–387 | `LearnSample`, MCTS `Child`, `Node` + backprop | 8 |
    | 389–442 | `create_node` — candidate enumeration + priors | 9 |
    | 444–515 | `mcts_agent` — determinize, search, make labels | 10 |
    | 518–563 | `LearnInput`, `random_agent`, setup, losses | 11 |
    | 566–695 | Main loop: eval → self-play → train | 12 |
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 0 — What the author says this sample is for

    **Action:** read the creator's own intro (lines 1–6).

    ```text
{show(1, 6)}
    ```

    **Thought process:** the author frames the sample around two engine facilities,
    not around deep learning:

    - the **Search API** (`search_begin/step/end`) lets you ask *"what actually
      happens if I take this action right now?"* — invaluable because a Pokémon
      TCG effect can be hard to evaluate statically (damage, whether a drawn Item
      is playable, etc.);
    - the **Battle Start/Finish API** runs a full local game.

    So even before the neural network, the message is: the engine can *simulate*
    consequences for you. MCTS is built directly on that Search API. A rule-based
    agent could use the very same facility.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 1 — Imports and the `cg-lib` API surface

    **Action:** read lines 7–36.

    ```python
{show(7, 36)}
    ```

    **Thought process — read the imports as a table of contents.** The names pulled
    from `cg.api` tell you the whole engine contract this agent relies on:

    | Import | Role in this sample |
    |--------|---------------------|
    | `to_observation_class(obs_dict)` | turn the raw dict the engine passes into a typed `Observation` |
    | `Observation`, `PlayerState`, `Pokemon`, `Card` | typed views of board state |
    | `OptionType`, `SelectContext`, `AreaType` | enums describing *what a legal option is* and *where a card lives* |
    | `SearchState` | a clonable, advanceable search position |
    | `search_begin` / `search_step` / `search_end` | clone the game, apply a move in the clone, tear down |
    | `all_card_data` / `all_attack` | static card & attack metadata (for vocab sizes) |
    | `battle_start` / `battle_select` / `battle_finish` | run a real local episode |

    **✅ verified from code (line 18):**
    `sys.path.append(glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0])`.
    The sample hard-codes the **Kaggle** input path to `cg-lib`. That is why this
    walkthrough cannot execute the engine locally — the module is only present in
    the Kaggle sandbox (and, for our repo, in the `kiyotah/cg-lib` dataset). We read
    the code instead of importing it.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 2 — Card tables, vocabulary sizes, and the search budget

    **Action:** read lines 38–54.

    ```python
{show(38, 54)}
    ```

    **Thought process — this block sizes the model's vocabulary.** The network
    consumes *sparse* features (an `EmbeddingBag`), so it needs to know how many
    distinct feature slots exist for the encoder (state) and the decoder (a move).

    - **`card_count` (line 42)** = max cardId + 1 → one embedding slot per card id.
    - **`attack_count` (line 44)** = max attackId + 1 → one slot per attack id.
    - **`encoder_size = 22000` (line 47)** — a fixed vocabulary cap. The comment
      calls it "exceeding the vocabulary size": it is deliberately large enough to
      hold every encoder feature index the state builder can emit.
    - **Decoder layout (lines 49–52)** carves the decoder vocabulary into regions:
      first 8 slots for `SelectContext.Main`-style features, then one region of
      `attack_count` attack slots starting at `decoder_attack_offset = 14`, then
      several `card_count`-sized card-feature regions
      (`decoder_card_offset` onward). We meet those regions again in Section 6.
    - **`SEARCH_COUNT = 10` (line 54)** — MCTS runs **10 expansions** per decision.

    **❓ not documented:** the exact numeric choices (`24` encoder words,
    `22000` cap, `SEARCH_COUNT = 10`) come with no benchmark or rationale in the
    source. Record them as open design questions, not as "tuned" values.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Why a *sparse* representation at all?

    A Pokémon TCG state is enormous and mostly empty: hundreds of possible card
    ids, but only a few are on the board at once. Instead of a giant dense vector
    that is almost all zeros, the author lists only the **indices that are "on"**
    (plus a weight). `torch.nn.EmbeddingBag(vocab, d_model, mode="sum")` then sums
    the embeddings of exactly those active indices. That is why the next thing we
    read (Section 4) is a tiny helper class whose only job is to accumulate
    `(index, value)` pairs.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 3 — The model: one Transformer, two heads

    **Action:** read the `DecoderLayer` (lines 57–73) and the `MyModel` shell
    (lines 75–96).

    ```python
{show(57, 96)}
    ```

    **Thought process — this is an encoder–decoder Transformer with two outputs.**

    - **Encoder** (`encoder_bag` + `TransformerEncoder`, lines 88–91): reads the
      **state** as a sequence of `num_words_encoder = 24` "words" (Section 5 builds
      exactly 24 words). `encoder_fc` maps each to a scalar; their mean, squashed by
      `tanh`, is the **value** of the state — a number in [-1, 1].
    - **Decoder** (`decoder_bag` + stacked `DecoderLayer`, lines 92–96): reads the
      **candidate moves** (one "word" per candidate) and cross-attends to the
      encoded state. `decoder_fc` produces **one scalar per candidate** — the
      **policy score** for that move.

    So a single forward pass answers two questions at once: *how good is this
    state?* and *how good is each move from it?* This shared trunk is the classic
    AlphaZero-style policy/value split, done here with a Transformer instead of a
    conv net.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ### `forward` — trace the two tensors (lines 98–120)

    ```python
{show(98, 120)}
    ```

    **Thought process — follow the shapes:**

    - Lines 106–111 (**value head**): sum sparse embeddings → reshape to
      `(24 words, batch, d_model)` → run the encoder → `encoder_fc` to a scalar per
      word → `tanh(mean over words)`. Result **`v`**: one value per batch item.
    - Lines 113–119 (**policy head**): sum sparse embeddings for the candidate
      words → reshape → each `DecoderLayer` cross-attends to `encoder_out` (line
      116, the `attention(x, encoder_out, encoder_out)` inside the layer) →
      `decoder_fc` to one scalar per candidate → `tanh`. Result **`p`**: a score
      per candidate.
    - Line 120 returns `(v, p)` — value and policy together.

    **✅ verified from code:** the decoder attends over `encoder_out` (the full
    encoded state), so every candidate is scored *in the context of the current
    board*, not in isolation. **🔎 inference:** using `tanh` on both heads keeps
    value and per-candidate scores on the same [-1, 1] scale, which is convenient
    because the training targets (Section 10) are also clamped to [-1, 1].
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 4 — `SparseVector`: the EmbeddingBag input builder

    **Action:** read lines 122–152.

    ```python
{show(122, 152)}
    ```

    **Thought process — this is just a growable "list of active features".** It
    holds three parallel arrays that map exactly onto `EmbeddingBag`'s arguments:

    - **`index`** — the active feature ids (globally offset, see below);
    - **`value`** — the weight for each active id (skips zeros — line 137/146);
    - **`offset`** — where each "word" starts inside `index` (one bag per word).

    The clever part is **`pos`**, a running base offset:

    - `add(index, value)` records feature `pos + index` (line 138) — a *local*
      index placed into a *global* slot;
    - `add_pos(n)` (line 142) advances the base by `n`, reserving a block of `n`
      slots for the field just written;
    - `add_single(value)` (lines 144–149) writes one boolean/float feature and
      advances `pos` by 1;
    - `word_start()` (line 152) marks the boundary of a new "word" (a new bag).

    This is how the author packs many heterogeneous fields (HP, a card id, a list
    of energy cards, …) into one flat, collision-free index space. We can run the
    exact same logic without torch — the demo below re-implements it and shows the
    arrays it produces.
    """
    )
    return


@app.cell
def _():
    # Faithful pure-Python re-implementation of the source SparseVector
    # (source lines 122-152) so we can SEE the arrays it builds. No torch needed.
    class DemoSparseVector:
        def __init__(self):
            self.index = []
            self.value = []
            self.offset = []
            self.pos = 0

        def add(self, index, value):
            value = float(value)
            if value != 0.0:
                self.index.append(self.pos + index)
                self.value.append(value)

        def add_pos(self, pos):
            self.pos += pos

        def add_single(self, value):
            value = float(value)
            if value != 0.0:
                self.index.append(self.pos)
                self.value.append(value)
            self.pos += 1

        def word_start(self):
            self.offset.append(len(self.index))

    return (DemoSparseVector,)


@app.cell
def _(DemoSparseVector, pd):
    # Encode a toy "Pokemon word": alive flag + hp + a card id, in a 20-slot block.
    _sv = DemoSparseVector()
    _sv.word_start()          # start word 0
    _sv.add_single(0)         # alive flag = 0 -> skipped (value 0), pos -> 1
    _sv.add_single(120 / 400) # hp fraction  -> index 1, value 0.30, pos -> 2
    _sv.add(7, 1)             # card id 7    -> index 2 + 7 = 9, value 1.0
    _sv.add_pos(20)           # reserve a 20-slot card region, pos -> 22
    _sv.word_start()          # start word 1
    _sv.add(3, 0.5)           # another feature -> index 22 + 3 = 25

    demo_sparse_df = pd.DataFrame(
        {"index": _sv.index, "value": _sv.value}
    )
    demo_sparse_offsets = _sv.offset
    return demo_sparse_df, demo_sparse_offsets


@app.cell
def _(demo_sparse_df, demo_sparse_offsets, mo):
    mo.md(
        f"""
    ### Demo output — the sparse arrays

    Word boundaries (`offset`) landed at **{demo_sparse_offsets}** — i.e. word 0
    starts at position 0 in `index`, word 1 starts at position 2. Notice the
    `alive=0` feature was **dropped** (zeros are never stored), and the card id
    `7` was placed at global slot `9` because it sat inside a block that started
    at `pos = 2`.

    {mo.ui.table(demo_sparse_df, selection=None)}
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 5 — Encoder features: turning the board into 24 "words"

    **Action:** read the per-object feature helpers, lines 154–194.

    ```python
{show(154, 194)}
    ```

    **Thought process — each helper encodes one game object into the sparse vector:**

    - `add_card` (154–158): mark a single card id present, then reserve a full
      `card_count` block so the next field starts clean.
    - `add_cards` (160–165): mark a *set* of cards, each with the same weight
      (e.g. energy attached, discard pile).
    - `add_pokemon` (167–177): if the slot is empty, write a single "empty" flag
      and skip its whole region; otherwise write `alive=0`, `hp/400`, the card
      itself, its tools, and its energy (energy weighted `0.5`).
    - `add_player` (179–194): scalar summary of a player — deck/discard/hand/bench
      counts (each normalised), a prize-count feature, the five status conditions,
      then the discard contents at weight `0.25`.

    Notice the **normalisation constants** (`/400` HP, `/60` deck, `/8` hand,
    `/5` bench, `/10` turn). They scale raw counts into roughly [0, 1] so the
    embedding weights stay comparable. **❓ not documented:** the exact divisors
    are reasonable magnitudes (a deck is 60 cards, max hand ~8) but the source
    gives no derivation — treat them as sensible defaults, not tuned values.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ### Assembling the 24 words — `get_encoder_input` (lines 196–241)

    ```python
{show(196, 241)}
    ```

    **Thought process — count the `word_start()` calls; that is the sequence length.**
    The encoder emits exactly **`num_words_encoder = 24`** words, and here is why:

    | Words | Source | Content |
    |------:|--------|---------|
    | 16 | lines 201–211 | 8 bench slots × 2 players (yours first via `i ^ yourIndex`) |
    | 2  | lines 213–219 | active Pokémon × 2 players |
    | 2  | lines 221–224 | player summaries × 2 |
    | 1  | lines 226–227 | **your** hand (weight 0.25 — you know your own hand) |
    | 1  | lines 229–232 | **your** known deck list (weight 0.25) |
    | 1  | lines 234–235 | the stadium |
    | 1  | lines 237–240 | turn/order scalars |

    That is **16 + 2 + 2 + 1 + 1 + 1 + 1 = 24** — exactly the encoder length the
    model reshapes to in `forward` (line 107). **✅ verified from code.**

    **🔎 inference — perspective handling:** `i ^ your_index` (lines 202, 214, 222)
    always lists *your* objects before the opponent's, so the model sees a
    consistent "me vs. them" ordering regardless of which seat you are in. This is
    the encoder half of the imperfect-information story: your own hand and deck are
    encoded honestly (you know them), while the opponent's hidden cards are only
    represented by *counts* in `add_player`, not by identities.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 6 — Decoder features: describing each candidate move

    **Action:** read the decoder helpers and the `get_card` lookup, lines 243–278.

    ```python
{show(243, 278)}
    ```

    **Thought process — a "move" must be turned into features the model can score.**
    `get_card` (243–263) resolves an option's `(area, index)` into the actual card
    object, wherever it lives (deck, hand, discard, active, bench, prize, stadium,
    looking). The three `decoder_*` helpers then write card-based features into the
    decoder regions that were sized back in Section 2:

    - `decoder_main` (266–268) → the 8 `Main` feature slots (play/attach/evolve/…);
    - `decoder_card_id` / `decoder_card` (271–277) → a card feature keyed by the
      current `SelectContext` (so "choosing a card to discard" and "choosing a
      card to search" land in different regions).
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ### One word per candidate — `get_decoder_input` (lines 279–331)

    ```python
{show(279, 331)}
    ```

    **Thought process — this is the heart of the action representation.** For each
    candidate `action` (a list of option indices, e.g. `[2, 7]`):

    - `word_start()` (line 285) begins that candidate's own "word";
    - an **empty** action (pass/no-op) writes a single sentinel feature (287–289);
    - otherwise, for every option index `i` in the action, it looks at
      `obs.select.option[i].type` and writes **type-specific** features
      (lines 293–329): `END/YES/NO`, special conditions, numbers, attacks, and the
      `PLAY/ATTACH/EVOLVE/ABILITY/DISCARD/RETREAT/CARD/…` main options with their
      referenced cards.

    So the decoder does **not** emit a global action id. It builds a feature
    description of *this specific move in this specific state*, and the model
    scores that description. **✅ verified from code** — this is exactly why the
    "64 candidates" limit is a *scoring width*, not a fixed action space (we prove
    that in Section 9).

    **🔎 inference:** the big `match` over `OptionType` is effectively a hand-built
    "action encoder". Any option type the author forgot would silently produce an
    almost-empty feature word — a place future work could audit against the full
    CABT `OptionType` enum.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 7 — `eval_nn`: one forward pass, unpacked

    **Action:** read lines 333–344.

    ```python
{show(333, 344)}
    ```

    **Thought process:** this is a thin wrapper. It converts the two `SparseVector`s
    (state + candidates) into tensors, runs `model(...)`, and returns a plain
    `(value, policy_list)` — the **state value** (one float) and a **list of
    per-candidate scores**. Everything MCTS needs from the network flows through
    this one function. **✅ verified from code:** `value.tolist()[0][0]` is a single
    scalar; `policy.tolist()[0]` is the row of candidate scores.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 8 — The MCTS data structures

    **Action:** read `LearnSample`, `Child`, and `Node` (lines 346–387).

    ```python
{show(346, 387)}
    ```

    **Thought process — three tiny records hold the whole search:**

    - **`LearnSample` (346–352)** — one training example: the value target, the
      per-candidate policy targets, and the two sparse inputs that produced them.
      (The targets are overwritten later, in Section 10, once search finishes.)
    - **`Child` (354–363)** — an edge from a node: the `select` (option indices for
      this move) and its **prior** `prob` from the policy head. `node` is `None`
      until the child is expanded (lazy expansion).
    - **`Node` (365–387)** — a search position: `value` (its own estimate),
      `total`/`visit` (running sum and count for the average), `parent`,
      `children`, and the `SearchState` clone it wraps.

    The one behaviour worth reading closely is **`backprop` (383–387)**: it adds
    `value` to `total`, increments `visit`, and recurses to the parent. That is the
    standard MCTS backup — every node on the path from a newly evaluated leaf up to
    the root gets the same value credited to its running average.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 9 — `create_node`: enumerate candidates, evaluate, set priors

    **Action:** read lines 389–442.

    ```python
{show(389, 442)}
    ```

    **Thought process — read it in three beats:**

    1. **Terminal check (399–408):** if the engine says the game is over
       (`state.result >= 0`), set the node value directly — `0` for a draw,
       `+1` if *you* won, `-1` otherwise — backprop it, and make **no** training
       sample (there is nothing to choose).
    2. **Candidate enumeration (410–422):** starting from `indices = [0..maxCount-1]`,
       advance lexicographically to the next strictly-increasing combination, and
       keep **at most 64** of them (`for _ in range(64)`). This is combinations
       *without replacement* — the same generator your `03` walkthrough studied.
    3. **Evaluate + priors (424–439):** build the encoder + decoder sparse inputs,
       call `eval_nn` once, flip the value if it is the opponent's turn
       (428–429), backprop it, then turn each policy score into a **prior**:
       `p = exp(policy[i] * 10.0)`, normalised to sum to 1 (433–439). One `Child`
       is created per retained candidate.

    **✅ verified from code:** the `* 10.0` sharpening (line 435) makes the softmax
    over candidate scores much peakier — small score gaps become large prior gaps.
    **❓ not documented:** why `10.0` specifically. Record as an open question.
    """
    )
    return


@app.cell
def _(itertools, math):
    # Reproduce the EXACT enumeration of source lines 410-422, then cross-check it
    # against Python's itertools.combinations. Pure Python, no engine needed.
    def enumerate_candidates(n_options, max_count, cap=64):
        indices = list(range(max_count))
        actions = []
        for _ in range(cap):
            actions.append(indices.copy())
            for i in range(len(indices)):
                index = len(indices) - i - 1
                if indices[index] < n_options - i - 1:
                    indices[index] += 1
                    for j in range(index + 1, len(indices)):
                        indices[j] = indices[j - 1] + 1
                    break
            else:
                break
        return actions

    def enum_report(n_options, max_count, cap=64):
        got = enumerate_candidates(n_options, max_count, cap)
        want = [list(c) for c in itertools.combinations(range(n_options), max_count)]
        total = math.comb(n_options, max_count)
        kept = len(got)
        # When under the cap, the author's order must match itertools exactly.
        matches = got == want[:kept]
        return total, kept, matches, got

    return enum_report, enumerate_candidates


@app.cell
def _(enum_report, pd):
    _rows = []
    for _n, _k in [(5, 1), (8, 2), (12, 2), (20, 3), (30, 3)]:
        _total, _kept, _match, _ = enum_report(_n, _k)
        _rows.append(
            {
                "option entries (N)": _n,
                "required (k)": _k,
                "all C(N,k)": _total,
                "candidates kept": _kept,
                "truncated?": "yes" if _kept < _total else "no",
                "order matches itertools": "✅" if _match else "❌",
            }
        )
    enum_df = pd.DataFrame(_rows)
    return (enum_df,)


@app.cell
def _(enum_df, enumerate_candidates, mo):
    _first = enumerate_candidates(12, 2)[:5]
    mo.md(
        f"""
    ### Demo — the enumeration, verified against `itertools`

    The table below runs the author's exact loop and confirms that, while it stays
    under the cap, it produces **the same combinations in the same order** as
    `itertools.combinations`. Where `C(N,k) > 64`, it keeps only the first 64 in
    lexicographic order — a hard **truncation**, not a sample.

    For `N=12, k=2` the first five candidates are `{_first}` — i.e. option pairs
    `(0,1), (0,2), …`. There are `C(12,2)=66` legal pairs, so **2 are dropped**.

    {mo.ui.table(enum_df, selection=None)}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Why "64 candidates" is **not** a fixed action head

    The model scores a *different* list of up to 64 candidates at every decision.
    Slot 0 today might mean "attach Energy A to Pokémon B"; next state, slot 0
    might mean "choose a prize card". The candidate's **sparse feature word**
    (Section 6) gives the slot its meaning — not a permanent action id. `64` is a
    shared **budget** reused in three places: enumeration width (line 412), tree
    branching (one `Child` per candidate, 436), and batch padding width
    (Section 12). This is the single most misread part of the sample.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 10 — `mcts_agent`: determinize, search, and make labels

    This is the function the training loop calls at every decision. Read it in
    three parts.

    ### 10a — Determinize the hidden information (lines 444–458)

    ```python
{show(444, 458)}
    ```

    **Thought process — MCTS needs a concrete world to simulate, but the real game
    is hidden.** `search_begin` requires a *fully specified* position, so the author
    fills in the unknowns:

    - **your** remaining deck and prizes: `random.sample` from your known deck list
      (451–453) — a legitimate guess, since you know your own decklist;
    - **opponent** deck / prize / hand / active: filled with **placeholders** —
      Snorlax id `1072` and Basic Energy id `1` (454–457), with the code's own
      comment "There is no deep meaning".

    **✅ verified from code / ⚠️ limitation:** this determinization is *incomplete*.
    Your hidden cards are sampled plausibly, but the opponent's hidden cards are
    dummies. So the search does not do real opponent modelling — it assumes a
    fixed nonsense opponent hand/deck. A stronger agent would sample plausible
    opponent decks consistent with what has been seen. Carry this forward as the
    #1 improvement target.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ### 10b — The search loop (lines 459–488)

    ```python
{show(459, 488)}
    ```

    **Thought process — this is PUCT-style selection, run `SEARCH_COUNT = 10` times.**
    Each iteration walks from the root choosing the child with the best score:

    ```text
    score(child) = Q(child)  +  c · prior(child) / (1 + visits(child))
    where  c = 0.4 · sqrt(parent.visit)      # line 465
    ```

    - `Q` is the child's average value (`total/visit`), or the parent's average if
      the child is unexpanded (468–472);
    - the value is flipped when it is the opponent's turn (473–474) so the walk is
      always "good for whoever is about to move";
    - the exploration term `c · prior / (1 + visits)` (475) pulls toward
      high-prior, rarely-visited moves — the standard PUCT trade-off.

    When the walk reaches an **unexpanded** child (480), it applies that move in the
    cloned engine with `search_step` (481) and creates the new node (482) — one
    expansion per iteration. If it reaches a **terminal** node, it just backprops
    the known result (486–488).
    """
    )
    return


@app.cell
def _(math, pd):
    # Reproduce the exact PUCT score of source lines 464-475 for one root with
    # three children, to show how priors + visits steer selection. No engine.
    def puct_scores(parent_visit, children):
        c = 0.4 * math.sqrt(parent_visit)
        rows = []
        for name, q, prior, visit in children:
            explore = c * prior / (1 + visit)
            rows.append(
                {
                    "child": name,
                    "Q (avg value)": round(q, 3),
                    "prior": prior,
                    "visits": visit,
                    "explore term": round(explore, 3),
                    "PUCT score": round(q + explore, 3),
                }
            )
        return c, rows

    _c, _rows = puct_scores(
        parent_visit=9,
        children=[
            ("A high-Q, visited", 0.60, 0.20, 6),
            ("B low-Q, unvisited", -0.10, 0.50, 0),
            ("C mid-Q, few visits", 0.30, 0.30, 1),
        ],
    )
    puct_df = pd.DataFrame(_rows)
    puct_c = round(_c, 3)
    return puct_c, puct_df


@app.cell
def _(mo, puct_c, puct_df):
    mo.md(
        f"""
    #### Demo — one PUCT selection step

    With `parent.visit = 9`, the exploration coefficient is
    `c = 0.4·sqrt(9) = {puct_c}`. The winner is the row with the highest **PUCT
    score** — notice how the high-prior, never-visited child B gets a large
    exploration bonus even though its current Q is negative. That is exactly the
    "optimism about untried moves" MCTS relies on early in a search.

    {mo.ui.table(puct_df, selection=None)}
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ### 10c — Choose the move and build training targets (lines 490–515)

    ```python
{show(490, 515)}
    ```

    **Thought process — search is over; now turn the tree into a decision + labels.**

    - **The move played** is the **most-visited** root child (491–498) — the robust
      AlphaZero choice, not the highest-value one.
    - **The value target** becomes the root's average `total/visit` (504) — MCTS's
      refined estimate of the current state, better than the raw network value.
    - **The policy targets** (505–512): for each root child, the label is
      `child_Q − root_value` (an *advantage*: how much better than average this
      move looked), clamped to [-1, 1]. Unexpanded children get a small negative
      label (`min_value − v − 0.03`, line 509) so untried moves are gently
      discouraged relative to the ones search actually explored.
    - `search_end()` (514) tears down the clone; the function returns
      `(chosen_move, training_sample)`.

    **✅ verified from code:** the policy target here is an **advantage**, not a
    visit-count distribution — a notable difference from vanilla AlphaZero (which
    trains the policy toward normalised visit counts). **🔎 inference:** pairing an
    advantage-style policy target with the `tanh` decoder (Section 3) keeps targets
    and outputs on the same [-1, 1] scale.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 11 — Batch builder, random opponent, and setup

    **Action:** read lines 518–563.

    ```python
{show(518, 563)}
    ```

    **Thought process — plumbing before the main loop:**

    - **`LearnInput` (519–534)** concatenates many samples' sparse arrays into one
      big batch, shifting each sample's `offset` by the running index length (534)
      so all the bags stay correctly delimited after concatenation.
    - **`random_agent` (537–539)** is the evaluation opponent: pick `maxCount`
      option indices uniformly at random. **⚠️ limitation:** beating a random agent
      is only a smoke test for legality and "is training doing anything" — it is
      **not** evidence of strength against real ladder opponents or strong
      heuristics. Keep that caveat attached to any win-rate number this prints.
    - **Setup (555–564):** the training deck (`sample_deck`, a real 60-card list),
      device selection, `MyModel(128, 2, 256, 1, 1)` (d_model 128, 2 heads, ff 256,
      1 encoder + 1 decoder layer), `AdamW(lr=3e-4)`, and two Huber losses — one
      for value, one **per-element** for policy (`reduction="none"`, so it can be
      masked). `os.makedirs("out")` is where checkpoints will go.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ## Section 12 — The main loop: evaluate → self-play → train

    **Action:** read the outer loop and the evaluation phase, lines 566–609.

    ```python
{show(566, 609)}
    ```

    **Thought process — the outer loop runs 5 cycles (567); each cycle:**

    1. **Checkpoint first** (568): save `out/model{{counter}}.pth` before doing
       anything, so every cycle's starting weights are recoverable.
    2. **Evaluate** (571–609): under `inference_mode`, play 50 games of
       `mcts_agent` vs `random_agent`, alternating seats (`your_index = i % 2`),
       and print a win rate. Lines 578–588 decode deck-legality errors — a handy
       reference for what makes `battle_start` reject a deck.
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ### 12a — Self-play data collection (lines 611–637)

    ```python
{show(611, 637)}
    ```

    **Thought process — this is where training targets are born.** Play 100
    self-play games; at every decision `mcts_agent` returns both the move *and* a
    `LearnSample`, bucketed by which player was to move (621). When the game ends,
    the code walks each player's samples **backwards** (633) and blends the final
    game outcome with each state's own MCTS value:

    ```text
    label      = (value + sample.value) * 0.5        # line 634
    value      = value*LAMBDA + sample.value*(1-LAMBDA)   # LAMBDA = 0.9, line 635
    ```

    So the credit assigned to an early move is mostly the eventual result, softened
    by the searched value of the states along the way — a TD(λ)-style blend between
    "what actually happened" and "what search thought". The demo below runs this
    exact recurrence on a toy game.
    """
    )
    return


@app.cell
def _(pd):
    # Reproduce the backward credit-assignment recurrence of source lines 627-637.
    def assign_credit(mcts_values, outcome, lam=0.9):
        # mcts_values in play order; recurrence runs in REVERSE (line 633).
        value = float(outcome)
        rows = []
        for step, sv in reversed(list(enumerate(mcts_values))):
            label = (value + sv) * 0.5
            value = value * lam + sv * (1.0 - lam)
            rows.append(
                {
                    "step": step,
                    "state MCTS value": sv,
                    "label written": round(label, 4),
                    "value carried up": round(value, 4),
                }
            )
        rows.reverse()
        return rows

    _win = assign_credit([0.10, -0.20, 0.40, 0.75], outcome=1.0)
    credit_df = pd.DataFrame(_win)
    return (credit_df,)


@app.cell
def _(credit_df, mo):
    mo.md(
        f"""
    #### Demo — backward credit assignment on a won game

    A 4-move game that ends in a **win** (`outcome = +1`), with per-state MCTS
    values `[0.10, -0.20, 0.40, 0.75]`. The recurrence runs from the last move
    back to the first. Late states get labels close to the win; earlier states are
    pulled toward the outcome but tempered by their own search value — exactly the
    `LAMBDA = 0.9` blend in the source.

    {mo.ui.table(credit_df, selection=None)}
    """
    )
    return


@app.cell
def _(mo, show):
    mo.md(
        f"""
    ### 12b — The training step and the masked policy loss (lines 639–695)

    ```python
{show(639, 695)}
    ```

    **Thought process — this is the part everyone misreads, so read it slowly.**
    For each batch of 128 samples, the author packs a **rectangular** tensor by
    padding every sample's candidate list to length **64** (lines 659–664): real
    candidates get `mask = 1.0`; padding slots get `mask = 0.0` **and** a zero
    label **and** an empty bag offset. Then the loss (685–690):

    ```python
    loss_enc = loss_fn_enc(out_enc, label_tensor_enc)   # value head — ONE scalar/sample, NOT masked
    loss_dec = loss_fn_dec(out_dec, label_tensor_dec)   # policy head — [BATCH, 64] per-slot errors
    loss_dec = loss_dec * mask_tensor                   # zero out the padding slots
    loss_dec = loss_dec.sum() / float(BATCH_SIZE)       # sum real-slot errors, normalize by batch
    loss = loss_enc + loss_dec
    ```

    **✅ verified from code — the key facts:**

    - The multiply-by-mask **before** `.sum()` means padding slots contribute
      **exactly zero gradient**; they do not dilute the loss toward zero.
    - The **value/encoder loss is not masked** — it is one scalar per sample
      (686), so there is nothing to pad.
    - This is why `64` is a *padding width*, not a fixed action head (Section 9):
      the mask governs which of the 64 slots are real for *this* sample.
    """
    )
    return


@app.cell
def _(pd):
    # Reproduce the masked-policy-loss arithmetic of source lines 687-689 to PROVE
    # padding contributes zero. Pure Python (Huber with delta=0.1, reduction none).
    def huber(pred, target, delta=0.1):
        a = abs(pred - target)
        return 0.5 * a * a if a <= delta else delta * (a - 0.5 * delta)

    # One sample: 3 real candidates, padded to 6 (stand-in for 64).
    real = [(0.5, 0.7), (-0.2, -0.1), (0.9, 0.4)]   # (pred, target)
    pad = [(0.8, 0.0), (0.3, 0.0), (-0.6, 0.0)]     # padding preds are arbitrary
    mask = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    per_slot = [huber(p, t) for p, t in real + pad]
    masked = [e * m for e, m in zip(per_slot, mask)]

    mask_rows = [
        {
            "slot": i,
            "kind": "real" if mask[i] else "padding",
            "raw Huber": round(per_slot[i], 5),
            "mask": mask[i],
            "after mask": round(masked[i], 5),
        }
        for i in range(len(mask))
    ]
    mask_df = pd.DataFrame(mask_rows)
    unmasked_sum = round(sum(per_slot), 5)
    masked_sum = round(sum(masked), 5)
    padding_contrib = round(unmasked_sum - masked_sum, 5)
    return mask_df, masked_sum, padding_contrib, unmasked_sum


@app.cell
def _(mask_df, masked_sum, mo, padding_contrib, unmasked_sum):
    mo.md(
        f"""
    #### Demo — the mask really does zero out padding

    A single sample with 3 real candidates padded to 6 slots. If we naively summed
    all Huber errors we would get **{unmasked_sum}**; after `× mask` the sum is
    **{masked_sum}**, so the padding slots contributed **{padding_contrib}** — i.e.
    nothing. That is the source's `loss_dec = loss_dec * mask_tensor` at work.

    {mo.ui.table(mask_df, selection=None)}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Wrap-up — the whole pipeline, and what to fix next

    ### The loop in one diagram
    ```text
    ┌─ for 5 cycles ───────────────────────────────────────────────┐
    │  save checkpoint                                              │
    │  EVAL:      50 games  mcts_agent vs random_agent → win rate   │
    │  SELF-PLAY: 100 games mcts_agent vs mcts_agent               │
    │      each decision → (move, LearnSample)                      │
    │      game end → backward TD(λ=0.9) credit → training targets  │
    │  TRAIN:     batches of 128, pad candidates to 64,            │
    │      Huber(value) + masked Huber(policy), AdamW               │
    └──────────────────────────────────────────────────────────────┘
    Inside mcts_agent: determinize hidden info → 10 PUCT expansions
      → play most-visited child → labels = advantages + root value
    ```

    ### ✅ Verified from the source
    - One Transformer with a **value head** (state → [-1,1]) and a **policy head**
      (one score per legal candidate move).
    - Candidates are **combinations of option indices**, enumerated
      lexicographically and **capped at 64** (a scoring/padding width, not a fixed
      action space).
    - MCTS = 10 PUCT expansions; the move played is the **most-visited** root child.
    - Training targets: value = root average; policy = per-move **advantage**,
      clamped to [-1,1]; padding contributes **zero** loss via the mask.
    - Self-play credit uses a **TD(λ=0.9)** backward blend of outcome and search
      value.

    ### ⚠️ Limitations to carry forward (honest, from the code)
    1. **Incomplete determinization** — the opponent's hidden cards are dummies
       (Snorlax / Basic Energy), so there is **no real opponent modelling**.
    2. **Candidate truncation** — beyond 64 combinations, legal moves are dropped
       unseen; worse when `maxCount > 1`.
    3. **Weak evaluation** — win rate is measured only vs a **random** agent, which
       proves legality/learning-signal, not ladder strength.
    4. This same-competition result is on record: SearchScorer heuristic **660.5 μ**
       vs RL+MCTS basic **651.3 μ** vs wider-field RL+MCTS **580.6 μ** — a learned
       policy here did **not** beat a strong search/heuristic baseline. Treat that
       as a caution, not proof RL can't work: it must be trained and measured
       against the **real field**, not random or narrow opponents.

    ### ❓ Undocumented design choices (open questions, do NOT invent reasons)
    - Why `64` candidates, `SEARCH_COUNT=10`, `encoder_size=22000`,
      `num_words_encoder=24`, the `*10.0` prior sharpening, `LAMBDA=0.9`, the
      exploration constant `0.4`, and the various normalisation divisors.
    - Whether the ordering of `obs.select.option` puts good moves early (which would
      make the 64-cap truncation much less harmful).

    These are experiments to run against our own replay corpus — not blanks to fill
    with plausible-sounding rationales.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Recall check — answer in plain English

    1. The game presents 12 option entries and requires 2 selections. What does the
       candidate `[0, 1]` mean, how many legal pairs exist, and does the sample
       score all of them?
    2. Why is calling `64` a "fixed action head" misleading?
    3. In `mcts_agent`, which child does the agent actually play — highest value or
       most visited? What is the policy training target for each child?
    4. What exactly does `loss_dec = loss_dec * mask_tensor` accomplish, and why is
       the value loss not masked?
    5. Name the single biggest correctness limitation of the search, and the one
       thing wrong with evaluating only against `random_agent`.

    (Answers are all in Sections 9, 10, and 12 above — with source line numbers.)

    ## How to run this notebook
    ```bash
    cd ~/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026
    uv run marimo edit notebooks/05_kiyotah_rl_mcts_full_walkthrough.py   # interactive
    uv run marimo run  notebooks/05_kiyotah_rl_mcts_full_walkthrough.py   # read-only app
    uv run python      notebooks/05_kiyotah_rl_mcts_full_walkthrough.py   # headless, exits 0 if clean
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()
