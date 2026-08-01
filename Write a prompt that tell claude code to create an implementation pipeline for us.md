# Write a prompt that tell claude code to create an implementation pipeline for us

The prompt below is the brief to hand to Claude Code. The checkbox list at the
bottom is the acceptance checklist — each open question becomes one todo item
that the generated pipeline must resolve (or explicitly defer with a reason).

---

## Prompt for Claude Code

You are a senior ML engineer and Kaggle competition specialist. Write clean,
runnable **PyTorch** code for a **Behavioral Cloning (BC)** imitation-learning
pipeline that learns to play Pokémon TCG from the official **daily episode replay
datasets** of the Kaggle *Pokémon TCG AI Battle Challenge* **Simulation (agent)
track**. (Note: the competition also has a separate Strategy/writeup track — this
code targets the battling agent, not the report. *[USER TO FILL: confirm you mean
the Simulation track, not the Strategy writeup track.]*)

**Data source (ground truth — use these exact shapes).**
- Datasets: `kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD` — a directory of
  per-episode JSON files, one per episode (`<EpisodeId>.json`, ~21 GiB/day, CC0 license).
- Each file body: `{ "info": {...}, "rewards": [int,int], "statuses": [...],
  "steps": [[obsAgent0, obsAgent1], [obsAgent0, obsAgent1], ...] }`.
- Per-step agent record: `{ "action": list[int], "observation"|"obs": obs_dict,
  "reward", "status", "info" }`.
- `obs_dict.select.option` = the **legal options the engine offered this step** —
  a **variable-length list (observed 0–29)** of `{index, type, ...}`. This is what
  your policy scores/samples over.
- `obs_dict.current` = the visible board (your POV): `players[]` with `active`,
  `bench` (max 5), `hand`/`handCount`, `deckCount`, `discard`, `prize`, `stadium`,
  status flags, `turn`, `turnActionCount`. **Opponent `hand` is `null`** (hidden
  information) — design around partial observability, do not assume you see it.
- Recorded `action` = the option indices the expert actually picked (your BC target labels).
- Split rule: use a **held-out DAY** split (e.g. train on `2026-07-26`, eval on
  `2026-07-27`), NOT a random split — see `scripts/replay_episode.py` for the
  split-folder convention.

**Architecture decision you must make and justify (core of the task).**
You are choosing a **Transformer** (self-attention over the move/state history) as
the policy backbone because, per parameter count, it is the most effective model for
this competition and its attention naturally captures **game history**. Explicitly
compare it against a **regular MLP baseline** in the style of **CleanRL / PufferLib**
implementations (flat encoding → MLP → action head), and address the tradeoff you raised:
- **Why Transformer over MLP:** history/ordering matters in TCG; attention models
  long-range dependencies without hand-built recurrence.
- **Why not LSTM:** you found adding an LSTM hard — note that a Transformer avoids
  recurrent state management and parallelizes over the history sequence, while an
  LSTM forces sequential hidden-state bookkeeping at both train and (CPU) inference time.
- **The real tiebreaker — eval constraints:** submitted agents run **CPU-only** in a
  Kaggle sandbox with a **600 s per-agent overage budget** and a **~200 s/step** soft
  cap, no GPU, no network. So the comparison must weigh *parameter efficiency per size*
  against *CPU inference cost*: a Transformer's attention is O(seq²) — state the max
  history length you'll attend over and why it stays within budget at play time.

**Deliverable / output format.**
Produce a small, well-structured codebase (not a single notebook cell) with:
1. `data.py` — loader that streams episodes from the split folders, extracts
  `(obs_dict, action)` BC pairs, handles variable-length `option`, and pads/truncates
  history to a fixed context window for the Transformer.
2. `model.py` — (a) the **Transformer policy** (tokenize each history step + current
  legal options, attend, then a **per-option scorer head** that scores only the offered
  options — do NOT use a fixed `N_ACTIONS` head, since option count varies) and (b) an
  **MLP baseline** sharing the same option-scoring interface, for the comparison.
3. `train.py` — BC training loop (cross-entropy over expert `action` indices, masked to
  legal options), with the held-out-day eval harness reporting action-match accuracy / top-k accuracy.
4. `agent.py` — the competition interface: `def agent(obs_dict: dict) -> list[int]` that
  encodes the current obs + recent history, runs the trained policy, returns legal
  `select.option` indices within `[minCount, maxCount]`, with a **never-crash legal
  fallback** (lowest legal indices) if the model errors.
5. A short `README` noting the architecture tradeoff conclusions and how to build the
  `submission.tar.gz` (`main.py` + `deck.csv` at top level).

**Success criteria.**
- Pipeline trains on the daily episodes and loads real `(obs, action)` pairs end-to-end.
- Policy scores only legal options; produced `agent()` returns valid indices and survives
  a self-play `env.run([agent, agent])` without illegal-move forfeits.
- The Transformer vs MLP comparison is reported with numbers (accuracy + a CPU
  inference-time estimate per move), and a clear recommendation for the competition given the budget.

**Edge cases to handle.**
- Variable-length `select.option` (0–29) → masked option-scoring head, never a fixed output layer.
- Null opponent hand / hidden info → no opponent-hand features or explicit "unknown" sentinel.
- `obs.select is None` → deck-selection step (return the 60-card deck), kept outside the battle policy.
- Out-of-range/illegal returned index → forfeit; always fall back to a legal move.

---

## Open Questions — Todo Checklist (one checkbox per item)

- [ ] **Card encoding:** embed each card id directly (1267-card vocab), or featurize (HP, energy cost, type, stage)? State choice and its impact on the ~198 MiB submission cap.
- [ ] **Variable-length collections:** how are hand (~10), bench (max 5), discard, prizes, deck handled — truncate / pad-with-sentinel / pool? Don't infer slot meaning from arithmetic.
- [ ] **Order sensitivity:** are bench/hand order meaningful, or should those sets be encoded order-invariant (DeepSets / Set-Transformer) rather than positionally?
- [ ] **Per-option scoring features:** what features go into each offered `select.option` (type, area, inPlayArea, inPlayIndex) for the scorer head?
- [ ] **Context window:** how many past steps does the Transformer attend over, and what fills them — `current` board only, or also the `logs` move-history? Prove O(seq²) attention fits the 600 s CPU overage budget.
- [ ] **Deck conditioning:** one universal policy across all decks, or condition on deck identity (submission ships `deck.csv`)? State the choice.
- [ ] **Winner filtering:** clone all trajectories, or only winners (data carries `rewards`/winner)?
- [ ] **Multi-index actions:** `action` is a `list` bounded by `minCount`/`maxCount`. Autoregressive per-index pick, or whole-set scoring? Categorical-over-combinations explodes.
- [ ] **Distribution shift:** daily data is sampled top-ranked matchups only; BC inherits expert blind spots (missing recovery states). Plan a DAgger follow-up, or accept BC-only?
- [ ] **Eval metric:** action-match accuracy on held-out replays, or actual win rate via self-play? They diverge.
- [ ] **Deck-selection step:** episodes include `select.type == 9` (return 60 card IDs). Clone the deck, or keep a fixed `deck.csv` and BC only the battle policy?
- [ ] **Hidden opponent info:** opponent `hand` is `null`. Model it (determinization), or follow the 967.7 agent and model *no* opponent hidden state?
