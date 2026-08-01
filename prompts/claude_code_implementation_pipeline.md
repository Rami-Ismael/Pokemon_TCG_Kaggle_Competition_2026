# Claude Code Prompt — Pokémon TCG AI Battle Challenge: Full Implementation Pipeline (Simulation + Strategy tracks)

> Paste the block below (from `# GOAL` to `# ACCEPTANCE`) into Claude Code. It is self-contained — it embeds the competition rules and proven pitfalls so you do not need any external context.

---

# GOAL
Build a reproducible **implementation pipeline** in this repo (`~/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026`) that produces (1) a competition-ready Pokémon TCG agent for the **Simulation track** and (2) the real evidence/analysis behind a **Strategy-track writeup** (the Strategy track requires a Simulation entry, so it is downstream of the agent). You are building **code, scripts, and a verification harness — NOT submitting**. Do NOT run `kaggle competitions submit`; that burns one of only 5 daily slots and is out of scope.

# HARD CONSTRAINTS (competition rules + proven pitfalls — must respect all)
- **Engine** = `cabt` (C++ Pokémon TCG simulator inside `kaggle-environments`). Agent contract:
  `def agent(obs_dict: dict) -> list[int]` — return indices into `obs_dict["select"]["option"]`, length within `[minCount, maxCount]`, no duplicates. An illegal index (out of range, or not in the offered list) forfeits the game. When `obs_dict["select"] is None`, return the 60-card deck id list.
- **Two hardware contexts.** Training is unrestricted (your own GPU/CPU). EVAL is the Kaggle **CPU-only** sandbox: ~30 GiB RAM, `runTimeout` 2000 s/episode, **per-agent overage bank 600 s** (exceeding it errors the game out), and **NO network ingress/egress during a game** (Rules §12). Every weight/logic must live INSIDE the bundle. Design neural parts for CPU inference near real-time.
- **Submission bundle**: top-level `main.py` (NOT nested) + `deck.csv` + `cg/` copied in. Build with `tar -czvf submission.tar.gz *`. Size cap ≈ **197.7 MiB**. Only the latest 2 submissions are active. On upload, a validation episode plays your agent vs a copy of itself.
- **External data/models** allowed if publicly available and free to all participants; document provenance.
- **Action space is VARIABLE per observation** (observed option count 0–29; true legal ceiling C(N,k) measured up to **8,568**, with 1,812 states exceeding a 64 cap). Use a **per-option scorer + legality mask**, never a fixed `N_ACTIONS` output head. The "64 = game max" belief is FALSE.
- **Win-condition resolution** (verify the engine actually implements this once locally importable): count win conditions met per player (prizes == 0, no Pokémon in play, deck-out); the player satisfying MORE conditions wins; tie on count → draw. Must pass the **Dusknoir scenario**: P1 = 0 prizes + 0 Pokémon in play (2 conditions) vs P2 = 2 prizes + 0 Pokémon in play (1 condition) → **P1 wins**.
- **Eval timing**: 600 s overage bank is the real budget. Give search a tight node/time cap and ALWAYS keep a minimal legal fallback so a slow decision cannot forfeit.

# REPO CONVENTIONS (already established — extend, do not reinvent)
- `agents/<deck_name>/agent_core.py` = factored policy module (set `agent_core.my_deck = <60 ids>` after import; do NOT read deck.csv at module level).
- `submissions/<deck_name>/` holds `main.py` + `deck.csv` + `submission.tar.gz`.
- `data/external/cg-lib/cg` = engine SDK (api.py, sim.py, lib*.so/.dylib). Run `uv run python` / `uv run marimo` from repo root.
- Reuse existing scripts: `scripts/download_episodes.py` (resume-safe ~21 GiB/day pull), `scripts/scan_obs_surface.py` (obs_dict surface scan), `scripts/verify_cabt_agent.py` (self-play + search-path-proof harness).
- Existing artifacts to build ON TOP of: `agents/mega_lucario/agent_core_improved.py` + `bc_prior.py`, `models/il_agent/model.safetensors` + `config.json` (behavior-cloned agent), `submissions/mega_lucario_improved/`.
- Episode datasets: ~21 GiB/day of per-episode JSON (`<EpisodeId>.json`), `manifest.csv` has a UTF-8 BOM (`\ufeffdate`). Need **≥2 days** for a held-out-DAY train/eval split (NOT a random split).

# BUILD ORDER (implement in this sequence; each step has a verification gate — do not skip ahead)
1. **Scaffold & engine bring-up.** Set up uv/venv, pull `kiyotah/cg-lib` into `data/external/cg-lib`, apply the macOS dylib fix (copy `libcg.dylib` + `libcg-arm64.so` from the venv's `kaggle_environments/envs/cabt/cg/` into `data/external/cg-lib/cg/`). Smoke-test `from kaggle_environments import make; env = make("cabt")` + `env.run(["random","random"])`. Confirm `cg.api.search_begin` works locally with the 6-arg signature (see Gotchas).
2. **Observation encoder + action interface.** Write `encode_obs(obs_dict) -> tensor` sourced from the REAL obs_dict shape (scan with `scripts/scan_obs_surface.py`; do NOT infer semantics from slot arithmetic). Implement a **variable option-scoring head + mask** (`Categorical` over offered options; pad+mask to -inf for batched PPO). Document, for each fixed slot: collection, ordering, empty-slot behavior, and information loss for truncated/omitted variable collections (hand, discard, prizes, legal actions).
3. **Baseline agent ladder** (each must pass the self-play gate before the next):
   - (a) `random_legal` — returns lowest legal indices; never-crash floor.
   - (b) **Heuristic "Improved Probabilistic" pattern** (Ternovsky ~967.7): `choose()` returns a FULL RANKED list of legal indices (best first), not one pick.
   - (c) **Search re-ranker** (the proven high-score pattern): take the heuristic's top-8 candidates, re-rank each by a one-turn engine rollout via `cg.api` `search_begin/search_step/search_end` scored with `evaluate_state`; flat UCB1 bandit at depth 1 (no tree, no opponent model). Keep the three never-crash fallbacks: search fails → heuristic ranking; heuristic fails → `[0,1,...]`; parse fails → deck or `[0]`.
4. **Data pipeline for IL/BC.** Extend `scripts/download_episodes.py` to pull ≥1 train day + ≥1 eval day into `data/episodes/splits/{train,eval}-YYYY-MM-DD/`. Stream-parse episodes to a BC dataset: extract both players' decks (hash card-id multiset → deck identity), take the **winner's** moves as (encoded_state → chosen option index), store memory-bounded (do NOT load 21 GiB into RAM). Keep the `splits/splits.json` manifest.
5. **IL / behavior-cloning agent.** Train (or fine-tune the existing `models/il_agent`) a policy that scores offered options from the encoder, with the same mask applied at both rollout and update time. Keep weights under the 197.7 MiB cap; verify CPU-loadable. Wire it as a **prior / initial ranking** feeding the heuristic+search-reranker.
6. **Evaluation harness** (`scripts/verify_cabt_agent.py`, extend it): self-play assert, search-path proof, win-condition Dusknoir test, benchmark vs baselines, and a **bundle sandbox reproduction** (build tar → extract to scratch → chdir scratch → exec `main.py` with ONLY bundled files on `sys.path` → self-play again).
7. **Submission builder.** Script that assembles `submissions/<deck>/` with top-level `main.py` (loads `deck.csv` relative to its own dir, fallback `/kaggle_simulations/agent/deck.csv`), copies `cg/`, verifies deck legality (exactly 60 cards) and size < 197.7 MiB, and includes the LOCAL/LADDER unwrap helpers. STOP here — do not submit.
8. **Strategy-track writeup** (`reports/`). Generate the report FROM REAL RESULTS: agent architecture (heuristic + search re-ranker + IL prior), deck/archetype choice and why, determinization approach (note: sample only own deck order; opponent hidden info is a placeholder prediction, not a belief model), eval numbers (self-play, baseline benchmarks, search-path proof), ablations, and honest limitations. Cite the Dusknoir win-condition handling.

# GOTCHAS TO BAKE INTO THE CODE (from prior porting experience — these silently break naive ports)
- **cg-lib LOCAL vs LADDER API mismatch** (write ONE file that runs in both with unwrap helpers):
  - `search_begin` requires **SIX** prediction args locally: `your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active` (not just `your_deck=`).
  - LOCAL engine **REJECTS `0` placeholders** (`ValueError("Invalid Card ID")`). Build predictions from `my_deck` padded with `filler=6`; only need a Basic Pokémon on the opponent side at setup. Use a `_predict(obs)` helper.
  - LOCAL `cg.api` returns a `SearchState` **directly and raises on error**; LADDER wraps it in `ApiResult(.state/.error)`. Helpers: `_state_of(obj)` → `obj.state` if present else `obj`; `_ok(obj)` → `getattr(obj, "error", 0) == 0`.
  - `SearchState` field is `.searchId` (NOT `.id`).
  - Always `search_end()` in a `try/finally` to free engine memory.
- **macOS import**: see step 1 dylib copy. Harmless in the Linux eval sandbox (loader picks `libcg.so`).
- **`main.py` deck load**: relative to its own dir, fallback `/kaggle_simulations/agent/deck.csv`.
- **Encoder honesty**: read source, do not infer slot meaning from arithmetic. Flag information loss explicitly.
- **Determinization**: only sample own deck order; opponent hidden info is a placeholder, not a belief model. Don't over-engineer opponent determinization unless running a deliberate experiment.

# OUT OF SCOPE / DO NOT
- Do NOT run `kaggle competitions submit`.
- Do NOT assume CUDA at eval time.
- Do NOT hardcode a fixed action vocabulary / `N_ACTIONS` head.
- Do NOT use all-zeros determinization locally (use `my_deck`-based padding; `0` only works on the ladder).
- Do NOT leave README/tree out of sync with the actual repo layout.

# VERIFICATION GATES (all must pass before calling the pipeline "done")
1. Self-play: `env.run([agent, agent])` → both `status == "DONE"` and `sorted(rewards) == [-1, 1]`.
2. Bundle check: build tar → extract to scratch → `os.chdir(scratch)` → exec `main.py` with only bundled files on `sys.path` → self-play passes again.
3. Search-path proof: take a real MAIN `obs`, set `obs.select.maxCount = min(8, len(obs.select.option))`, confirm `simulate_action` returns finite and the re-rank happens.
4. Win-condition test: Dusknoir scenario → P1 wins.
5. Data pipeline: ≥1 train day + ≥1 eval day streamed; BC dataset built from winner moves; memory-bounded.
6. IL agent loads on CPU, infers within the tight time budget, returns legal indices.

# ACCEPTANCE
When done, the repo contains: a working agent ladder (`random_legal` → heuristic → search-reranker → IL-prior), a streaming data pipeline, an eval/verification harness, a submission builder (size-checked, not submitted), and a draft Strategy writeup in `reports/` backed by real eval numbers. All six gates are green. Report what you built, which gates passed, and exactly what remains for the human to do (deck tuning, final submission, writeup polish).

---

## How to use (notes for the human, not part of the prompt)
- The block above is what you paste into Claude Code. It is self-contained.
- After Claude Code finishes, the human runs the submission (5/day limit) and polishes the Strategy writeup.
- Anchor deck: **Mega Lucario ex** (repo already has the baseline + an IL model). Swap the archetype by changing `my_deck` / the deck.csv.
