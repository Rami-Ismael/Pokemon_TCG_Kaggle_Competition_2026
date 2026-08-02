# notebooks/reference — Kaggle Source Crawler Mirror

Mirror of competition Code-tab notebooks pulled via `uv run kaggle kernels pull <user>/<slug> -p notebooks/reference/<folder>`.
Each folder holds the verbatim `.ipynb`. For code-heavy single-cell/long notebooks, a `source_with_linenumbers.py` export (all code cells joined) lets walkthroughs cite real editor line numbers. Conventions match the existing `kiyotah-rl-mcts` mirror.

## Seed (this crawl's origin)

- **ahmedabdelhmed/notebook94c79d7292** — *Bayesian Heuristics with Collision-Avoidance for Pokémon TCG* `[Untitled Notebook]`
  - Self-contained heuristic play-agent: Bayesian deck-tracking belief + multi-objective reward shaping `R = α(Dmg/Energy) + β(HP) − γ(Waste)` + collision-avoidance retreat, Lightning Aggro archetype. `agent(obs, config)` entrypoint. 28-D float16 state vector.

## Similar play-agents in this mirror (Simulation division)

| Folder | Kaggle ref | Votes | Method | Status |
|--------|-----------|-------|--------|--------|
| avikdas567-heuristic-agent | [avikdas567/ptcg-ai-battle-heuristic-agent-data-pipeline](https://www.kaggle.com/code/avikdas567/ptcg-ai-battle-heuristic-agent-data-pipeline) | 23 | Heuristic agent + data pipeline | ADDED, wired as `avikdas567_heuristic` (weak: str(option) substring scoring) |
| lucifer19-battlecore-agent | [lucifer19/battlecore-compact-agent](https://www.kaggle.com/code/lucifer19/battlecore-compact-agent) | 22 | Payload-wrapped agent + falsification/validation protocol (no plain-text agent() source) | ADDED |
| romanrozen-strong-start-agent | [romanrozen/strong-start-baseline-agent-v10-lb-950](https://www.kaggle.com/code/romanrozen/strong-start-baseline-agent-v10-lb-950) | 142 | Probabilistic Expectimax baseline + UCB1/MCTS re-ranker over real cg engine search rollouts (LB 950+) | ADDED, wired as `romanrozen_strong_start`; **byte-identical to `aristophanivan`** (same lineage), only this copy wired |
| makimakiai-tiny-rl-baseline | [makimakiai/ptcg-tiny-rl-to-submission-baseline-guide](https://www.kaggle.com/code/makimakiai/ptcg-tiny-rl-to-submission-baseline-guide) | 49 | Beginner RL -> submission baseline guide | ADDED, wired as `makimakiai_rl` (**UNTRAINED default weights** — bake not reproduced; plays near-random) |
| dovhan | [dovhan/code-pokemon-ai](https://www.kaggle.com/code/dovhan/code-pokemon-ai) | n/a | EDA + Bayesian/heuristic scoring walkthrough | pre-existing |
| aristophanivan | [aristophanivan/improved-probabilistic-agent](https://www.kaggle.com/code/aristophanivan/improved-probabilistic-agent) | 172 | Probabilistic Expectimax agent (multi-part) | pre-existing |
| kiyotah-rl-mcts | [kiyotah/reinforcement-learning-and-mcts-sample-code](https://www.kaggle.com/code/kiyotah/reinforcement-learning-and-mcts-sample-code) | 935 | Host: RL + MCTS sample code (single 689-line cell) | pre-existing |
| kiyotah-rule-based | [kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck) | 909 | Host: sample rule-based agent (Mega Lucario ex) | pre-existing |
| kiyotah-dragapult | [kiyotah/a-sample-rule-based-agent-dragapult-ex-deck](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-dragapult-ex-deck) | n/a | Host: sample rule-based agent (Dragapult ex) | ADDED, wired into benchmark_agents.py |
| kiyotah-iono | [kiyotah/a-sample-rule-based-agent-iono-s-deck](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck) | n/a | Host: sample rule-based agent (Iono's deck) | ADDED, wired into benchmark_agents.py |
| kiyotah-abomasnow | [kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck) | n/a | Host: sample rule-based agent (Mega Abomasnow ex) | ADDED, wired into benchmark_agents.py |
| dedquoc-rule-engine | [dedquoc/the-pok-mon-rule-based-engine](https://www.kaggle.com/code/dedquoc/the-pok-mon-rule-based-engine) | n/a | Rule-based engine, Mega Lucario ex deck (same archetype as our rule_baseline, independent implementation) | ADDED, wired into benchmark_agents.py |
| ryotasueyoshi-alakazam | [ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th](https://www.kaggle.com/code/ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th) | n/a | Rule-based Alakazam, author-claimed "best 5th" | ADDED, wired into benchmark_agents.py |
| makthanithin-improved-probabilistic | [makthanithin/improved-probabilistic-agent](https://www.kaggle.com/code/makthanithin/improved-probabilistic-agent) | n/a | Probabilistic Expectimax agent, Mega Lucario ex deck variant. Upstream of `aristophanivan`'s notebook and of `agents/improved_probabilistic/main.py` — **not** the ancestor of `agent_core_improved` (three separate lineages; see `Agent Teardown — Improved Probabilistic (967.7)` in the yakumsi vault) | ADDED, wired into benchmark_agents.py |
| mechi22-alakazam | [mechi22/ptcg-1070-9-alakazam-rule-based-skeleton](https://www.kaggle.com/code/mechi22/ptcg-1070-9-alakazam-rule-based-skeleton) | n/a | Rule-based Alakazam, author-claimed LB 1070.9 | ADDED (decoded), wired into benchmark_agents.py |
| jek1wantaufik-strategy-analysis | [jek1wantaufik/pok-mon-tcg-ai-strategy-analysis](https://www.kaggle.com/code/jek1wantaufik/pok-mon-tcg-ai-strategy-analysis) | n/a | **Analysis-only notebook** (AST metrics + strategy report); the playable agent is a rule-based Mega Lucario ex `main.py`+`deck.pkl` recovered from the Kaggle model `jek1wantaufik/buddy` it references. See NOTES.md | ADDED (recovered from model), wired into benchmark_agents.py as `jek1wantaufik_lucario` |

## Phase 1 archetype-coverage recruitment (2026-08-02)

Sourced from `makimakiai/ptcg-public-23-plus-sample-4-roster-update`'s referenced-notebooks
list (pulled via `kaggle kernels pull`, not the browser — Kaggle notebook pages are
client-rendered and the browser tool couldn't reliably extract cell content).

| Folder | Kaggle ref | Archetype | Status |
|--------|-----------|-----------|--------|
| plamen06-steel | [plamen06/pokemon-steel](https://www.kaggle.com/code/plamen06/pokemon-steel) | Archaludon ex / Duraludon / Cinderace, Metal-type ("metal-tempo") | **ADDED**, wired as `agents/plamen06_steel/agent_core.py` — first non-Lucario/Alakazam/Dragapult/Bellibolt/Abomasnow archetype in the pool |
| aman5153684-crustle-fighting | [aman5153684/a-crustle-aware-fighting-agent](https://www.kaggle.com/code/aman5153684/a-crustle-aware-fighting-agent) | Mega Lucario ex (Crustle-matchup tuning, same deck IDs 673-678 as the frozen deck) | checked, NOT added — same archetype we already have 9 copies of, adds no diversity |
| seokjeongeum-psychic-v8 | seokjeongeum/strong-start-psychic-anti-meta-v8-lb-1100 | unknown (Psychic, per label) | **403 Forbidden on pull** — private/unavailable, same pattern as the already-noted seokjeongeum 1208 notebook |

`plamen06-steel`'s `main.py` cell was adapted (not copied verbatim) before wiring in: its
`read_deck_csv()` read a relative `"deck.csv"` path, which under this repo's benchmark
harness (cwd = repo root, which has its own `deck.csv` for a *different* agent's frozen
deck) would have silently loaded the wrong deck instead of erroring. Changed to `return
my_deck`, matching every other bare-module agent's convention (external injection from a
sibling `deck.csv`, see `kiyotah_dragapult`). Safety-scanned (no subprocess/eval/exec/
network calls) and card-ID-verified against `cg.api.all_card_data()` before wiring in.

Also investigated and rejected as non-functional: `ahmedabdelhmed-bayesian-heuristic`
(this crawl's original seed notebook) references observation fields that do not exist in
this competition's real API (`obs.active`, `obs.bench`, `obs.initial_deck_composition`,
`obs.observed_discards`, `obs.estimated_opp_damage` — none of these are real; the actual
schema is `obs_dict['select']`/`obs_dict['current']`, see `notes/phase0_discovery_report.md`
§0.1) and never imports `cg.api` at all. Its claimed "78% win rate" and "Bayesian tracking
improves opponent prediction by 23%" are unverifiable against a fictional API and are very
likely fabricated/decorative rather than measured. Not wired into the benchmark pool.

## Honesty flags

- `lucifer19-battlecore-agent`: real agent, but its `.ipynb` embeds the agent as a compressed+base64 payload with an integrity gate — **no plain-text `agent()` in source**. No `source_with_linenumbers.py` export (would be the payload blob). Kept and flagged, not dropped.
- `kaiwalyaatulraut/pokemon-ai-battle-challenge-simulation-solution` (20 votes) was a candidate but **403 Forbidden** on pull — private/disabled for download. NOT included (no fabrication).
- `seokjeongeum/max-elo-1208-libraryout-w-crustle-great-tusk` (author-claimed LB 1208, sourced from the yakumsi vault's `Public Agent Roster — Makimakiai Matchup Matrix`) was a candidate but is **no longer reachable** — 403 Forbidden on the Kaggle API pull *and* "We can't find that page" on a direct browser visit (checked 2026-08-01). Deleted, made private, or re-slugged since the vault note captured it two days prior. NOT included (no fabrication); the requested teardown of this notebook could not be produced.
- `mechi22-alakazam`: `main.py` is embedded as a **base64 blob, SHA256-integrity-checked, not encrypted** — decoded locally (`notebooks/reference/mechi22-alakazam/main_decoded.py`) to a plain-text `agent()` with no red flags (no eval/exec/network/subprocess). Unlike `lucifer19-battlecore-agent`, full source recovery was possible here, so it's marked ADDED rather than excluded. Obfuscation appears to be an anti-fork measure, not a security concern.
- `dovhan`, `aristophanivan`, `kiyotah-*` were already present before this crawl (pre-existing mirror).
- Live leaderboard check (browser-rendered, 2026-08-01): #1 = 1298.5, top-8 cutoff = 1141.0, top-50 ≈ 1055.7 — all higher than the vault note's 2026-08-01 capture (1280.8 / 1122.0 / 1050.7), and no visible team in the top 49 matches the claimed 1208 score under any recognizable name. Ladder is live and moving; treat any stated score as time-stamped, not current.

## Benchmark-wiring wave (2026-08-02, consolidated) — widening the pool's strength range

Goal (vault note `Rule Base Agent in pokemon TCG` + Q25): make the local pool span a
wider strength range so it predicts the ladder, not just rank our own agents. Three
net-new, source-reviewed, cabt-compatible agents wired into `scripts/benchmark_agents.py`
(pool 17 → 20):

- **`romanrozen_strong_start`** — wired. LB ~950 Expectimax + UCB1/MCTS re-ranker over
  real cg engine search rollouts; strongest public opponent in the pool. Smoke: ~71% in a
  6-agent run (2nd behind kojimar_lucario).
- **`avikdas567_heuristic`** — wired. Weak (str(option) substring scoring); fills the tier
  between `random_legal` and the real policies (~29%).
- **`makimakiai_rl`** — wired, **UNTRAINED**. The upstream notebook ships only the RL
  *training harness*; its `LEARNED_WEIGHTS` bake is nondeterministic and was **not**
  reproduced, so this runs on the notebook's default weights. A legal, distinct-method
  (linear-weights-over-option-types) opponent, but **not** the tuned agent — it plays
  near-random (~29%, ≈ avikdas567/random tier). Kept for method diversity; drop it if the
  pool wants only faithful agents.

Reviewed but **deliberately not wired** (so this isn't mistaken for full coverage):

- **`aristophanivan`** — `main.py` is **byte-identical** to `romanrozen` (verified by diff);
  represented by `romanrozen_strong_start`.
- **kojimar `simple-baseline-matchup-tests`** — same notebook already wired on `main` as
  **`kojimar_lucario`** (which is the more faithful port: it uses the notebook's own literal
  `DECK`, not the generic sample deck). The freshly-pulled `kojimar_simple_baseline` port
  was a duplicate and was dropped.
- **`aryachovapratama-basic-heavy-aggro`** — `agent()` is safe and cabt-correct, but its
  generated `deck.csv` is **illegal in the cg engine** (9 ACE SPEC copies — Master Ball ×4,
  Amulet of Hope ×4… where the rule allows 1). Not wired; mirror kept for the trail
  (`notebooks/reference/aryachovapratama-basic-heavy-aggro/`).
- **`ahmedabdelhmed-bayesian-heuristic`** — invents a fake observation API; would crash in
  `make("cabt")` (already flagged in the Phase 1 section above).
- **`lucifer19-battlecore-agent`** (note's 860.2 BattleCore) — payload-wrapped, no plaintext
  `agent()` (first honesty flag).

Also fixed here: `benchmark_agents.py` couldn't find `cg` when run from a git worktree
(`data/external/cg-lib` isn't checked out there, and the other in-repo `cg` copies ship a
Linux-only `libcg.so` that dlopen-crashes on macOS). The cg-path resolver now walks
`REPO.parents` to find the primary checkout's `cg-lib` (whose `libcg.dylib` is the arm64
build).

## Files per folder

- **ahmedabdelhmed-bayesian-heuristic** — 🏆 Bayesian Heuristics with Collision-Avoidance for Pokémon TCG
  - files: notebook94c79d7292.ipynb, source_with_linenumbers.py

- **avikdas567-heuristic-agent** — Pokémon TCG AI Battle Challenge Simulation Pipeline
  - files: ptcg-ai-battle-heuristic-agent-data-pipeline.ipynb, source_with_linenumbers.py

- **lucifer19-battlecore-agent** — PTCG AI Battle — Max-Efficiency Challenger Build (V4)
  - files: battlecore-compact-agent.ipynb

- **romanrozen-strong-start-agent** — Algorithm: 04 Probabilistic Expectimax Agent\n\nThis notebook implements a strictly heuristic agent utilizing the 04 Probabilistic Expectimax Agent strategy to significantly outperform the simple-baseline.
  - files: source_with_linenumbers.py, strong-start-baseline-agent-v10-lb-950.ipynb

- **makimakiai-tiny-rl-baseline** — Pokemon TCG AI Battle: beginner RL-to-submission baseline
  - files: ptcg-tiny-rl-to-submission-baseline-guide.ipynb, source_with_linenumbers.py

- **dovhan** — # 📋 Table of Contents
  - files: code-pokemon-ai.ipynb, source_with_linenumbers.py

- **aristophanivan** — Algorithm: 04 Probabilistic Expectimax Agent\n\nThis notebook implements a strictly heuristic agent utilizing the 04 Probabilistic Expectimax Agent strategy to significantly outperform the simple-baseline.
  - files: agent_runner.py, improved-probabilistic-agent.ipynb, main.py, run_proof.json

- **kiyotah-rl-mcts** — (untitled)
  - files: README.md, reinforcement-learning-and-mcts-sample-code.ipynb, source_with_linenumbers.py

- **kiyotah-rule-based** — # Rule-Based Agent for Mega Lucario ex
  - files: a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb

- **kiyotah-dragapult** — Dragapult ex Deck sample rule-based agent
  - files: a-sample-rule-based-agent-dragapult-ex-deck.ipynb, source_with_linenumbers.py
  - wired as `agents/kiyotah_dragapult/agent_core.py` (deck-load + submission-packaging cells stripped, `def agent()` body kept verbatim)

- **kiyotah-iono** — Iono's Deck sample rule-based agent
  - files: a-sample-rule-based-agent-iono-s-deck.ipynb, source_with_linenumbers.py
  - wired as `agents/kiyotah_iono/agent_core.py` (same stripping as above)

- **kiyotah-abomasnow** — Mega Abomasnow ex Deck sample rule-based agent
  - files: a-sample-rule-based-agent-mega-abomasnow-ex-deck.ipynb, source_with_linenumbers.py
  - wired as `agents/kiyotah_abomasnow/agent_core.py` (same stripping as above)

- **dedquoc-rule-engine** — The Pokémon Rule-Based Engine (Mega Lucario ex, "Kaggle Production Ready Edition")
  - files: the-pok-mon-rule-based-engine.ipynb, source_with_linenumbers.py
  - wired as `agents/dedquoc_rule_engine/agent_core.py`; no embedded deck list, so `deck.csv` reuses `submissions/mega_lucario/deck.csv` verbatim (same archetype the notebook's own docstring declares)

- **ryotasueyoshi-alakazam** — Rule-based, not-psychic Alakazam, author-claimed best 5th
  - files: rule-based-not-psychic-alakazam-best-5th.ipynb, source_with_linenumbers.py
  - wired as `agents/ryotasueyoshi_alakazam/agent_core.py`; deck.csv taken from the notebook's own embedded `%%writefile deck.csv` cell

- **makthanithin-improved-probabilistic** — Improved Probabilistic Agent (Mega Lucario ex variant)
  - files: improved-probabilistic-agent.ipynb, source_with_linenumbers.py
  - wired as `agents/makthanithin_improved_prob/agent_core.py`; kept the notebook's own literal `DECK` constant, dropped the local file round-trip (`Path("deck.csv").write_text(...)` then read-back) in favor of a direct `my_deck = DECK` assignment

- **mechi22-alakazam** — PTCG 1070.9 Alakazam rule-based skeleton
  - files: ptcg-1070-9-alakazam-rule-based-skeleton.ipynb, source_with_linenumbers.py, main_decoded.py (base64-decoded `main.py`, see Honesty flags)
  - wired as `agents/mechi22_alakazam/agent_core.py`; deck.csv decoded from the notebook's own embedded `PAYLOADS["deck.csv"]` blob

## Regenerate source_with_linenumbers.py
```python
import json, pathlib
nb = pathlib.Path('notebooks/reference/<folder>/<notebook>.ipynb').read_text()
src = '\n'.join(''.join(c['source']) for c in json.loads(nb)['cells'] if c['cell_type']=='code')
pathlib.Path('notebooks/reference/<folder>/source_with_linenumbers.py').write_text(src + '\n')
```
