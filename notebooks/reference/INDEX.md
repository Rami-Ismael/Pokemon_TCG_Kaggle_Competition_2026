# notebooks/reference — Kaggle Source Crawler Mirror

Mirror of competition Code-tab notebooks pulled via `uv run kaggle kernels pull <user>/<slug> -p notebooks/reference/<folder>`.
Each folder holds the verbatim `.ipynb`. For code-heavy single-cell/long notebooks, a `source_with_linenumbers.py` export (all code cells joined) lets walkthroughs cite real editor line numbers. Conventions match the existing `kiyotah-rl-mcts` mirror.

## Seed (this crawl's origin)

- **ahmedabdelhmed/notebook94c79d7292** — *Bayesian Heuristics with Collision-Avoidance for Pokémon TCG* `[Untitled Notebook]`
  - Self-contained heuristic play-agent: Bayesian deck-tracking belief + multi-objective reward shaping `R = α(Dmg/Energy) + β(HP) − γ(Waste)` + collision-avoidance retreat, Lightning Aggro archetype. `agent(obs, config)` entrypoint. 28-D float16 state vector.

## Similar play-agents in this mirror (Simulation division)

| Folder | Kaggle ref | Votes | Method | Status |
|--------|-----------|-------|--------|--------|
| avikdas567-heuristic-agent | [avikdas567/ptcg-ai-battle-heuristic-agent-data-pipeline](https://www.kaggle.com/code/avikdas567/ptcg-ai-battle-heuristic-agent-data-pipeline) | 23 | Heuristic agent + data pipeline | ADDED |
| lucifer19-battlecore-agent | [lucifer19/battlecore-compact-agent](https://www.kaggle.com/code/lucifer19/battlecore-compact-agent) | 22 | Payload-wrapped agent + falsification/validation protocol (no plain-text agent() source) | ADDED |
| romanrozen-strong-start-agent | [romanrozen/strong-start-baseline-agent-v10-lb-950](https://www.kaggle.com/code/romanrozen/strong-start-baseline-agent-v10-lb-950) | 142 | Probabilistic Expectimax baseline agent (LB 950+) | ADDED |
| makimakiai-tiny-rl-baseline | [makimakiai/ptcg-tiny-rl-to-submission-baseline-guide](https://www.kaggle.com/code/makimakiai/ptcg-tiny-rl-to-submission-baseline-guide) | 49 | Beginner RL -> submission baseline guide | ADDED |
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

## Honesty flags

- `lucifer19-battlecore-agent`: real agent, but its `.ipynb` embeds the agent as a compressed+base64 payload with an integrity gate — **no plain-text `agent()` in source**. No `source_with_linenumbers.py` export (would be the payload blob). Kept and flagged, not dropped.
- `kaiwalyaatulraut/pokemon-ai-battle-challenge-simulation-solution` (20 votes) was a candidate but **403 Forbidden** on pull — private/disabled for download. NOT included (no fabrication).
- `seokjeongeum/max-elo-1208-libraryout-w-crustle-great-tusk` (author-claimed LB 1208, sourced from the yakumsi vault's `Public Agent Roster — Makimakiai Matchup Matrix`) was a candidate but is **no longer reachable** — 403 Forbidden on the Kaggle API pull *and* "We can't find that page" on a direct browser visit (checked 2026-08-01). Deleted, made private, or re-slugged since the vault note captured it two days prior. NOT included (no fabrication); the requested teardown of this notebook could not be produced.
- `mechi22-alakazam`: `main.py` is embedded as a **base64 blob, SHA256-integrity-checked, not encrypted** — decoded locally (`notebooks/reference/mechi22-alakazam/main_decoded.py`) to a plain-text `agent()` with no red flags (no eval/exec/network/subprocess). Unlike `lucifer19-battlecore-agent`, full source recovery was possible here, so it's marked ADDED rather than excluded. Obfuscation appears to be an anti-fork measure, not a security concern.
- `dovhan`, `aristophanivan`, `kiyotah-*` were already present before this crawl (pre-existing mirror).
- Live leaderboard check (browser-rendered, 2026-08-01): #1 = 1298.5, top-8 cutoff = 1141.0, top-50 ≈ 1055.7 — all higher than the vault note's 2026-08-01 capture (1280.8 / 1122.0 / 1050.7), and no visible team in the top 49 matches the claimed 1208 score under any recognizable name. Ladder is live and moving; treat any stated score as time-stamped, not current.

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
