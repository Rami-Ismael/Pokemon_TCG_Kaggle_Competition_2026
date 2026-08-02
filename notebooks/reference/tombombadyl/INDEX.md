# notebooks/reference/tombombadyl — TomBombadyl rule-based agent pool

Source: [github.com/TomBombadyl/kaggle_pokemon](https://github.com/TomBombadyl/kaggle_pokemon)
— a solo competitor's public workspace for **this exact competition**, with 21 decoded
ladder submissions and their real public μ scores. Per his `eval/AGENT_CATALOG_FULL.md`,
every submission is a **`brain × deck`** packaged for Kaggle `main.agent(obs)`.

We ported his **rule-based** brains (Rules ✓, RL ✗) into our benchmark pool as opponents
to test our trained model against and to compete against later
(checklist: *"Add all the rule base algorithm to my internal ladder to compare to my
trained model"*). His RL+MCTS and Track-B *learned* brains were **excluded** — they are
not rule-based and need model weights he deleted from the repo (Session-44 graveyard).

## What was ported (10 agents, `tb_*`)

All μ below are **his real Kaggle public leaderboard scores** (his catalog, UI 2026-06-26),
not ours. They are the reason these agents are worth having: a preview of what hand-written
rules score on the live ladder.

### Six self-contained per-deck rule pilots

| Pool name | His source | Deck (carried) | His μ | Notes |
|-----------|-----------|----------------|------:|-------|
| `tb_archaludon` | `agent/archaludon_agent.py` | `archaludon_ex_cinderace.csv` | **1196.1** | His **ladder leader**. Community v5 + his R7 empty-bench guard. New archetype for our pool. |
| `tb_dragapult`  | `agent/dragapult_agent.py`  | `dragapult_ex_sample.csv`     | 880.9 | Official kiyotah Dragapult sample + never-crash + his R7 bench guard. Overlaps upstream with our `kiyotah_dragapult` but behaviourally distinct (guard + his list). |
| `tb_alakazam`   | `agent/alakazam_agent.py`   | `ryotasueyoshi_alakazam_best5.csv` | 659.0 | ryotasueyoshi best5 rules + dragapult matchup levers. Same upstream as our `ryotasueyoshi_alakazam`. |
| `tb_starmie`    | `agent/starmie_agent.py`    | `starmie_froslass_ashleysandlin.csv` | 277.5 | PrizeTracker + finish search + R7 bench guard. New archetype. |
| `tb_abomasnow`  | `agent/abomasnow_agent.py`  | `real_mega_abomasnow_ex.csv`  | n/a   | Not on his ladder. Overlaps upstream with `kiyotah_abomasnow`. |
| `tb_iono`       | `agent/iono_agent.py`       | `real_iono.csv`               | n/a   | Not on his ladder. Overlaps upstream with `kiyotah_iono`. |

### Four deck-agnostic scorer brains (`build_agent(scorer=…)`)

| Pool name | Scorer (his `--scorer` flag) | Deck (carried) | His μ | Notes |
|-----------|------------------------------|----------------|------:|-------|
| `tb_search`   | `SearchScorer` (`search`)     | `real_mega_lucario_ex.csv` | **660.5** | **His best home-grown brain.** Generic hand-rules + shallow cg `SearchBegin` (~200 ms) on promote/switch/setup, falling back to `HeuristicScorer`. |
| `tb_heuristic`| `HeuristicScorer` (`heuristic`)| `real_dragapult_ex.csv`   | 633.0* | Hand-tuned MAIN priorities, no search. *His 633.0 was on `a2_kyogre_33_energy` — **not in his repo** (tarball-only), so we pair the deck-agnostic brain with `real_dragapult_ex` for deck diversity. |
| `tb_rulecore` | `RuleCoreScorer` (`rulecore`) | `real_mega_lucario_ex.csv` | 535.6 | Deck-agnostic attack plan from engine tables + `deck_tech`, falling back to `HeuristicScorer`. |
| `tb_lucario`  | `LucarioScorer` (`lucario`)   | `real_mega_lucario_ex.csv` | ~500–535 | kiyotah Lucario sample port + smart bench + `matchup_levers`. |

## How they are wired (porting method)

His two agent styles are packaged two different ways upstream, so they are shimmed two ways:

- **Pilots** are his standalone tarballs (`scripts/package_{archaludon,starmie,dragapult,
  alakazam}.py`): `main.py` → `<name>_agent.py` + bench-guard siblings + `deck.csv` + `cg/`.
  Each `agents/tb_<pilot>/agent_core.py` is a **thin shim** that (a) sets his own
  `<NAME>_DECK` env override — which his `_resolve_deck_path()` checks *first* — to this
  dir's `deck.csv` (our harness runs from the repo root, which ships a decoy `deck.csv`),
  (b) puts `tb_shared` on `sys.path`, then (c) re-exports `agent` / `my_deck` from
  `agent.<name>_agent`. **His source is byte-verbatim** in `tb_shared`; nothing edited.

- **Scorer brains** are his `scripts/package_submission.py --scorer <flag>` bundles, which
  copy the whole `agent/` package and generate a `main.py` doing `build_agent(scorer=…)`.
  Each `agents/tb_<scorer>/agent_core.py` **mirrors that generated `main.py`**: boot
  `tb_shared`, resolve deck next to `__file__`, `build_agent(seed=0, deck_path=_DECK,
  scorer=…)`, and `agent(obs)` returns the deck when `select is None` else `_AGENT.act(obs)`.

His entire `agent/` package (32 `.py`, 448 KB, no model artifacts) is vendored **once** at
[`agents/tb_shared/agent/`](../../../agents/tb_shared/agent). All ten `tb_*` agents share it.

## Safety review

Every ported file was scanned before wiring in (same bar as the rest of
[`notebooks/reference/INDEX.md`](../INDEX.md)): **no `eval` / `exec` / `subprocess` /
`socket` / `urllib` / `requests` / `__import__` / `os.system`.** The only `open()` calls
read each agent's own `deck.csv`. The scorer closure (`agent.agent`, `agent.search_policy`,
`agent.rule_core`, `agent.lucario_policy` + transitive imports) pulls in **no torch and no
deleted modules** — verified by importing and `build_agent`-ing all four against our arm64
`cg`. His `lucario_mcts_*` / `learned_policy` files are present in `tb_shared` but never
imported by any `tb_*` agent.

## Caveats (repo standing rules #3, #5)

- **Local head-to-head is not the ladder.** Our `agent_core_improved` beat `tb_archaludon`
  5–1 *locally* at n=6 — this says nothing about the real leaderboard, which has diverged
  from local ranking before. Trust his μ column (real ladder) over any local win-rate here.
- **His μ are time-stamped 2026-06-26**, from his catalog, on his decks. They are evidence
  about *his* submissions, not a guarantee for these ports.
- `data/external/cg-lib` (the arm64 `cg` engine the harness imports) is an untracked local
  dependency, staged from the main checkout — not part of this change.

## Rebuild / refresh from upstream

Cloned shallow to a scratch dir; ported with the vendor-once + shim method above. To refresh:
re-clone `github.com/TomBombadyl/kaggle_pokemon`, re-copy `agent/` → `agents/tb_shared/agent`
(drop `__pycache__` and his stray `agent/deck.csv`), and re-copy the decks from his
`agent_decks/`. The shims and wrappers do not change.
