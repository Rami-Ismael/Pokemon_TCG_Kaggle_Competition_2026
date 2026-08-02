# jek1wantaufik — "Pokémon TCG AI Strategy Analysis"

Kaggle ref: https://www.kaggle.com/code/jek1wantaufik/pok-mon-tcg-ai-strategy-analysis

## Key finding: the notebook is analysis-only

`pok-mon-tcg-ai-strategy-analysis.ipynb` contains **no runnable agent**. Its 13
code cells:

1. load the EN/JP card CSVs,
2. `pickle.load` a `deck.pkl` and tabulate its composition/roles,
3. `ast.parse` a `main.py` and report *static* metrics only — counts of
   functions, classes, `if`/`for`/`while`, unique call names, "heuristic
   categories", state-feature coverage — then
4. emit a prose `strategy_report.md`.

Both artifacts are read by absolute path from a private-looking Kaggle **model**:

    /kaggle/input/models/jek1wantaufik/buddy/other/pokemon/1/{main.py, deck.pkl}

So the playable agent is never printed in the notebook. "Convert this notebook
into an agent" therefore means: recover `main.py`/`deck.pkl` from that model and
wire *those* into the benchmark — which is what was done.

## Recovery

The model `jek1wantaufik/buddy` is publicly downloadable:

    uv run kaggle models instances versions download jek1wantaufik/buddy/other/pokemon/1

The tarball ships `main.py`, `deck.pkl`, and a copy of the `cg/` engine (already
in this repo). `deck.pkl` decodes to a 60-card Mega Lucario ex list
(counts: 673×2 674×2 675×2 676×3 677×3 678×4 1102×4 1123×2 1141×4 1142×4
1152×4 1159×1 1182×2 1192×4 1227×4 1252×2 6×13) — mirrored verbatim into
`agents/jek1wantaufik_lucario/deck.csv`.

## Safety review (passed)

`main.py` imports only `os`, `pickle`, `collections`, and `cg.api`. No network,
no `eval`/`exec`, no `subprocess`, no file writes, no obfuscation. The only I/O
is reading its own `deck.pkl`. Wired in as `agents/jek1wantaufik_lucario/`.

## What the agent is

Pure hand-written heuristic, **no search / no learning**. Commits to one
`AttackPlan` (attacker, target, attack_index, remain_hp, energy) per MAIN turn,
then scores every legal option so attach/switch/retreat/gust/evolve all serve
that plan; returns options best-first, truncated to `select.maxCount`. Same
archetype as `rule_baseline` / `dedquoc_rule_engine` but an independent
implementation — distinct scoring constants (prize×1000, energies×150,
lethal=50000, no water-/Crustle-matchup special-casing), so it earns its own
slot in the pool rather than duplicating an existing one.

The `agent()` body in `agents/jek1wantaufik_lucario/agent_core.py` is
byte-for-byte the recovered `main.py`; the only change is the deck-loading
preamble (deck.csv-first, never crashes on import) — see that file's docstring.
