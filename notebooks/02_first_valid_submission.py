import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import tarfile
    from pathlib import Path

    return Path, mo, sys, tarfile


@app.cell
def _(mo):
    mo.md(r"""
    # [Beginner Guide → marimo] From Deck List to First Valid Submission

    Marimo adaptation of ichigoe's
    [Beginner Guide: From Deck List to First Valid Sub](https://www.kaggle.com/code/ichigoe/beginner-guide-from-deck-list-to-first-valid-sub),
    itself based on kiyotah's
    [Sample Rule-Based Agent: Mega Lucario ex](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck/notebook).

    **Goal (unchanged): not the strongest agent — a valid `submission.tar.gz` and
    understanding of every file involved.**

    The original notebook builds the bundle in the Kaggle cloud
    (`/kaggle/input` → `/kaggle/working`). This version is adapted to the
    **local repo workflow**:

    - `cg/` comes from the `kiyotah/cg-lib` Kaggle dataset (pulled to
      `data/external/cg-lib/`), plus the macOS dynamic libs copied from the
      project venv so `cg.api` imports natively on this machine
    - the agent runs **locally** via `kaggle_environments.make("cabt")`
      self-play — verified end-to-end (110 steps, rewards ±1)
    - `submission.tar.gz` is written to `submissions/mega_lucario/` and
      uploaded with the Kaggle CLI (`uv run kaggle`)

    Run with: `uv run marimo edit notebooks/02_first_valid_submission.py`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The deck: 60 Card IDs

    `deck.csv` is just a 60-line list of Card IDs (not names). The agent returns
    this list when the engine first asks for the deck (`obs.select is None`).

    The Mega Lucario ex deck below is hardcoded (easiest first-submission path).
    Card metadata (name, stage, ex flags) is fetched from the engine's
    `all_card_data()` so you can read what each ID actually is.
    """)
    return


@app.cell
def _(Path, sys):
    PROJECT_ROOT = Path.cwd()
    # repo root when run from the project dir via `uv run marimo`
    if not (PROJECT_ROOT / "src" / "pokemon_tcg").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent

    # cg/ ships in the kiyotah/cg-lib Kaggle dataset (pulled to
    # data/external/cg-lib), plus libcg.dylib / libcg-arm64.so copied from the
    # venv for macOS. In the Linux eval sandbox sim.py picks libcg.so instead.
    CG_LIB_DIR = PROJECT_ROOT / "data" / "external" / "cg-lib"
    sys.path.insert(0, str(CG_LIB_DIR))

    from cg.api import all_card_data

    all_card = all_card_data()
    card_table = {c.cardId: c for c in all_card}
    print(f"card metadata: {len(card_table)} cards")
    return CG_LIB_DIR, PROJECT_ROOT, card_table


@app.cell
def _():
    HARD_CODED_DECK = [
        673, 673, 674, 674, 675, 675, 676, 676,
        676, 677, 677, 677, 678, 678, 678, 678,
        1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141,
        1141, 1141, 1142, 1142, 1142, 1142, 1152, 1152,
        1152, 1152, 1159, 1182, 1182, 1192, 1192, 1192,
        1192, 1227, 1227, 1227, 1227, 1252, 1252, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6,
    ]

    assert len(HARD_CODED_DECK) == 60, (
        f"deck must be 60 cards, got {len(HARD_CODED_DECK)}"
    )
    return (HARD_CODED_DECK,)


@app.cell
def _(HARD_CODED_DECK, card_table, mo):
    from collections import Counter as _Counter

    counts = _Counter(HARD_CODED_DECK)
    rows = [
        {"ID": cid, "×": n, "Name": card_table[cid].name}
        for cid, n in sorted(counts.items(), key=lambda kv: card_table[kv[0]].name)
    ]
    mo.ui.table(rows, selection=None)
    return


@app.cell
def _(HARD_CODED_DECK, PROJECT_ROOT):
    OUT_DIR = PROJECT_ROOT / "submissions" / "mega_lucario"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    deck_csv_path = OUT_DIR / "deck.csv"
    deck_csv_path.write_text("".join(f"{cid}\n" for cid in HARD_CODED_DECK))
    print(f"deck.csv → {deck_csv_path} ({len(HARD_CODED_DECK)} lines)")
    return OUT_DIR, deck_csv_path


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The agent: `main.py` + `agent_core.py`

    The rule-based policy is unchanged from the sample notebook (switching
    between Mega Lucario ex as main attacker and Hariyama/Solrock as
    secondary). Keep two kinds of hardcoding straight:

    1. **The actual 60-card deck** — read from `deck.csv` at game start
       (`my_deck`)
    2. **Card-ID constants used by the rule policy** (`Mega_Lucario_ex = 678`,
       …) — these only score options

    The importable policy already lives in the repo at
    `agents/mega_lucario/agent_core.py` (extracted verbatim from the sample
    notebook's `%%writefile main.py` cell, minus the module-level deck read).
    `main.py` is a thin wrapper: load `deck.csv`, set `agent_core.my_deck`,
    expose `agent`. Both files go into the bundle.
    """)
    return


@app.cell
def _(HARD_CODED_DECK, PROJECT_ROOT, sys):
    sys.path.insert(0, str(PROJECT_ROOT / "agents" / "mega_lucario"))
    import agent_core

    agent_core.my_deck = HARD_CODED_DECK
    print("agent_core loaded:", agent_core.__file__)
    return (agent_core,)


@app.cell
def _(OUT_DIR, PROJECT_ROOT):
    main_py = (
        "import os\n"
        "import sys\n"
        "\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
        "\n"
        "import agent_core\n"
        "\n"
        'file_path = "deck.csv"\n'
        "if not os.path.exists(file_path):\n"
        '    file_path = "/kaggle_simulations/agent/" + file_path\n'
        'with open(file_path, "r") as f:\n'
        "    agent_core.my_deck = [\n"
        "        int(x) for x in f.read().splitlines() if x.strip()\n"
        "    ][:60]\n"
        "\n"
        "agent = agent_core.agent\n"
    )
    main_py_path = OUT_DIR / "main.py"
    main_py_path.write_text(main_py)

    agent_core_src = (PROJECT_ROOT / "agents" / "mega_lucario" / "agent_core.py").read_text()
    (OUT_DIR / "agent_core.py").write_text(agent_core_src)
    print(f"main.py → {main_py_path}")
    print(f"agent_core.py → {OUT_DIR / 'agent_core.py'} ({len(agent_core_src)} bytes)")
    return (main_py_path,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Sanity check: local self-play

    The original guide validates by *submitting*. Locally we can do better
    before burning one of the 5 daily submissions: run the agent against
    itself with `kaggle_environments.make("cabt")`. A completed game with
    rewards ±1 means the deck is legal and the policy returns valid option
    indices (any illegal index raises and forfeits the game).
    """)
    return


@app.cell
def _(agent_core):
    from kaggle_environments import make

    env = make("cabt")
    trace = env.run([agent_core.agent, agent_core.agent])
    final = trace[-1]
    print(f"steps: {len(trace)}")
    print(f"rewards: {[final[i]['reward'] for i in range(2)]}")
    print(f"statuses: {[final[i]['status'] for i in range(2)]}")
    return env, final


@app.cell
def _(final, mo):
    assert all(final[i]["status"] == "DONE" for i in range(2)), "game did not finish"
    assert sorted(final[i]["reward"] for i in range(2)) == [-1, 1], "bad rewards"
    mo.md("✅ **Self-play OK** — the agent plays a full legal game. The bundle is safe to submit.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Build `submission.tar.gz`

    Bundle layout (top-level `main.py`, NOT nested — same rule as the
    original guide):

    ```text
    submission.tar.gz
    ├── main.py
    ├── agent_core.py
    ├── deck.csv
    └── cg/
    ```

    `cg/` comes from `data/external/cg-lib` (the `kiyotah/cg-lib` dataset —
    the same `cg` the original guide finds under `/kaggle/input/**/cg-lib/cg`).
    Its `libcg.dylib` / `libcg-arm64.so` are harmless extras in the Linux eval
    sandbox (the loader picks `libcg.so` there).
    """)
    return


@app.cell
def _(CG_LIB_DIR, OUT_DIR, deck_csv_path, main_py_path, tarfile):
    tar_path = OUT_DIR / "submission.tar.gz"
    with tarfile.open(tar_path, "w:gz") as _tar:
        _tar.add(main_py_path, arcname="main.py")
        _tar.add(OUT_DIR / "agent_core.py", arcname="agent_core.py")
        _tar.add(deck_csv_path, arcname="deck.csv")
        _tar.add(CG_LIB_DIR / "cg", arcname="cg")

    print(f"created: {tar_path} ({tar_path.stat().st_size / 2**20:.2f} MiB)")
    return (tar_path,)


@app.cell
def _(mo, tar_path, tarfile):
    with tarfile.open(tar_path, "r:gz") as _tar:
        names = _tar.getnames()

    # main.py must be top-level
    assert "main.py" in names, "main.py must sit at the top level of the archive"
    mo.ui.table({"file": names})
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Verify the bundle actually runs (strongest pre-submission check)

    Unpack the tar into a scratch dir and play a game with **only** the
    bundled files on `sys.path` — this reproduces the eval sandbox's view of
    the world (minus Linux vs macOS, which `cg/sim.py` handles via its
    platform-specific lib loader).
    """)
    return


@app.cell
def _(OUT_DIR, Path, env, sys, tar_path, tarfile):
    import shutil
    import tempfile
    import os

    scratch = Path(tempfile.mkdtemp(prefix="subcheck_"))

    with tarfile.open(tar_path, "r:gz") as _tar:
        _tar.extractall(scratch)

    # fresh module view: only the bundle on sys.path
    for mod in ["agent_core", "cg", "cg.api", "cg.sim", "cg.game", "cg.utils"]:
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(scratch))

    import agent_core as bundled_core

    main_src = (scratch / "main.py").read_text()
    # run main.py with cwd = scratch so its relative "deck.csv" resolves inside
    # the bundle (mirrors the eval sandbox, where the bundle is the cwd)
    _cwd = os.getcwd()
    os.chdir(scratch)
    try:
        exec(compile(main_src, "main.py", "exec"), {"__file__": str(scratch / "main.py")})
    finally:
        os.chdir(_cwd)
    print("bundled deck:", bundled_core.my_deck[:5], "... len", len(bundled_core.my_deck))

    trace2 = env.run([bundled_core.agent, bundled_core.agent])
    final2 = trace2[-1]
    assert all(final2[i]["status"] == "DONE" for i in range(2))
    print(
        f"bundled self-play: {len(trace2)} steps, "
        f"rewards {[final2[i]['reward'] for i in range(2)]}"
    )

    shutil.rmtree(scratch)
    print(f"scratch cleaned (OUT_DIR kept: {OUT_DIR})")
    return


@app.cell
def _(mo):
    mo.md("""
    ✅ **Bundle verified** — the exact files inside `submission.tar.gz` play a full game.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Upload to the ladder

    Equivalent of the original guide's "Save Version → check Output → Submit",
    from the CLI:

    ```bash
    uv run kaggle competitions submit -c pokemon-tcg-ai-battle \
        -f submissions/mega_lucario/submission.tar.gz \
        -m "Mega Lucario ex rule-based agent (first valid sub)"
    ```

    Then watch the validation episode (your agent vs a copy of itself) on the
    competition **My Submissions** page. If it errors, download the agent
    logs — the local self-play check above means an error there would most
    likely be environment/packaging, not policy logic.

    Not run from this notebook on purpose: **only 5 submissions/day**, and
    only the latest 2 stay active. Submit deliberately.
    """)
    return


@app.cell
def _(mo, tar_path):
    mo.md(f"""
    **Ready:** `{tar_path}` — {tar_path.stat().st_size / 2**20:.1f} MiB "
        "(cap ≈ 197.7 MiB). Run the `kaggle competitions submit` command above when ready.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary (unchanged from the original)

    1. Hardcode (or read) the 60-card deck → write `deck.csv`
    2. Write `main.py` (+ `agent_core.py` here, since the policy is factored out)
    3. Package `main.py`, `agent_core.py`, `deck.csv`, `cg/` →
       `submission.tar.gz`, top-level `main.py`
    4. Inspect the archive; locally: unpack-and-replay as the ultimate check
    5. Submit deliberately — first a valid submission, then improve
       deck/policy/MCTS

    **Next steps from here:** swap the rule policy for a learned one trained
    on `data/episodes/splits/` (behavior cloning), or wire
    `search_begin_input` for MCTS.
    """)
    return


if __name__ == "__main__":
    app.run()
