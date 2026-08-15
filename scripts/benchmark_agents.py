"""Agent performance comparison benchmark for the Pokémon TCG AI Battle Challenge.

Runs a round-robin tournament between the agent implementations we have and
reports head-to-head win rates, a win matrix, and Glicko-1 ratings.

Win-rate alone is a noisy, history-blind estimate: it weighs every game
equally and forgets everything once you rerun the script. Glicko-1 (see
scripts/glicko1.py) instead tracks a rating + rating deviation (RD) per
agent, persisted in reports/glicko_ratings.json across runs, so ratings
compound over the agent's *full* battle history and carry a confidence
interval that narrows as more evidence comes in. GXE (also from
scripts/glicko1.py) folds rating + RD into a single win-probability-%
against a fixed average opponent -- the same pair of stats Pokemon
Showdown itself reports, and what the Metamon paper cites for its own
agent evaluations.

WHY THIS EXISTS
---------------
We now have several "our" algorithms in the repo:
  * rule_baseline        -- the sample rule-based Mega Lucario ex agent
                            (agents/mega_lucario/agent_core.py)
  * improved_prob_main   -- the earlier cleaned reimplementation
                            (agents/improved_probabilistic/main.py)
  * agent_core_improved  -- the faithful, engine-search-verified reimplementation
                            (agents/mega_lucario/agent_core_improved.py)
  * proto                -- the prototype search agent (scripts/_proto_agent.py)
  * grunt                -- deck-agnostic greedy one-ply agent: always takes
                            the attack with the highest type-chart-adjusted
                            damage, and switches to the best type matchup
                            when forced (agents/grunt/agent_core.py)

This script pits them against each other (and against themselves) so we can
see whether the search re-ranker actually buys win rate over the pure rule
baseline, and whether the two "improved probabilistic" variants differ.

THE RUNG-2 OPPONENT POOL (Q25)
------------------------------
Beating our own baselines was never evidence about the ladder; a round-robin
against a curated field of *public* agents is. data/opponent_pool.csv is that
field -- makimakiai's public roster of 65 public + 4 official sample agents (69
total), harvested by scripts/build_opponent_pool.py. It is a *registry*: most
rows are reference-only (a notebook URL we have not mirrored). The subset with a
`local_agent` value has been pulled, safety-reviewed, and wired as a real agent
here -- those form the `rung2` group below. Run `--list-pool` to see the roster
and how much of it is actually runnable locally.

Named groups usable anywhere `--agents` is: `ours`, `rung2`, `floor`, `anchors`,
`heuristics`, `all`.

WHAT COUNTS AS A RESULT (Rami, 2026-08-06)
------------------------------------------
`ours` means our TRAINED agents -- policies driven by a checkpoint. Our
hand-written heuristics (`rule_baseline`, `improved_prob_main`,
`agent_core_improved`) are no longer part of it and must not be reported as
opponents our models beat; that comparison is not what we are trying to learn.
They stay registered only as ladder ANCHORS: `agent_core_improved` has a real
ladder score we verified ourselves, which is what lets a local Glicko number be
read against the ladder at all. Report trained-vs-trained and trained-vs-`rung2`;
let anchors do their calibration job silently (they print marked "(anchor)").

USAGE
-----
    uv run python scripts/benchmark_agents.py            # default 8 games/pair
    uv run python scripts/benchmark_agents.py --games 12
    uv run python scripts/benchmark_agents.py --agents ours,rung2   # our models vs the public field
    uv run python scripts/benchmark_agents.py --agents ours,rung2,anchors  # + silent ladder calibration
    uv run python scripts/benchmark_agents.py --list-pool           # show the roster, then exit

    # Deck-arm sweep: same il_agent policy/weights, different submitted deck
    # (configs/deck_lists/<tag>.csv), isolated from the standing Glicko file:
    uv run python scripts/benchmark_agents.py \
        --agents il_agent@dragapult_ex,il_agent@alakazam,kiyotah_dragapult \
        --no-glicko-persist --out reports/deck_selection_benchmark.json

Output: a win-rate table (rows = player A, cols = player B) and per-agent
overall win rate, printed to stdout and saved next to the script.

Note: each match is 2 games (seat A=p0,p1 then A=p1,p0) so first-player bias
is cancelled. With --games N you get N such mirrored pairs per ordered pair.

NO COMMON RANDOM SEEDS. Checked against the actual engine: `cabt.json`'s
configuration schema exposes no seed field, `kaggle_environments.make()`'s
generic `configuration` dict has no seed convention this env reads, and the
Python-side `cg.game.battle_start(deck0, deck1)` call takes no seed
parameter -- deck shuffling and opening hands are decided inside the
compiled native engine (`libcg-arm64.so` / `cg.dll` / `libcg.dylib`), which
exposes no RNG control from Python. Different runs of the same pairing see
different shuffles/hands; there is no way to hold that constant with the
harness as given. Recorded here so this isn't silently assumed later.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glicko1  # noqa: E402

GLICKO_PATH = REPO / "reports" / "glicko_ratings.json"
# The Rung-2 opponent-pool registry (scripts/build_opponent_pool.py). Source of
# truth for which public agents are in the field and which are runnable locally.
OPPONENT_POOL_CSV = REPO / "data" / "opponent_pool.csv"

# Make `cg` importable (packaged engine) before importing any agent module.
# Agents import `from cg.api import ...`, so we need a cg package that ships
# api.py (the agent-facing wrapper) *and* a native lib for this platform. The
# canonical copy is data/external/cg-lib (its libcg.dylib is the arm64 build).
# When this script runs from a git worktree, that dir isn't checked out, so we
# also walk up parent directories to find the primary checkout's cg-lib. The
# other in-repo cg copies (sample-agent-output, submissions/*) ship only a
# Linux libcg.so and dlopen-crash on macOS, so they're intentionally not used.
# See memory `worktree-cg-lib-symlink`.
_cg_cands = [REPO / "data" / "external" / "cg-lib"]
_cg_cands += [p / "data" / "external" / "cg-lib" for p in REPO.parents]
_cg_cands.append(REPO / "agents" / "mega_lucario")
for _cand in _cg_cands:
    if (_cand / "cg" / "api.py").exists():
        sys.path.insert(0, str(_cand))
        break


# Registered but DELIBERATELY EXCLUDED from the `all` group and from Glicko
# persistence. These are diagnostics, not players: they exist to be pointed at
# a candidate on purpose (`--agents mycandidate,exploiter_regression`), never to
# be swept up by `--agents all`.
#
# Why exclusion is enforced rather than documented: `agent_groups()["all"]` is
# `list(AGENT_FILES)`, and Glicko COMPOUNDS across runs in
# reports/glicko_ratings.json -- so one accidental `--agents all` would inject a
# permanent rating for an agent that plays nothing like a real opponent and
# would drag every other agent's rating with it.
BENCHMARK_ONLY_AGENTS = {"exploiter_regression"}

AGENT_FILES = {
    "rule_baseline": REPO / "agents" / "mega_lucario" / "agent_core.py",
    "improved_prob_main": REPO / "agents" / "improved_probabilistic" / "main.py",
    "agent_core_improved": REPO / "agents" / "mega_lucario" / "agent_core_improved.py",
    "proto": REPO / "scripts" / "_proto_agent.py",
    "il_agent": REPO / "agents" / "il_agent" / "agent_core.py",
    # Same policy code + deck as il_agent, bundle-local model/ holding the
    # hfstream_v2 checkpoint (trained 2026-08-05 on all 9 Hub train days,
    # 17,622 eps, offline eval 0.7583). il_agent-vs-this is checkpoint-vs-
    # checkpoint, code and deck held fixed.
    "il_agent_v2": REPO / "agents" / "il_agent_v2" / "main.py",
    # v3 corpus-scaling arms (46 Hub days, 210,512 eps, 1 epoch/596k steps):
    # _final = annealed end (offline .7342), _best = step-520k peak (.7528).
    # Local verdict vs il_agent_v2: indistinguishable (v2 10-6 over _best,
    # 8-8 vs _final, n=16/pair) -- see reports/benchmark_v3_arms.json.
    "il_agent_v3_final": REPO / "agents" / "il_agent_v3_final" / "main.py",
    "il_agent_v3_best": REPO / "agents" / "il_agent_v3_best" / "main.py",
    # v4 = the complete 53-day hub corpus (52 train days, 260,065 eps, 1 epoch,
    # seed 42) -- adds 07-24 + 08-05..08-07 (the newest meta days) over v3.
    # Best checkpoint step 620k, offline .7619 (final .7606). Pool stars vs
    # il_agent_v3_best: reports/pool_star_*.json (2026-08-10).
    "bc_alldays52_jun16_aug07_seed42":
        REPO / "agents" / "bc_alldays52_jun16_aug07_seed42" / "main.py",
    # v6 step 2 (rl_pipeline_v6.md §1.1): binary-advantage weighted BC
    # (w = 1[outcome - V(s) > 0], calibrated outcome critic), 1 epoch over
    # train_combined_v4 (264,495 eps), init from bc_alldays52. Bundle-local
    # model/ + Teal Mask Ogerpon ex deck (same deck as the 55491496 and
    # 55478780 subs -> same-deck A/B vs bc_alldays52@ogerpon and the wbc arm).
    "binaryadv_alldays52_jun16_aug07_seed42":
        REPO / "agents" / "binaryadv_alldays52_jun16_aug07_seed42" / "main.py",
    # v6 step 3 mint-rule experiment arms (rl_pipeline_v6.md §1.2, card
    # notes/experiments/2026-08-13-lineage-minting-rule.md): one generation of
    # lineage-only OSFP self-play (1000 games, p_opt 0.5, both decks sampled)
    # + 0.25-epoch binary-advantage resume over human ∪ self-play, init from
    # binaryadv_alldays52. Identical configs; the ONLY difference is the mint
    # rule (tryout: 47W/53L, not minted / cadence: minted unconditionally).
    # Same Ogerpon bundle as binaryadv -> deck held fixed across all our arms.
    "lineage_selfplay_tryout_gen1_seed42":
        REPO / "agents" / "lineage_selfplay_tryout_gen1_seed42" / "main.py",
    "lineage_selfplay_cadence_gen1_seed42":
        REPO / "agents" / "lineage_selfplay_cadence_gen1_seed42" / "main.py",
    # IL-prior MCTS (search_prior_mcts.py): same IL checkpoint and
    # deck as il_agent, plus Search-API lookahead. Any il_agent-vs-this
    # comparison is therefore policy-vs-policy+search, deck held fixed.
    "mcts_il_agent": REPO / "agents" / "mcts_il_agent" / "agent_core.py",
    # A0-family search arm: agent_core_improved with USE_SEARCH/USE_BC_PRIOR
    # forced on and the PUCT prior served by models/il_alldays_0804 (strongest
    # BC clone by exploit-robustness evidence). Distinct name on purpose --
    # Glicko history compounds by name and this is a different policy from the
    # search-off `agent_core_improved` control.
    "search_prior_alldays": REPO / "agents" / "search_arms" / "prior_alldays_lucario" / "agent_core.py",
    # UCB1 depth-1 re-rank arm: the bc_alldays52 IL policy as the candidate
    # RANKER (not a PUCT bonus) feeding the repaired improved_probabilistic
    # bandit through its base_order seam; leaf is still evaluate_state (known
    # suspect). Control = bc_alldays52_jun16_aug07_seed42 (same checkpoint, no
    # search). Card: notes/experiments/2026-08-11-il-ucb1-depth1-rerank.md.
    "bc_alldays52_ucb1_rerank":
        REPO / "agents" / "search_arms" / "bc_alldays52_ucb1_rerank" / "agent_core.py",
    # Same bandit + IL ranker, but the leaf is the retrained outcome critic
    # (Platt-scaled + centered per its calibration.json) and overrides can be
    # margin-gated via OVERRIDE_MARGIN. Card:
    # notes/experiments/2026-08-12-critic-leaf-margin-gate.md.
    "bc_alldays52_ucb1_criticleaf":
        REPO / "agents" / "search_arms" / "bc_alldays52_ucb1_criticleaf" / "agent_core.py",
    # Standing regression opponent (BENCHMARK_ONLY_AGENTS). The PPO exploiter
    # trained against frozen il_agent. NEVER SUBMIT IT -- it loses to
    # rule_baseline (0.440) and improved_prob_main (0.230); it beats only its
    # training target and the random floor. Samples at T=1.0, so it is
    # stochastic: quote CIs. See agents/exploiter_regression/main.py.
    "exploiter_regression": REPO / "agents" / "exploiter_regression" / "main.py",
    # kojimar's "Simple Baseline + Matchup Tests" Mega Lucario ex, ported as a
    # bare module (literal DECK -> my_deck). Distinct 60-card list from
    # rule_baseline/mega_lucario; see agents/kojimar_lucario/agent_core.py.
    "kojimar_lucario": REPO / "agents" / "kojimar_lucario" / "agent_core.py",
    "grunt": REPO / "agents" / "grunt" / "agent_core.py",
    # Floor test, not a competitor: uniform-random legal moves on the same
    # frozen deck. If a trained policy doesn't beat this decisively, its
    # offline accuracy isn't credible evidence it learned anything.
    "random_legal": REPO / "agents" / "random_legal" / "agent_core.py",
    # Public opponent-pool vetted batch (closes the discovery-pass gap:
    # benchmarking only against our own agents was never evidence about
    # the ladder). Pulled from Kaggle's competition Code tab, individually
    # reviewed for safety before wiring in -- see notebooks/reference/INDEX.md.
    "kiyotah_dragapult": REPO / "agents" / "kiyotah_dragapult" / "agent_core.py",
    "kiyotah_iono": REPO / "agents" / "kiyotah_iono" / "agent_core.py",
    "kiyotah_abomasnow": REPO / "agents" / "kiyotah_abomasnow" / "agent_core.py",
    "dedquoc_rule_engine": REPO / "agents" / "dedquoc_rule_engine" / "agent_core.py",
    "ryotasueyoshi_alakazam": REPO / "agents" / "ryotasueyoshi_alakazam" / "agent_core.py",
    "makthanithin_improved_prob": REPO / "agents" / "makthanithin_improved_prob" / "agent_core.py",
    # jek1wantaufik's rule-based Mega Lucario ex agent, recovered from the
    # Kaggle model `jek1wantaufik/buddy` (main.py + deck.pkl) that its
    # "Strategy Analysis" notebook only statically analyses -- the notebook
    # itself ships no runnable agent(). Independent implementation of the
    # Lucario archetype (its own scoring constants), reviewed for safety
    # before wiring in -- see notebooks/reference/jek1wantaufik-strategy-analysis/NOTES.md.
    "jek1wantaufik_lucario": REPO / "agents" / "jek1wantaufik_lucario" / "agent_core.py",
    # mechi22's notebook ships main.py as a base64 blob (SHA256-verified, not
    # encrypted) to deter forking -- decoded to plain source for this repo's
    # review; see notebooks/reference/mechi22-alakazam/main_decoded.py.
    "mechi22_alakazam": REPO / "agents" / "mechi22_alakazam" / "agent_core.py",
    # Archetype-coverage recruitment for the opponent pool:
    # the pool was 9/14 agents on the exact frozen Mega Lucario ex deck before
    # this. Archaludon ex / Cinderace metal-tempo -- a genuinely different
    # archetype (Metal-type, not Fighting/Psychic/Dragon/Electric/Grass-Ice
    # like everything else in the pool).
    "plamen06_steel": REPO / "agents" / "plamen06_steel" / "agent_core.py",
    # Stage-2 REWEIGHT arms (rl_pipeline_v1.md §2.1): il_agent's inference core
    # pointed at each fine-tuned checkpoint via thin wrappers. One entry per
    # (arm, seed) -- distinct names on purpose, because Glicko history compounds
    # by name and these are genuinely different policies.
    "s2_e0_s42": REPO / "agents" / "s2_arms" / "e0_seed42" / "agent_core.py",
    "s2_e0_s43": REPO / "agents" / "s2_arms" / "e0_seed43" / "agent_core.py",
    "s2_e0_s44": REPO / "agents" / "s2_arms" / "e0_seed44" / "agent_core.py",
    "s2_e1_s42": REPO / "agents" / "s2_arms" / "e1_seed42" / "agent_core.py",
    "s2_e1_s43": REPO / "agents" / "s2_arms" / "e1_seed43" / "agent_core.py",
    "s2_e1_s44": REPO / "agents" / "s2_arms" / "e1_seed44" / "agent_core.py",
    "s2_e2_s42": REPO / "agents" / "s2_arms" / "e2_seed42" / "agent_core.py",
    "s2_e2_s43": REPO / "agents" / "s2_arms" / "e2_seed43" / "agent_core.py",
    "s2_e2_s44": REPO / "agents" / "s2_arms" / "e2_seed44" / "agent_core.py",
    # Rung-2 roster batch-pull (2026-08-02): pulled all 61 reference-only
    # roster notebooks, static-safety-scanned (capability audit: stdlib + cg
    # only, no network/subprocess/exec/eval/dynamic-import, read-only file I/O),
    # and wired the subset that adds a NEW deck (= archetype, the confound axis
    # here) or a distinct policy -- not raw count. Relative "deck.csv" reads
    # were made module-relative so they can't pick up the repo-root deck (same
    # fix as plamen06). Same-deck clones stay reference-only in
    # data/opponent_pool.csv. See notebooks/reference/INDEX.md.
    "prvsiyan_control_v11": REPO / "agents" / "prvsiyan_control_v11" / "agent_core.py",           # Fighting/Great Tusk/Crustle (new deck)
    "prvsiyan_grimbelief_alakazam": REPO / "agents" / "prvsiyan_grimbelief_alakazam" / "agent_core.py",  # Alakazam belief (new deck)
    "prvsiyan_templates_alakazam": REPO / "agents" / "prvsiyan_templates_alakazam" / "agent_core.py",    # Alakazam templates (same deck, diff policy)
    "pllinas_alakazam": REPO / "agents" / "pllinas_alakazam" / "agent_core.py",                   # Alakazam rising-tide (new deck)
    "biohack44_day2": REPO / "agents" / "biohack44_day2" / "agent_core.py",                       # new deck
    "pixiux_lucario_v63": REPO / "agents" / "pixiux_lucario_v63" / "agent_core.py",               # Mega Lucario variant (new deck)
    "makthanithin_1084_baseline": REPO / "agents" / "makthanithin_1084_baseline" / "agent_core.py",  # scored target LB 1084.5
    "daniilkrasnovvv_conservative_prob": REPO / "agents" / "daniilkrasnovvv_conservative_prob" / "agent_core.py",  # distinct conservative-probabilistic policy
    # Second wave of public-pool opponents (from origin/main's benchmark-pool
    # consolidation), wired to widen the pool's *strength* range so it predicts
    # the ladder, not just rank our own agents. Each was individually
    # source-reviewed; see the "Benchmark-wiring wave" section of
    # notebooks/reference/INDEX.md.
    #   romanrozen_strong_start -- Kaggle LB ~950 Probabilistic Expectimax with a
    #     UCB1/MCTS re-ranker over *real* cg engine search rollouts. Strongest
    #     public opponent in the pool. Byte-identical to aristophanivan's
    #     improved-probabilistic-agent (same lineage), so only one copy is wired.
    #     (This is the same romanrozen notebook my batch-pull held back as a
    #     redundant-deck clone; origin/main wired it for strength coverage, so it
    #     stays -- its opponent_pool.csv row now maps here as runnable.)
    "romanrozen_strong_start": REPO / "agents" / "romanrozen_strong_start" / "agent_core.py",
    #   avikdas567_heuristic -- weak: scores options by substring-matching
    #     str(option). Fills the tier between random_legal and the real policies.
    "avikdas567_heuristic": REPO / "agents" / "avikdas567_heuristic" / "agent_core.py",
    #   makimakiai_rl -- RL-tuned linear-weights policy over option types. The
    #     upstream notebook only ships the *training harness*; its LEARNED_WEIGHTS
    #     bake is nondeterministic and wasn't reproduced, so this runs on the
    #     notebook's DEFAULT (untrained) weights -- a legal, distinct-method
    #     opponent, but NOT the tuned agent. Labeled untrained in INDEX.md.
    "makimakiai_rl": REPO / "agents" / "makimakiai_rl" / "agent_core.py",
    # ------------------------------------------------------------------
    # TomBombadyl pool (github.com/TomBombadyl/kaggle_pokemon) -- a solo
    # competitor's decoded ladder submissions with real public mu scores.
    # His rule-based brains ported here as opponents to test our trained
    # model against (checklist: "Add all the rule base algorithm to my
    # internal ladder"). Each tb_* agent is a thin shim / wrapper over his
    # verbatim source, vendored once under agents/tb_shared/agent. Every
    # file was safety-reviewed (no eval/exec/network/subprocess) -- see
    # notebooks/reference/tombombadyl/INDEX.md for provenance, the mu each
    # scored on the real Kaggle ladder, and the deck each carries.
    #
    # Six self-contained per-deck rule pilots (his bench-guard-wrapped
    # standalone tarballs):
    "tb_archaludon": REPO / "agents" / "tb_archaludon" / "agent_core.py",  # mu 1196.1 (his leader)
    "tb_dragapult": REPO / "agents" / "tb_dragapult" / "agent_core.py",    # mu 880.9
    "tb_alakazam": REPO / "agents" / "tb_alakazam" / "agent_core.py",      # mu 659.0
    "tb_starmie": REPO / "agents" / "tb_starmie" / "agent_core.py",        # mu 277.5
    "tb_abomasnow": REPO / "agents" / "tb_abomasnow" / "agent_core.py",    # not on his ladder
    "tb_iono": REPO / "agents" / "tb_iono" / "agent_core.py",              # not on his ladder
    # Four deck-agnostic scorer brains (build_agent(scorer=...) over his
    # shared agent/ package; deck each is paired with per his catalog):
    "tb_search": REPO / "agents" / "tb_search" / "agent_core.py",          # mu 660.5 (his best home-grown)
    "tb_heuristic": REPO / "agents" / "tb_heuristic" / "agent_core.py",    # mu 633.0 archetype
    "tb_rulecore": REPO / "agents" / "tb_rulecore" / "agent_core.py",      # mu 535.6
    "tb_lucario": REPO / "agents" / "tb_lucario" / "agent_core.py",        # LucarioScorer
    # ------------------------------------------------------------------
    # wmh pool (github.com/wmh/ptcg-abc @ 7854fde, vendored 2026-08-03) --
    # a solo competitor's rule-based ladder agents, divergence-mined from
    # the era's ladder top (Majkel1337 #1, keidroid #1, Yushin #2) and
    # validated by real-ladder A/B (their repo's own lesson: local sims
    # mispredicted their rank, so they tuned on the ladder). Wired for the
    # NINE deck archetypes the pool had zero coverage for, plus three
    # ladder-scored stronger pilots of covered decks. Safety-reviewed
    # 2026-08-03: stdlib + cg.api only, no network/subprocess/exec/eval,
    # module-relative deck.csv reads; AGENT_FILES points at their verbatim
    # main.py (module-level `my_deck` + `agent`). The five Generic* agents
    # carry byte-identical vendored policy_base/generic_policy siblings
    # (identical hashes, so first-import module caching is behavior-safe).
    # Ladder Elo where their docs record one:
    "wmh_alakazam": REPO / "agents" / "wmh_alakazam" / "main.py",        # Alakazam v4.x, ladder ~860 lineage (their best-developed)
    "wmh_megastarmie": REPO / "agents" / "wmh_megastarmie" / "main.py",  # keidroid (#1, 1341) clone; their v3 scored 871.5
    "wmh_bellibolt": REPO / "agents" / "wmh_bellibolt" / "main.py",      # Iono's Bellibolt ex, ladder 836
    "wmh_garchomp": REPO / "agents" / "wmh_garchomp" / "main.py",        # NEW deck: Cynthia's Garchomp ex, ladder 713.8
    "wmh_typhlosion": REPO / "agents" / "wmh_typhlosion" / "main.py",    # NEW deck: Ethan's Typhlosion, ladder 532
    "wmh_trevenant": REPO / "agents" / "wmh_trevenant" / "main.py",      # NEW deck: Hop's Trevenant (bespoke policy)
    "wmh_mewtwo": REPO / "agents" / "wmh_mewtwo" / "main.py",            # NEW deck: Team Rocket's Mewtwo ex (bespoke policy)
    "wmh_grimmsnarl": REPO / "agents" / "wmh_grimmsnarl" / "main.py",    # NEW deck: Grimmsnarl ex (GenericPolicy, top-player list)
    "wmh_kangaskhan": REPO / "agents" / "wmh_kangaskhan" / "main.py",    # NEW deck: Kangaskhan ex (GenericPolicy, top-player list)
    "wmh_ogerpon": REPO / "agents" / "wmh_ogerpon" / "main.py",          # NEW deck: Ogerpon (GenericPolicy, top-player list)
    "wmh_chandelure": REPO / "agents" / "wmh_chandelure" / "main.py",    # NEW deck: Chandelure (GenericPolicy)
    "wmh_froslass": REPO / "agents" / "wmh_froslass" / "main.py",        # NEW deck: Mega Froslass ex (GenericPolicy)
    # Marnie's Grimmsnarl ex EXPERT, added 2026-08-05 to repair the pool's
    # inability to contest the corpus's #1 archetype (51.3% of ladder seats,
    # while our only 3 pilots ranked 39/44/46 of 52). Found by reading decks,
    # not names: scripts/scan_public_kernel_decks.py over the top 30 public
    # kernels turned up exactly ONE Grimmsnarl agent -- people publish Lucario
    # and play Grimmsnarl. Source: tetsutani/grimmsnarl-ex-damage-transfer-control
    # (89 votes). Shipped base64-tarball-wrapped; decoded, sha256-verified
    # against the notebook's own declared hash, and audited across its 178 .py
    # files: no subprocess/socket/network/exec/eval/ctypes, ZERO write-mode file
    # I/O, imports stdlib + cg + its own modules. Its one pickle
    # (models/feature_schema.pkl.gz) disassembles under pickletools to integer
    # opcodes only -- zero GLOBAL/REDUCE/INST/OBJ/BUILD -- so unpickling cannot
    # execute code. Deck legality PASS. Supplies its own deck (main.py:215).
    # Strength: 84.1% [75.0, 90.3] over 88 games; beats wmh_grimmsnarl (37.5%).
    "tetsutani_grimmsnarl": REPO / "agents" / "tetsutani_grimmsnarl" / "main.py",  # Marnie's Grimmsnarl ex expert
    # (The nine s2_* REWEIGHT arms are registered once, above -- they used to be
    # repeated here, which Python silently resolved to whichever block came
    # last. Add new arms in one place only.)
    # RESTARTED Stage-2 arms (rl_pipeline_v2.md §2.B2, 2026-08-04): advantage-
    # weighted BC on the restored streamed corpus, critic-first. e0=control,
    # e1=binary advantage, e2b{05,1,2}=exp advantage at beta 0.5/1/2, e3=
    # skill-gated best arm, efb=outcome-weighted fallback (only if the critic
    # audit fails). Wrappers exist before their checkpoints; loading one
    # without its checkpoint raises at import -- the desired loud failure.
    "s2v2_e0_s42": REPO / "agents" / "s2v2_arms" / "e0_s42" / "agent_core.py",
    "s2v2_e0_s43": REPO / "agents" / "s2v2_arms" / "e0_s43" / "agent_core.py",
    "s2v2_e0_s44": REPO / "agents" / "s2v2_arms" / "e0_s44" / "agent_core.py",
    "s2v2_e3_s42": REPO / "agents" / "s2v2_arms" / "e3_s42" / "agent_core.py",
    "s2v2_e3_s43": REPO / "agents" / "s2v2_arms" / "e3_s43" / "agent_core.py",
    "s2v2_e3_s44": REPO / "agents" / "s2v2_arms" / "e3_s44" / "agent_core.py",
    "s2v2_efb_s42": REPO / "agents" / "s2v2_arms" / "efb_s42" / "agent_core.py",
    "s2v2_efb_s43": REPO / "agents" / "s2v2_arms" / "efb_s43" / "agent_core.py",
    "s2v2_efb_s44": REPO / "agents" / "s2v2_arms" / "efb_s44" / "agent_core.py",
    # All-days imitation (plain-name scheme, 2026-08-04): fresh 3-epoch BC on
    # every Hub train day (15,032 episodes) -- Rami's data-scale experiment.
    "il_alldays_0804": REPO / "agents" / "il_alldays_0804" / "agent_core.py",
    # Self-play generation 1 (provisional base = all-days imitation): the
    # budget-end final policy and the anchor-gate-promoted teacher (beat its
    # frozen reference 73-27 at step 430k).
    "selfplay_g1_final": REPO / "agents" / "selfplay_g1_final" / "agent_core.py",
    "selfplay_g1_ref430k": REPO / "agents" / "selfplay_g1_ref430k" / "agent_core.py",
    "selfplay_g2_final": REPO / "agents" / "selfplay_g2_final" / "agent_core.py",
    "selfplay_g3_final": REPO / "agents" / "selfplay_g3_final" / "agent_core.py",
    # Stage-3 SELFPLAY candidates (rl_pipeline_v1.md §3.3): PufferLib-PPO
    # fine-tuned snapshots from models/ppo_puffer/, wrapped like s2_arms.
    "ppo_u60416": REPO / "agents" / "ppo_arms" / "u60416" / "agent_core.py",
    "ppo_u120832": REPO / "agents" / "ppo_arms" / "u120832" / "agent_core.py",
    "ppo_g2_u107520": REPO / "agents" / "ppo_arms" / "g2_u107520" / "agent_core.py",
    # 2x2 size-x-data BC grid (notes/adr_rl_objective_progression.md, Option B):
    # il_agent core pointed at each cell's checkpoint, all sharing il_agent's
    # deck. The 4th cell is il_agent itself (Small @ 07-26, the PRIOR).
    "grid_medium": REPO / "agents" / "grid_cells" / "medium_prior" / "agent_core.py",
    "grid_small_comb": REPO / "agents" / "grid_cells" / "small_combined" / "agent_core.py",
    # grid_medium_comb DEREGISTERED 2026-08-10: models/il_agent_medium_combined
    # has been an empty dir since 2026-08-03 (see the il_arms comment below), so
    # the arm never ran its model -- every game was _safe_choice. The strict
    # loader (PR #53) now refuses that instead of degrading, which crashed the
    # whole pool at load time. No weights exist locally or on the HF backup.
    # IL checkpoint sweep (reports/il_model_deck_selection.md): every DISTINCT
    # BC checkpoint in models/, one identical wrapper each so the only thing
    # varying across the model axis is the weights. Deduped by sha256 --
    # models/il_agent_3ep is byte-identical to models/il_agent, and
    # models/il_agent_winning_827.8 is byte-identical to il_agent_2ep_backup,
    # so each pair contributes ONE arm. models/il_agent_medium_combined is an
    # EMPTY directory (no config.json/safetensors), which is why the older
    # `grid_medium_comb` arm above silently falls back to non-ML behaviour --
    # it is deliberately not re-wired here.
    "il_bc_2ep": REPO / "agents" / "il_arms" / "il_bc_2ep" / "agent_core.py",
    "il_bc_3ep": REPO / "agents" / "il_arms" / "il_bc_3ep" / "agent_core.py",
    "il_bc_4ep": REPO / "agents" / "il_arms" / "il_bc_4ep" / "agent_core.py",
    "il_medium_3ep": REPO / "agents" / "il_arms" / "il_medium_3ep" / "agent_core.py",
    "il_small_comb_2ep": REPO / "agents" / "il_arms" / "il_small_comb_2ep" / "agent_core.py",
    "il_hfstream_comb_3ep": REPO / "agents" / "il_arms" / "il_hfstream_comb_3ep" / "agent_core.py",
    "il_alldays_3ep": REPO / "agents" / "il_arms" / "il_alldays_3ep" / "agent_core.py",
    # FINAL checkpoint (step 736,715) of the 1-epoch all-52-days BC run
    # (models/bc_alldays52_jun16_aug07_seed42). Deliberately NOT the
    # best-eval-at-620k snapshot (…_seed42_best) -- registered 2026-08-10 for
    # the deck-selection rerun on the stronger policy.
    "il_bc_alldays52_final": REPO / "agents" / "il_arms" / "il_bc_alldays52_final" / "agent_core.py",
    # Equal-steps control for il_alldays_3ep (standing rule 4: compare at equal
    # STEPS, not equal epochs). 38,562 steps vs 127,748; offline acc .7414 vs
    # .7583 but ECE .0124 -- the best calibration of any checkpoint here.
    # Trained by a concurrent session; lived only in that worktree.
    # il_alldays_equalsteps DEREGISTERED 2026-08-10: its checkpoint
    # (models/il_alldays_equalsteps_0804) lived in worktree
    # rewrite-kaggle-pokemon-tcg-prompt-c69997, which was deleted. Every
    # surviving reference is a dangling symlink and the HF backup repo
    # (Rami/ptcg-s2v2-arms) has no copy -- the weights are LOST. Its offline
    # numbers (acc .7414, ECE .0124) remain in the registry history above.
}

# Where each agent's real competition entry point (main.py) lives, if any.
# Loading via main.py guarantees the deck is wired up exactly as the harness
# would at submission time.
AGENT_MAIN = {
    "rule_baseline": REPO / "submissions" / "mega_lucario" / "main.py",
    "agent_core_improved": REPO / "submissions" / "mega_lucario_improved" / "main.py",
}

# Our own algorithms, split by whether a learned policy is in the loop.
#
# OUR_TRAINED is what `--agents ours` now means, and what results should be
# reported about: agents whose behaviour comes from a checkpoint we trained.
#
# OUR_HEURISTICS are hand-written policies with no ML in them at all
# (improved_prob_main in particular is pure rules -- see memory
# `search-bc-prior-no-measurable-gain`). Rami, 2026-08-06: do not report our
# trained agents against these. They are kept registered for ONE reason --
# `agent_core_improved` is one of the few agents whose real ladder score we
# read ourselves (685.3), so leaving it in a pool keeps local Glicko tied to
# a verified ladder reference point. That is an instrument-calibration role,
# not a rival. Get them with the explicit `heuristics` / `anchors` groups; they
# are no longer swept in by `ours`.
# Verified 2026-08-06 by grepping each entry point: of the standing agents only
# `il_agent`, `il_agent_v2` and `mcts_il_agent` load a checkpoint at all.
# `proto` (search + hand-written eval) and `grunt` (greedy MaxDamage one-ply)
# have no ML in them despite living in our tree, so they moved to
# OUR_HEURISTICS.
#
# `ours` is deliberately just the STANDING trained agents. The arm families
# (s2v2_*, il_*, selfplay_*, ppo_*, grid_*) are trained too but have always
# been named explicitly per experiment, and sweeping ~30 of them into a default
# group would silently change every benchmark's cost and its Glicko field. Name
# your arms; `ours` is the reference point you compare them to.
#
# Two things are deliberately in NEITHER list. `search_prior_alldays` is a
# hybrid -- hand-written eval, trained BC prior -- so it sits cleanly on
# neither side; it is an experiment arm and stays explicitly named.
# `makimakiai_rl` is trained but is somebody else's, so it belongs to `rung2`.
#
# Caveat when reading mcts_il_agent: it spends unbudgeted local think time,
# which flatters it relative to the ladder (see memory `anchored-pool-rho-0.93`).
OUR_TRAINED = ["il_agent", "il_agent_v2", "il_agent_v3_final", "il_agent_v3_best", "mcts_il_agent"]
OUR_HEURISTICS = ["rule_baseline", "improved_prob_main", "agent_core_improved",
                  "proto", "grunt"]

# Silent ladder anchors: included in a pool so ratings stay calibrated, marked
# "(anchor)" in the report, never quoted as a head-to-head result.
ANCHOR_AGENTS = ["agent_core_improved"]

# Everything authored/adopted here, as opposed to the public field pulled from
# Kaggle. Used to keep `rung2` meaning "the external field, not us".
OUR_AGENTS = OUR_TRAINED + OUR_HEURISTICS
FLOOR_AGENTS = ["random_legal"]


def is_non_learned(agent: str) -> bool:
    """True if `agent` is a hand-written policy, not one of our checkpoints.

    Handles the `name@deck` arm syntax. Used only to LABEL report rows, so a
    reader never mistakes an anchor's rating for a result about our models.
    """
    return agent.split("@", 1)[0] in set(OUR_HEURISTICS) | set(FLOOR_AGENTS)


def report_tag(agent: str) -> str:
    """Suffix marking why a non-learned row is in the table at all."""
    base = agent.split("@", 1)[0]
    if base in ANCHOR_AGENTS:
        return "  (anchor -- ladder reference, not a result)"
    if base in FLOOR_AGENTS:
        return "  (floor)"
    if base in OUR_HEURISTICS:
        return "  (hand-written, not a result)"
    return ""


def load_opponent_pool(path: Path = OPPONENT_POOL_CSV) -> list[dict]:
    """Return the Rung-2 roster rows from data/opponent_pool.csv (empty if absent).

    Each row: key, source_ref, url, label, local_agent, runnable. `local_agent`
    is the AGENT_FILES key when the notebook has been mirrored and wired here;
    otherwise the row is a reference-only watchlist entry we cannot run.
    """
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rung2_pool() -> list[str]:
    """AGENT_FILES keys for the runnable *public* opponents in the roster.

    Data-driven from opponent_pool.csv: every runnable row's local_agent that is
    (a) actually registered in AGENT_FILES and (b) not one of OUR_AGENTS -- the
    sample Mega Lucario, for instance, maps to our own `rule_baseline`, so it is
    excluded here to keep `rung2` meaning "the external field, not us".
    """
    seen: dict[str, None] = {}  # dict preserves insertion order, dedupes
    for row in load_opponent_pool():
        local = row.get("local_agent", "").strip()
        if (row.get("runnable", "").strip().lower() == "true"
                and local in AGENT_FILES and local not in OUR_AGENTS):
            seen.setdefault(local, None)
    return list(seen)


def agent_groups() -> dict[str, list[str]]:
    """Named agent selections usable anywhere `--agents` accepts a name."""
    return {
        # `ours` = our TRAINED agents only. Hand-written heuristics are opt-in
        # via `heuristics` / `anchors`; see OUR_TRAINED for why.
        "ours": [a for a in OUR_TRAINED if a in AGENT_FILES],
        "heuristics": [a for a in OUR_HEURISTICS if a in AGENT_FILES],
        "anchors": [a for a in ANCHOR_AGENTS if a in AGENT_FILES],
        "rung2": rung2_pool(),
        "floor": [a for a in FLOOR_AGENTS if a in AGENT_FILES],
        # Diagnostics are never swept up by a group; name them explicitly.
        "all": [a for a in AGENT_FILES if a not in BENCHMARK_ONLY_AGENTS],
    }


def resolve_agent_selection(tokens: list[str]) -> list[str]:
    """Expand group names in a --agents token list; keep order, drop duplicates."""
    groups = agent_groups()
    resolved: dict[str, None] = {}
    for tok in tokens:
        for name in (groups[tok] if tok in groups else [tok]):
            resolved.setdefault(name, None)
    return list(resolved)


def print_pool_report() -> None:
    """Print the Rung-2 roster and how much of it is runnable locally."""
    rows = load_opponent_pool()
    if not rows:
        print(f"no opponent pool at {OPPONENT_POOL_CSV} -- run "
              f"scripts/build_opponent_pool.py to build it.")
        return
    runnable = [r for r in rows if r.get("runnable", "").strip().lower() == "true"]
    print(f"Rung-2 opponent pool: {len(rows)} roster entries, "
          f"{len(runnable)} runnable locally ({OPPONENT_POOL_CSV.name})\n")
    print(f"  rung2 group (public field, us excluded): {rung2_pool()}\n")
    print("  RUNNABLE (mirrored + safety-reviewed + wired):")
    for r in runnable:
        print(f"    {r['key']:48s} -> {r['local_agent']}")
    print("\n  REFERENCE-ONLY (notebook URL in roster; not pulled/reviewed here):")
    for r in rows:
        if r.get("runnable", "").strip().lower() != "true":
            print(f"    {r['key']:48s}    {r['source_ref']}")


def _write_fallback_tb(fallback_diag: dict, tb_dir: Path, step: int) -> None:
    """One TensorBoard scalar per agent per fallback stat, under fallback/.

    `step` is the run's epoch-seconds timestamp so successive benchmark runs
    append points to the same curves. Lazy import: torch's SummaryWriter is
    heavyweight and the benchmark must still run where it's absent -- skipping
    is announced on stderr, never silent.
    """
    try:
        if str(REPO / "src") not in sys.path:
            sys.path.insert(0, str(REPO / "src"))
        from pokemon_tcg.logging_utils import TensorBoardLogger
    except Exception as e:
        print(f"[fallback-diag] TensorBoard scalars skipped ({e!r}); "
              f"the JSON report still has everything", file=sys.stderr)
        return
    logger = TensorBoardLogger(tb_dir)
    for a, snap in fallback_diag.items():
        for k, v in snap.items():
            if k == "enabled" or not isinstance(v, (int, float)):
                continue
            # ':' in reason names (policy_exception:ValueError) breaks TB tag
            # grouping; '.' keeps one chart per variant under the same agent.
            logger.log_scalar(f"fallback/{a}/{k.replace(':', '.')}", float(v), step)
    logger.close()
    print(f"[fallback-diag] TensorBoard scalars -> {tb_dir}", file=sys.stderr)


# Populated by load_agent as a side effect, keyed by the same `name` passed
# in. Lets callers that only kept the `agent` callable (e.g. eval_rung3_sanity.py)
# stay on the unchanged load_agent(name) -> fn signature, while run_benchmark
# can still reach each module's diag_snapshot()/diag_reset() (see "Fallback
# diagnostics" below) if the agent exposes the fallback-tracking pattern.
_LOADED_MODULES: dict[str, object] = {}


# Top-level .py module names seen across loaded agent dirs this run.
# Python's sys.modules caches by bare name, so if two agents both ship e.g.
# a `policy_base.py` with DIFFERENT contents, whichever agent imports it
# first wins and the second agent silently runs the first one's code.
# (Checked byte-identical for the wmh family; unverified in general — hence
# this warning rather than an assumption.)
_TOP_LEVEL_PY: dict[str, tuple[str, str]] = {}  # stem -> (first agent, sha256)


def _warn_module_shadowing(agent_name: str, mod_dir: Path) -> None:
    for f in sorted(mod_dir.glob("*.py")):
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        seen = _TOP_LEVEL_PY.get(f.stem)
        if seen is None:
            _TOP_LEVEL_PY[f.stem] = (agent_name, digest)
        elif seen[1] != digest:
            print(f"[module-shadowing] {agent_name} ships {f.name} but a "
                  f"DIFFERENT {f.stem}.py was already loaded for {seen[0]} -- "
                  f"if either agent does `import {f.stem}`, sys.modules serves "
                  f"the first copy to both. Rename the module or verify it is "
                  f"never imported by bare name.", file=sys.stderr)


def warn_unregistered_agent_dirs() -> None:
    """Drift alarm for the CLAUDE.md rule that pool comparisons cover EVERY
    agent in agents/. AGENT_FILES is hand-maintained, so an agents/<dir>
    nobody registered silently drops out of every group (incl. `all`)
    forever. Top-level granularity: a registered family dir (s2_arms/...)
    counts as covered even if individual arms inside it are not."""
    agents_root = REPO / "agents"
    if not agents_root.is_dir():
        return
    covered = {"tb_shared"}  # vendored shared package for the tb_* shims, not an agent
    for p in list(AGENT_FILES.values()) + list(AGENT_MAIN.values()):
        try:
            covered.add(p.relative_to(agents_root).parts[0])
        except ValueError:
            continue  # lives outside agents/ (scripts/, submissions/)
    missing = sorted(d.name for d in agents_root.iterdir()
                     if d.is_dir() and d.name not in covered)
    if missing:
        print(f"[registry-drift] agents/ dirs with NO AGENT_FILES entry -- "
              f"excluded from every group and every pool number until "
              f"registered or deleted: {missing}", file=sys.stderr)


DECK_LISTS_DIR = REPO / "configs" / "deck_lists"


def load_agent(name: str):
    """Load an agent's `agent` callable.

    Prefers the bundle `main.py` (real entry point, deck wired up). Falls back
    to the bare module — for bare modules that need `my_deck` injected (e.g.
    the sample rule baseline), we read it from the matching deck.csv.

    `name` may carry a deck override as `<agent>@<deck-tag>` (e.g.
    `il_agent@dragapult_ex`), where `<deck-tag>.csv` must exist under
    configs/deck_lists/. The base agent loads and runs completely unmodified;
    only its module-level `my_deck` is overwritten after loading, so the SAME
    policy/weights can be benchmarked piloting a different 60-card deck. Each
    call builds a fresh module object (no caching), so distinct deck-tagged
    labels for the same base agent never share state.
    """
    base_name, _, deck_tag = name.partition("@")
    main_py = AGENT_MAIN.get(base_name)
    _warn_module_shadowing(
        name, (main_py.parent if main_py and main_py.exists()
               else AGENT_FILES[base_name].parent))
    if main_py and main_py.exists():
        spec = importlib.util.spec_from_file_location(f"_bench_{name}", main_py)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(main_py.parent))
        spec.loader.exec_module(mod)
        fn = getattr(mod, "agent", None)
        if not callable(fn):
            raise AttributeError(f"{name} bundle has no callable `agent`")
    else:
        path = AGENT_FILES[base_name]
        if not path.exists():
            raise FileNotFoundError(f"no agent file for {base_name}: {path}")
        spec = importlib.util.spec_from_file_location(f"_bench_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(mod)
        # Bare modules that reference a module-level `my_deck` (e.g. the sample
        # rule baseline) have it injected by their main.py; wire it here from deck.csv.
        if not hasattr(mod, "my_deck") and (path.parent / "deck.csv").exists():
            deck = [int(x) for x in (path.parent / "deck.csv").read_text().splitlines() if x.strip()][:60]
            mod.my_deck = deck
        fn = getattr(mod, "agent", None)
        if not callable(fn):
            raise AttributeError(f"{base_name} has no callable `agent`")

    if deck_tag:
        deck_csv = DECK_LISTS_DIR / f"{deck_tag}.csv"
        if not deck_csv.exists():
            raise FileNotFoundError(f"deck override '{deck_tag}' not found: {deck_csv}")
        deck = [int(x) for x in deck_csv.read_text().splitlines() if x.strip()][:60]
        # `agent()` returns the module-level `my_deck` it sees in ITS OWN globals.
        # For wrapper arms (agents/il_arms/, agents/s2_arms/, agents/ppo_arms/,
        # agents/grid_cells/) that owner is the inner il_agent core module the
        # wrapper exec'd, NOT the wrapper module bound to `mod` here. Writing
        # only to `mod` left every wrapper arm silently piloting the deck its
        # wrapper had already injected -- an override that reported success and
        # changed nothing. Write to the function's own globals, and keep the
        # `mod` write so plain modules (where they're the same dict) still work.
        owner = getattr(fn, "__globals__", None)
        if owner is None or "my_deck" not in owner:
            if not hasattr(mod, "my_deck"):
                raise AttributeError(
                    f"{base_name} has no `my_deck` to override (not deck-injectable)")
        if owner is not None:
            owner["my_deck"] = deck
        mod.my_deck = deck

    _LOADED_MODULES[name] = mod
    return fn


def load_glicko_ratings(path: Path = GLICKO_PATH) -> dict[str, glicko1.Rating]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {name: glicko1.Rating(rating=v["rating"], rd=v["rd"]) for name, v in raw.items()}


def save_glicko_ratings(ratings: dict[str, glicko1.Rating], path: Path = GLICKO_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {name: {"rating": r.rating, "rd": r.rd} for name, r in ratings.items()},
        indent=2,
    ))


def play_match(agent_a, agent_b, env_factory, pairs: int = 1,
               name_a: str = "agent_a", name_b: str = "agent_b") -> tuple[int, int, int, float]:
    """Play `pairs` mirrored game pairs (both seat orders each). Returns (a_wins, b_wins, draws, seconds).

    Each mirrored pair cancels first-player advantage: game 1 has A in seat 0,
    game 2 has B in seat 0. A win for "player 0" maps to whichever agent sits
    in player 0's seat.
    """
    t0 = time.time()
    a_wins = b_wins = draws = 0
    for _ in range(pairs):
        # a_in_seat0 tracks which *role* (a or b) sits in seat 0, independent
        # of object identity -- true self-play passes the same callable as
        # both agent_a and agent_b, so `is` checks can't distinguish the
        # roles (both "seat0 is agent_a" and "seat0 is agent_b" are true on
        # every game, double-counting every decisive result).
        for a_in_seat0 in (True, False):
            seat0 = agent_a if a_in_seat0 else agent_b
            seat1 = agent_b if a_in_seat0 else agent_a
            env = env_factory()
            trace = env.run([seat0, seat1])
            final = trace[-1]
            r = [final[i]["reward"] for i in range(2)]
            # kaggle_environments gives a crashed agent reward=None instead of
            # a comparable score (its exception ends the episode early) --
            # treat that seat as an outright loss instead of letting the
            # `None > int` comparison below take the whole tournament down.
            seat_role = ("a", "b") if a_in_seat0 else ("b", "a")
            for seat_idx, role in enumerate(seat_role):
                if r[seat_idx] is None:
                    who = name_a if role == "a" else name_b
                    print(f"  [WARN] {who} crashed (status={final[seat_idx].get('status')}); scoring as a loss")
                    r[seat_idx] = -float("inf")
            if r[0] == r[1]:
                draws += 1
            else:
                winner_role = seat_role[0] if r[0] > r[1] else seat_role[1]
                if winner_role == "a":
                    a_wins += 1
                else:
                    b_wins += 1
    return a_wins, b_wins, draws, time.time() - t0


def run_benchmark(agents: list[str], games_per_pair: int = 8,
                   glicko_path: Path = GLICKO_PATH, out_path: Path | None = None,
                   persist_glicko: bool = True, tb_dir: Path | None = None, focus: str | None = None):
    from kaggle_environments import make

    # The benchmark IS the diagnostic surface: turn on fallback tracking for
    # agents that read PTCG_FALLBACK_TRACK at import (agents/il_agent/
    # agent_core.py and the s2_arms wrappers). setdefault, not assignment, so
    # PTCG_FALLBACK_TRACK=0 in the caller's env still disables it. The
    # submission bundle never sets this, so Kaggle runs stay a genuine no-op.
    os.environ.setdefault("PTCG_FALLBACK_TRACK", "1")

    print(f"Loading {len(agents)} agents: {', '.join(agents)}")
    _LOADED_MODULES.clear()  # drop any modules from a prior run_benchmark() call in this process
    fns = {name: load_agent(name) for name in agents}
    for mod in _LOADED_MODULES.values():
        if hasattr(mod, "diag_reset"):
            mod.diag_reset()

    n = len(agents)
    wins = {a: {b: 0 for b in agents} for a in agents}  # wins[a][b] = a beat b
    games = {a: {b: 0 for b in agents} for a in agents}
    wall_clock = {a: {b: 0.0 for b in agents} for a in agents}  # seconds, symmetric per pairing
    totals = {a: {"w": 0, "g": 0} for a in agents}
    # (player_a, player_b, score_a) triples for the Glicko-1 rating period.
    # Self-play is excluded: an agent playing itself carries no information
    # about its rating relative to the field.
    glicko_games: list[tuple[str, str, float]] = []

    # `focus` turns the full round-robin into a STAR: only focus-vs-everyone is
    # played, skipping every other-vs-other pairing. Testing one model against a
    # 42-agent bed is then 42 pairings instead of 903 -- the pool's internal
    # ratings come from a separate one-off round-robin and are merged in, so
    # nothing is lost by not replaying them for every candidate.
    if focus:
        pairlist = [(focus, b) for b in agents if b != focus]
    else:
        pairlist = [(agents[i], agents[j]) for i in range(n) for j in range(i, n)]
    total_pairs = len(pairlist)
    done = 0
    run_t0 = time.time()
    if True:
        for a, b in pairlist:
            pairs = games_per_pair if a != b else max(1, games_per_pair // 2)
            aw, bw, dr, secs = play_match(fns[a], fns[b], lambda: make("cabt"), pairs, name_a=a, name_b=b)
            wins[a][b] += aw
            wins[b][a] += bw
            games[a][b] += aw + bw + dr
            games[b][a] += aw + bw + dr
            wall_clock[a][b] += secs
            wall_clock[b][a] += secs
            # Self-play stays in the matrix/JSON but NOT in the overall
            # totals: an agent's games against itself are ~50/50 by
            # construction and were diluting every overall win% toward 50 by
            # an amount that depended on pool size.
            if a != b:
                totals[a]["w"] += aw
                totals[b]["w"] += bw
                totals[a]["g"] += aw + bw + dr
                totals[b]["g"] += aw + bw + dr
            if a != b:
                glicko_games += [(a, b, 1.0)] * aw
                glicko_games += [(a, b, 0.0)] * bw
                glicko_games += [(a, b, 0.5)] * dr
            done += 1
            n_games_pair = aw + bw + dr
            print(
                f"[{done}/{total_pairs}] {a} vs {b}: "
                f"{a} {aw}-{bw}{('-' + str(dr) + 'd') if dr else ''}  "
                f"({n_games_pair} games, {secs:.1f}s, {secs/n_games_pair:.2f}s/game)"
            )
    total_wall_clock = time.time() - run_t0

    # ---- report ----
    print("\n=== Head-to-head win matrix (rows beat cols; cell = row's wins [95% CI]) ===")
    header = "agent".ljust(22) + "".join(b[:6].ljust(8) for b in agents) + "win% [95% CI]"
    print(header)
    overall = {}
    overall_ci = {}
    for a in agents:
        row = a.ljust(22)
        w = totals[a]["w"]
        g = totals[a]["g"]
        overall[a] = (100.0 * w / g) if g else 0.0
        overall_ci[a] = glicko1.wilson_ci(w, g)
        cells = ""
        for b in agents:
            if a == b:
                cells += "-".ljust(8)
            else:
                cells += f"{wins[a][b]}".ljust(8)
        lo, hi = overall_ci[a][1] * 100, overall_ci[a][2] * 100
        row += cells + f"{overall[a]:5.1f}  [{lo:4.1f},{hi:4.1f}]"
        print(row)

    print("\n=== Per-pairing win rate with Wilson 95% CI (row vs col) ===")
    for a in agents:
        for b in agents:
            if a >= b:
                continue
            n_ab = games[a][b]
            if n_ab == 0:
                continue
            p, lo, hi = glicko1.wilson_ci(wins[a][b], n_ab)
            print(f"  {a} vs {b}: {wins[a][b]}/{n_ab} = {p*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]  "
                  f"({wall_clock[a][b]:.1f}s, {wall_clock[a][b]/n_ab:.2f}s/game)")

    print("\n=== Overall win rate (all games, self-play excluded), Wilson 95% CI ===")
    for a in sorted(agents, key=lambda x: -overall[x]):
        lo, hi = overall_ci[a][1] * 100, overall_ci[a][2] * 100
        print(f"  {a:22s} {overall[a]:5.1f}%  [{lo:4.1f},{hi:4.1f}]  "
              f"({totals[a]['w']}/{totals[a]['g']}){report_tag(a)}")

    # ---- Glicko-1 ratings ----
    # Win-rate is a single-run snapshot; Glicko carries a rating + confidence
    # (RD) forward across every benchmark run, weighted by opponent strength,
    # so it's a better estimate of true skill than this run's win% alone.
    prior_ratings = load_glicko_ratings(glicko_path) if persist_glicko else {}
    glicko = glicko1.run_rating_period(prior_ratings, glicko_games)
    if persist_glicko:
        save_glicko_ratings(glicko, glicko_path)

    print("\n=== Glicko-1 ratings (persisted across runs, higher = stronger) ===")
    ranked = sorted(agents, key=lambda x: -glicko[x].rating)
    for a in ranked:
        r = glicko[a]
        print(
            f"  {a:22s} {r.rating:7.1f}  (RD {r.rd:5.1f}, 95% CI +/-{2*r.rd:5.1f})  "
            f"GXE {glicko1.gxe(r):5.1f}%{report_tag(a)}"
        )
    if any(is_non_learned(a) for a in ranked):
        print("  note: rows marked (anchor)/(floor)/(hand-written) are calibration "
              "only -- do not quote them as a comparison against a trained model.")

    # ---- Fallback diagnostics ----
    # Agents that expose the _DIAG/diag_snapshot pattern (see
    # agents/mega_lucario/agent_core_improved.py, agents/improved_probabilistic/main.py,
    # agents/mechi22_alakazam/agent_core.py, agents/il_agent/agent_core.py and
    # the agents/s2_arms wrappers) count how often their never-crash fallback
    # layers actually fire across every game just played. A nonzero
    # fallback_rate here means the policy is silently being bypassed on real
    # inputs -- worth investigating even though the agent never crashed. The
    # rate alone is not the story: read it against `decisions` (the
    # denominator) and the per-reason counts in `raw`.
    fallback_diag = {}
    fallback_first = {}
    for a in agents:
        mod = _LOADED_MODULES.get(a)
        if mod is not None and hasattr(mod, "diag_snapshot"):
            fallback_diag[a] = mod.diag_snapshot()
            first_fn = getattr(mod, "diag_first", None)
            if callable(first_fn) and first_fn():
                fallback_first[a] = first_fn()
    if fallback_diag:
        print("\n=== Fallback diagnostics (fraction of decisions that hit a fallback layer) ===")
        for a, snap in fallback_diag.items():
            print(f"  {a:22s} fallback_rate={snap.get('fallback_rate', 0.0):6.2%}  "
                  f"decisions={snap.get('decisions', 0)}  raw={dict(snap)}")
        # Also to stderr: survives `> results.txt` redirection, and is the
        # channel the smoke test / CI reads without parsing the full report.
        print("[fallback-diag] end-of-run summary "
              "(fallbacks/decisions; first occurrences in the result JSON):",
              file=sys.stderr)
        for a, snap in fallback_diag.items():
            reasons = {k: v for k, v in sorted(snap.items())
                       if k not in ("enabled", "decisions", "fallbacks", "fallback_rate")
                       and isinstance(v, int) and v > 0}
            print(f"[fallback-diag] {a}: {snap.get('fallbacks', 0)}"
                  f"/{snap.get('decisions', 0)} = "
                  f"{snap.get('fallback_rate', 0.0):.2%}  by_reason={reasons}",
                  file=sys.stderr)
        if tb_dir is not None:
            _write_fallback_tb(fallback_diag, tb_dir, step=int(time.time()))

    result = {
        "agents": agents,
        "games_per_pair": games_per_pair,
        "seeded": False,
        "seed_note": "cabt exposes no RNG seed from Python (native engine, no seed param on "
                      "battle_start or in cabt.json's configuration schema) -- deck shuffles "
                      "and opening hands are NOT held constant across runs or pairings.",
        "wins": wins,
        "games": games,
        "wall_clock_sec": wall_clock,
        "total_wall_clock_sec": total_wall_clock,
        "overall_win_pct": overall,
        "overall_win_pct_wilson_ci": {
            a: {"p": overall_ci[a][0] * 100, "lo": overall_ci[a][1] * 100, "hi": overall_ci[a][2] * 100}
            for a in agents
        },
        "glicko": {
            a: {"rating": glicko[a].rating, "rd": glicko[a].rd, "gxe": glicko1.gxe(glicko[a])}
            for a in agents
        },
        "fallback_diag": fallback_diag,
        "fallback_first": fallback_first,
    }
    out_path = out_path or (REPO / "reports" / "agent_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\ntotal wall clock: {total_wall_clock:.1f}s")
    print(f"saved: {out_path}")
    print(f"saved: {glicko_path}" if persist_glicko else "glicko: not persisted (isolated run)")
    return result


def main():
    ap = argparse.ArgumentParser(description="Benchmark agent vs agent performance.")
    # NB: the default must exclude BENCHMARK_ONLY_AGENTS. It enumerates
    # AGENT_FILES directly rather than going through the `all` group, so the
    # group-level exclusion does NOT cover it -- without this filter a bare
    # `benchmark_agents.py` run would pull in the exploiter and trip the
    # force-no-glicko-persist guard, silently stopping rating persistence.
    ap.add_argument("--agents",
                    default=",".join(k for k in AGENT_FILES
                                     if k not in BENCHMARK_ONLY_AGENTS),
                    help="comma-separated agents or group names. Groups: ours "
                         "(our TRAINED agents -- the reportable set), rung2 "
                         "(public field), anchors (silent ladder calibration, "
                         "never a result), heuristics (our hand-written "
                         "policies), floor, all. Agents: " + ", ".join(AGENT_FILES))
    ap.add_argument("--games", type=int, default=8, dest="games_per_pair",
                    help="mirrored game pairs per ordered agent pair")
    ap.add_argument("--list-pool", action="store_true",
                    help="print the Rung-2 opponent-pool roster and exit")
    ap.add_argument("--glicko-path", type=Path, default=GLICKO_PATH,
                    help="where to load/persist Glicko ratings (default: reports/glicko_ratings.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to save the result JSON (default: reports/agent_benchmark.json)")
    ap.add_argument("--focus", default=None,
                    help="star mode: play ONLY <focus> vs every other agent, skipping "
                         "other-vs-other. Testing one candidate against a 42-agent bed "
                         "becomes 42 pairings instead of 903.")
    ap.add_argument("--no-glicko-persist", action="store_true",
                    help="score Glicko for this run's printout but don't read/write --glicko-path "
                         "(use for isolated runs, e.g. deck-arm sweeps, that shouldn't pollute "
                         "the standing ratings)")
    ap.add_argument("--tb-dir", type=Path, default=REPO / "runs" / "benchmark_fallbacks",
                    help="TensorBoard log dir for fallback-diagnostic scalars "
                         "(default: runs/benchmark_fallbacks)")
    ap.add_argument("--no-tb", action="store_true",
                    help="skip writing fallback TensorBoard scalars")
    args = ap.parse_args()
    if args.list_pool:
        print_pool_report()
        return
    warn_unregistered_agent_dirs()
    # Expand group names (ours/rung2/floor/all); deck-arm tokens like
    # `il_agent@dragapult_ex` aren't groups and pass through untouched.
    tokens = [a.strip() for a in args.agents.split(",") if a.strip()]
    agents = resolve_agent_selection(tokens)
    unknown = [a for a in agents if a.partition("@")[0] not in AGENT_FILES]
    if unknown:
        sys.exit(f"unknown agent(s): {unknown}. Available: {list(AGENT_FILES)}; "
                 f"groups: {list(agent_groups())}")
    if not agents:
        sys.exit("no agents selected")
    # Diagnostics must never enter the standing, COMPOUNDING ratings file: one
    # such run would permanently bias every other agent's rating. Including one
    # forces this run to be Glicko-isolated (it is still scored for the
    # printout, exactly like --no-glicko-persist).
    diagnostics = [a for a in agents if a.partition("@")[0] in BENCHMARK_ONLY_AGENTS]
    persist_glicko = not args.no_glicko_persist
    if diagnostics and persist_glicko:
        persist_glicko = False
        print(f"NOTE: {diagnostics} is benchmark-only -- forcing --no-glicko-persist "
              f"so {args.glicko_path.name} is not polluted. Win rates below are "
              f"still valid; the exploiter samples at T=1.0, so quote CIs.",
              file=sys.stderr)
    run_benchmark(agents, args.games_per_pair, glicko_path=args.glicko_path,
                  out_path=args.out, persist_glicko=persist_glicko,
                  tb_dir=None if args.no_tb else args.tb_dir, focus=args.focus)


if __name__ == "__main__":
    main()
