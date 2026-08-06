# Search and RL under a deck confound (staged generalization test)

- **Hypothesis:** the search and self-play gains that died on the ladder died
  because both were trained *and* measured inside a single-deck mirror of the
  corpus's rarest archetype. Mechanism: `agents/mcts_il_agent/deck.csv` and
  `selfplay.py:39`'s `DECK_PATH` are both byte-identical to
  `configs/deck_lists/mega_lucario_ex.csv` (md5 `4d555b8b…`) — **1 of 4554 train
  episodes**. A policy and a search prior tuned in that mirror have no reason to
  transfer to a ladder where 76.6% of games involve Marnie's Grimmsnarl ex, so
  the observed local→ladder inversions are what the deck confound predicts.
- **Design:** staged generalization test, not an ablation. The question is
  whether a measured gain survives off the distribution it was measured on —
  attribution among components comes later, and only if a gain survives at all.
  Cheapest-first so the training arms are only paid for if the free arm holds.
- **Independent variable:** the deck the agent pilots (Stage A) / the size of the
  training deck pool K (Stage B). One per stage.
- **Baseline:** plain `il_agent` (BC, `models/il_agent`, epoch-2, sha256
  `bce726e5…`), same checkpoint used as the MCTS prior — so the MCTS arm differs
  from baseline only by the search wrapper.

## Prior work checked

| Source | What it already settles |
|---|---|
| [`reports/deck_selection.md`](../../reports/deck_selection.md) | Corpus deck census (Grimmsnarl 3488/4554 = 76.6%, Lucario 1). Do not recompute. |
| ledger ref 55248781 | MCTS on ladder: 600.0 → **291.4** converged. Its `deck:` field says "Grimmsnarl" — **factually wrong**, see md5 above. |
| ledger refs 55270787 / 55248985 | Grimmsnarl **314.1** vs Lucario **418.0**, same checkpoint. One reading, inside the ~230-pt same-build drift band. |
| memory `deck-needs-competent-pilots-in-pool` | Root cause of the above: verified-only ρ = +0.000 with the Grimmsnarl arm in, **+1.000 with it removed**. Pool has 13 Lucario pilots (6 in top 10) vs 2 floor-tier Grimmsnarl. |
| memory `search-bc-prior-no-measurable-gain` | A *different* search path (`agent_core_improved`, UCB1 + BC prior) gave Δ −1.1 pts over the pure heuristic at 900 games. Independent evidence that search-with-BC-prior is weak here. |
| `reports/g{1,2,3}_*.json` | Self-play g2 beats BC **59.0% [54.2, 63.8]** (n=400) locally; ladder inverted (267.4 vs 418.0). |
| [`notes/study-kiyotah-rl-mcts.md`](../study-kiyotah-rl-mcts.md) | kiyotah's reference loop is from-scratch AlphaZero, no IL, evaluated only vs `random_agent` (§5 #12). Not a transfer claim. |

**Phase ordering is settled, not under test:** keep the BC init. VGC-Bench's
BC-initialized self-play arms beat from-scratch RL in the same domain, and our
own from-scratch PPO scored **275.1** on the ladder.

---

## Stage 0 — repair the measurement instrument (prerequisite, blocking)

No deck arm produces a readable number until the pool can contest Grimmsnarl.
Current pool (`reports/glicko_ratings.json`, 39 agents): exactly **one**
Grimmsnarl pilot, `wmh_grimmsnarl` at **1320.3 (rank 32/39)**. Adding it was
already tried and *lowered* predictiveness (ρ +0.765 → +0.692) — presence ≠
contest.

- **Method (per Rami's call, 2026-08-05):** harvest public Grimmsnarl agents from
  the Kaggle Code tab and wire them in. Safety-review each before import, probe
  `battle_start` legality before trusting any decklist (>1 ACE SPEC ⇒ INVALID
  every game), and expect ~1/3 of slugs to be dead and ~1/4 payload-wrapped.
- **Pre-registered pass bar:** ≥3 Grimmsnarl pilots at Glicko **≥1600** (matching
  the Lucario contingent's density at the top), **and** verified-only Spearman ρ
  on the anchored bed recovers to **≥ +0.60** with a Grimmsnarl arm included.
- **If Stage 0 fails to find them:** that is itself a finding — it means no
  competent Grimmsnarl pilot is public, and the fallback is BC-training one from
  the 3488 corpus episodes filtered on `avg_score`. Do not proceed to Stage A on
  an uncontested pool; that is the exact error that produced the 08-05 fiasco.
- **Cost:** ~2h, no training, no MPS.

### Stage 0 progress (2026-08-05)

**The harvest may not be needed.** Before downloading anything, censused what is
already wired: `scripts/census_pool_archetypes.py` keys all 48 `agents/*/deck.csv`
by ace Pokémon (the same key as `reports/deck_selection.md` §1), so archetype
comes from the card database rather than the directory name. Output:
`reports/pool_archetype_census.json`.

**Agent names do not track archetypes.** Five pool agents are misfiled, and the
two that matter are Grimmsnarl pilots hiding under Alakazam names:

| Agent | Registry said | Actually pilots |
|---|---|---|
| `prvsiyan_grimbelief_alakazam` | Alakazam belief | **Marnie's Grimmsnarl ex** |
| `prvsiyan_templates_alakazam` | Alakazam templates | **Marnie's Grimmsnarl ex** |
| `wmh_mewtwo` | Team Rocket's Mewtwo ex | Team Rocket's Spidops |
| `wmh_ogerpon` | Ogerpon | Barbaracle |
| `tb_starmie` | Starmie | Mega Froslass ex |

Registry comments corrected in `scripts/benchmark_agents.py`.

**Revised pool coverage** (was: 13 Lucario / 2 Grimmsnarl):

| Archetype | Pilots | Rated | Top Glicko |
|---|---:|---:|---:|
| Mega Lucario ex | **18** | 7 | 1752.0 |
| Marnie's Grimmsnarl ex | **3** | **1** | 1320.3 |

So the skew is worse than recorded (18:3, not 13:2) — but **2 of the 3 Grimmsnarl
pilots have never been benchmarked**, which is why the archetype looked
floor-tier. Their rating is unknown, not low. Both pass the capability audit
(stdlib + `cg` only; no subprocess/socket/exec/network; all file I/O read-mode)
and both are already registered and runnable. All 3 Grimmsnarl decks pass the
legality gate (SUBMIT + play-out, `reports/deck_legality_grimmsnarl.json`).

**Queued:** `scripts/run_stage0_grimmsnarl_pool.sh` — the 3 pilots vs the 8-agent
anchored pool (ρ +0.929), 20 games/ordered pair, rating into an isolated
`reports/glicko_stage0.json` seeded from the live file. **Held behind a load
gate** (1-min loadavg < ncpu/2): `romanrozen_strong_start` and both prvsiyan
agents run live search with unbudgeted think time, so rating them while four
other worktrees saturate the CPU would measure contention, not policy. Launched
at loadavg 14.06/12 cores; gate polls every 5 min, aborts after 6h.

Public-agent harvest is deferred until this reads out — if the two unmeasured
pilots land at ≥1600 the bar is nearly met for free.

### 🛑 STAGE 0 RESULT — FAIL on the pre-registered bar (2026-08-05)

Read out of the sibling 52-agent tournament (`il-mcts-agent-glicko-bdd7f2`,
`reports/il_mcts_wide_benchmark.json`, 8 games/ordered pair, **408 games/agent**,
Glicko included) — zero additional compute spent.

**Bar was: ≥3 Grimmsnarl pilots at ≥1600.** Actual, out of 52 agents:

| Grimmsnarl pilot | Rank | Rating | Win% |
|---|---:|---:|---:|
| `wmh_grimmsnarl` | 39/52 | 1301.3 | 31.0% |
| `prvsiyan_grimbelief_alakazam` | **44/52** | 1196.2 | 21.2% |
| `prvsiyan_templates_alakazam` | **46/52** | 1186.1 | 20.2% |

**Zero field pilots clear 1600.** The two previously-unmeasured pilots came in
*below* `wmh_grimmsnarl`, not above it — they are the 44th and 46th of 52 agents.
The earlier read ("rating unknown, not low") is settled: it was low.

**And this is now a far sharper account of the 08-05 deck fiasco.** In the same
tournament, our own Grimmsnarl arms rank **1st and 3rd of 52**:

| Our arm | Rank | Rating |
|---|---:|---:|
| `il_agent@marnies_grimmsnarl_ex` | **1/52** | 1824.1 |
| `il_mcts_agent@marnies_grimmsnarl_ex` | 3/52 | 1813.9 |
| `il_agent` (same policy, Mega Lucario ex) | 41/52 | 1277.2 |

The *same BC policy* swings **+546.9 Glicko** on the deck axis alone, tops a
52-agent pool — and that exact configuration scored **314.1 on the ladder**,
against **418.0** for the Lucario version. The pool ranks the two decks in the
exact opposite order from Kaggle, and now we can say precisely why: the only
three agents that contest Grimmsnarl rank 39th, 44th and 46th. Our arm is
farming a field that cannot play the archetype.

**Consequences:**
1. The public-agent harvest is now **mandatory**, not the optional first try.
   Rami's Stage-0 method choice stands, but its fallback (BC-train a pilot from
   the 3,488 corpus episodes) is now the likely path — no *published* competent
   Grimmsnarl pilot has turned up in 48 wired agents.
2. **No deck claim is measurable until this is fixed** — in either direction.
   "Grimmsnarl is better" and "Grimmsnarl is worse" are both currently unsupported.
3. Also visible: `mcts_il_agent` 1254.4 (42/52) sits *below* plain `il_agent`
   1277.2 (41/52), and `il_mcts_agent@grimmsnarl` is Δ −10.1 vs the plain arm on
   the same deck — a third and fourth independent null for search.

### ⏸️ EARLIER PAUSE 2026-08-05, on Rami's call — session contention

The gated run was **killed without playing a game** and its scratch artifacts
(`reports/glicko_stage0.json`, `runs/stage0_grimmsnarl.log`) removed. Four
sessions were working this problem at once and stepping on each other:

| Worktree | State | Overlap |
|---|---|---|
| `il-mcts-agent-glicko-bdd7f2` | 1286/1378 (93%), 3h46m | 52-agent tournament incl. all 3 Grimmsnarl pilots **and** `il_agent@marnies_grimmsnarl_ex`, `il_mcts_agent@marnies_grimmsnarl_ex` → `reports/il_mcts_wide_benchmark.json` (`--games 4`, `--no-glicko-persist`) |
| `mcts-il-agent-v2-research` | done | the mechanism result that superseded Stage A |
| `il-model-deck-selection-ec748b` | PR #47 open | "local pools can't measure decks they can't contest" — independently concurrent with this card's Stage 0 |
| this one | paused | — |

Three of them held loadavg at 8–14 on 12 cores, which is why the gate never
cleared. Rami chose to let them finish rather than consume the sibling matrix
mid-flight.

**To resume, in order:**
1. `git fetch` + `gh pr list` first — PR #47 and the v2-research branch may land
   findings that moot more of this card (this exact overlap already cost one
   duplicated design here).
2. Read `reports/il_mcts_wide_benchmark.json` from the sibling worktree. It has
   408 games/agent across 51 opponents — enough to place
   `prvsiyan_grimbelief_alakazam` and `prvsiyan_templates_alakazam` without
   re-running anything. Glicko must be computed offline from its matrix
   (`--no-glicko-persist`, so no ratings file was written).
3. Only if that leaves <3 Grimmsnarl pilots at the bar: `scripts/run_stage0_grimmsnarl_pool.sh`
   (load-gated, ready as-is), then the public-agent harvest.

**Uncommitted in this worktree:** `scripts/census_pool_archetypes.py`,
`scripts/run_stage0_grimmsnarl_pool.sh`, this card,
`reports/pool_archetype_census.json`, `reports/deck_legality_grimmsnarl.json`,
and archetype-comment corrections in `scripts/benchmark_agents.py`.

## Stage A — ⛔ SUPERSEDED 2026-08-05, BEFORE IT RAN

**Falsified by a better instrument, not by its own result.** The
`mcts-il-agent-v2-research` worktree measured the search *mechanism* directly
(`scripts/probe_mcts_mechanism.py`, `scripts/ablate_search_count.py`, write-up
`reports/mcts_il_v2_diagnosis.md`) while this stage was still queued:

- the tree overrides its own prior's top child on **2.7%** of 1,905 decisions
  (0.93% when the prior is confident, Δp > 0.10);
- terminal nodes are **0.43%** of nodes created, so with leaf value ≡ 0 nearly
  every backed-up value is 0 and PUCT degenerates to a prior-proportional
  round-robin — `argmax c·p/(1+visit)`;
- paired ablation, 80 games/arm vs `il_agent`, 0 fallbacks both:
  **N=0 → 57.5% [46.6, 67.7] @ 1.06 s/game; N=30 → 55.0% [44.1, 65.4] @ 11.68
  s/game. Δ = −2.5 pp (0.3σ) at 11× the cost.**

Stage A asked whether the 67.2% search edge survives off the Lucario mirror.
**There is no search edge to transfer.** `mcts_il_agent` bundled three changes
and the report credited the wrong one: the real variable was the **decode rule**
(`il_agent` decodes autoregressively with re-masking as trained;
`mcts_il_agent` scores whole combos as the mean of member logits in one forward
and can only emit size `maxCount` or size 0). The stage's own kill criterion
(≤55%) is already met on Lucario — N=30 scored exactly 55.0%.

**Not rewritten to keep the stage alive.** Amending a pre-registration after
seeing the data is the failure mode pre-registration exists to prevent; the
honest record is that it was falsified pre-run. The deck-confound hypothesis is
**not** rescued by re-running this on Grimmsnarl: a mechanism fact (the tree
moves 2.7% of decisions, leaf value is identically zero) generalizes across
decks far better than a win rate does, and no λ sweep or larger N can fix a
search whose value signal is identically zero.

**What this does NOT touch:** Stage B. The RL half of the hypothesis stands
untested — g1/g2/g3 all trained at K=1 on Lucario, `--deck-pool` never used.

**New lead, deliberately not promoted to a stage yet:** the decode rule. N=0
sits at 57.5% over `il_agent`, but at n=80 the CI [46.6, 67.7] includes 50, and
the source report states the nominal CIs are optimistic. That is a point
estimate worth a properly-powered paired run, not a finding. Design it as its
own card if pursued.

<details>
<summary>Original Stage A design, kept for the record</summary>

### Stage A — does the search edge survive off the Lucario mirror? (free)

The load-bearing arm. No training: swap one file.

- **Change:** `agents/mcts_il_agent/deck.csv` → `configs/deck_lists/marnies_grimmsnarl_ex.csv`.
  The `<agent>@<deck-tag>` override in `scripts/benchmark_agents.py` does this
  without touching the policy.
- **Arms (2×2, deck × search):** {`il_agent`, `mcts_il_agent`} × {Lucario,
  Grimmsnarl}. The Lucario column reproduces the known 67.2% and acts as the
  positive control — if it doesn't reproduce, the harness changed and the
  Grimmsnarl column is uninterpretable.
- **Metric & protocol:** MCTS win rate vs plain BC, head-to-head, mirrored pairs,
  **≥300 decided games per cell** (the original 67.2% rested on n=119 — too thin
  to survive a 12pp move). Wilson 95% CI. Report vs the Stage-0-repaired pool too.
- **Pre-registered decision:** **drop search entirely if MCTS-over-BC on
  Grimmsnarl is ≤55%.** Rationale: 55% is the floor at which ~209 ms/decision and
  a 13.13 MiB bundle pay for themselves; the mirror reading was 67.2%, so ≤55%
  means the edge was mirror-specific. Adopt and proceed to Stage B if >55% with a
  CI excluding 50%.
- **Confound control:** run the `run-fallback-diagnostic` skill on the
  deck-swapped agent first. A silent `_safe_choice` fallback on an unfamiliar
  deck would read as "search doesn't transfer" and be entirely an artifact —
  this exact failure already voided one MCTS run (12.76% silent fallback
  pre-OOB-fix).
- **Cost:** ~4h wall-clock, CPU only, no MPS contention.

</details>

## Stage B — is K=1 why RL didn't transfer? (conditional on A)

- **Change:** `train_ppo_puffer.py --deck-pool` (already implemented,
  `:205`) — never used by any `run_selfplay_g*.sh`.
- **Arms:** K=1 (Lucario, reproduces g1/g2/g3) vs K=7
  (`configs/deck_pools/legal_decklists.txt`), mirrored draws so both seats play
  the episode's deck — otherwise the arm measures deck matchup, not policy.
  Equal **steps**, not equal epochs.
- **Metric:** win rate vs the Stage-0-repaired pool, and vs the frozen BC init.
  The g2 result to beat is 59.0% [54.2, 63.8] over BC *in-mirror*; the question
  is whether K=7 holds a comparable edge *out*-of-mirror.
- **Pre-registered decision:** adopt K>1 as the default training config if the
  K=7 arm's out-of-mirror edge over BC exceeds the K=1 arm's by ≥5pp with
  non-overlapping CIs. A flat curve is a real result — write it down.
- **Cost:** ~8h training per arm, single seed ⇒ **provisional**. Nice everything,
  never overlap MPS jobs.

## Stage C — close the AlphaZero loop (conditional on B)

Only reached if search *and* deck diversity both survive. The piece that does not
exist today: `search_prior_mcts.py` runs a **frozen** prior and never trains;
`ppo.py` trains and never searches. Closing the loop means retraining the policy
on MCTS visit counts and feeding it back as the next search prior — which is what
makes self-play and search compound instead of being alternatives. Design this as
its own card; do not scope it here.

---

## Result (fill after)
- **Observed:**
- **Decision:** adopted / dropped / inconclusive —
- **What we learned:**
