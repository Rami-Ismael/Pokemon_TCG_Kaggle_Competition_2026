ROLE
You are a senior ML engineer with Kaggle competition experience, working in my repo at
/Users/ramiismael/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026.
You own the architecture decision. Where I have stated a preference below and it is wrong,
say so and argue the alternative before writing code. Do not flatter the plan.
You are also designing STAGE 1 OF A THREE-STAGE PIPELINE, not a one-off model. Every
stage-1 choice that forecloses stage 2 or 3 is a bug even if it improves stage-1 metrics.

THIS PROMPT RUNS IN TWO SESSIONS, NOT ONE. Session A is Phase 0 and ends at the HARD STOP
GATE below. Session B is everything after, and only begins once I have written an approval.
If you find yourself writing model code in session A, you have misread the prompt.

GOAL (the thing that is actually being optimised)
Ship a behavior-cloning agent for the Pokemon TCG AI Battle Challenge (Simulation track,
Kaggle) that beats my existing rule-based/search baselines on ladder win-rate, CPU-only,
inside the submission time budget. The terminal deliverable is A SUBMISSION BUNDLE UPLOADED
TO KAGGLE WITH A RECORDED LADDER SCORE — not a checkpoint, not a notebook, not a metric.
Offline next-action accuracy is a proxy and a weak one.
Success = a submitted agent scoring above my current best, plus a reproducible training run
I can narrate in the Strategy-track writeup.
Failure I want detected early = high offline accuracy, loses matches.

THE ROADMAP THIS IS STAGE 1 OF
  Stage 1 (this work)  behavior cloning from the daily episode dump
  Stage 2              offline RL with weighted-BC filters over the same data
  Stage 3              self-play / synthetic fine-tuning (PPO), see PTCG Track F
This is the published progression from "Human-Level Competitive Pokemon via Scalable Offline
RL with Transformers" (Grigsby et al., RLC 2025, arXiv:2504.04395) for the closest analogous
domain. I am not asking you to build stages 2 and 3. I am asking you to make stage 1 a valid
initialisation for them, and to say out loud wherever a stage-1 convenience would cost me later.

I HATE RULE-BASED AGENTS. The deliverable is a trained neural network. A hand-written
heuristic is allowed in exactly two places and nowhere else: (a) the never-crash fallback in
the inference wrapper, (b) the fixed deck constant at the deck-selection step. If your
recommendation drifts toward "just write a scorer", say so explicitly as a recommendation I
must approve, do not do it quietly.

=== PHASE 0 - DISCOVERY GATES. Write NO model code until all seven are answered in writing. ===
Print your findings as a short markdown report and STOP for my review after Phase 0.

0.1 WHAT DOES THE AGENT ACTUALLY SEE?
    Load one episode JSON from data/episodes/splits/train-2026-07-26/ and dump the full
    obs_dict at 5 decision points spread across a match (setup, early, mid, late, terminal).
    Enumerate EVERY key that appears, with type, shape/cardinality, and observed range.
    Explicitly answer:
      - Which fields are the acting player's PRIVATE view vs public? Does the replay log
        leak the opponent's hand or deck order? If it does, you must strip it — training on
        information the agent will not have at inference is the single most likely way this
        project produces a model that scores well offline and loses on the ladder.
      - What is in obs["logs"] and is it usable as history?
      - What is obs["select"]["context"] (SelectContext) and what is the full set of values
        observed across the day? Report the frequency of each.
      - Confirm option list semantics: obs["select"]["option"], OptionType enum, minCount,
        maxCount. Confirm obs["select"] is None means the deck-selection step.
      - Confirm obs["current"]["result"] encoding (<0 in progress, 0 win, 1 loss, 2 draw)
        against real data. It is NOT symmetric +/-1; do not feed it anywhere raw.
      - Report whether obs["search_begin_input"] exists in the dumps and what it contains.
        If it is the pre-serialized determinization payload for the engine's SearchBegin,
        that is the highest-value unknown in my notes and it changes the Q28 hybrid design.

0.2 DO I HAVE THE SIMULATOR?  [REVISED 2026-08-01 — NO LONGER A HARD GATE]
    Determine, BY RUNNING IT, whether the cabt / kaggle-environments environment is
    steppable. Known and to be confirmed, not rediscovered: cabt ships inside the
    kaggle-environments PyPI wheel. As of the hosts' 2026-06-30 update the cg library
    SUPPORTS macOS AND LINUX ARM64 — libcg.dylib and libcg-arm64.so were added alongside
    the existing libcg.so and cg.dll, with no change to the cg API. So the expected path
    on this Apple Silicon laptop is NATIVE, not Docker.
      - FIRST, print which binary actually loads:
          python -c "from kaggle_environments.envs.cabt.cg import sim; print(sim.lib)"
        A path ending .dylib means native. A path ending .so means the library did not
        update — refresh the cg dataset / sample_submission from the competition Data tab
        and re-check BEFORE falling back to Docker --platform linux/amd64 under QEMU.
        Do not silently accept the emulated path; it costs roughly 3x throughput.
    Report:
      - which binary loaded (sim.lib), and native vs emulated
      - import path, how to instantiate a match, how to plug in an agent callable
      - measured steps/sec for two random agents over 20 matches. If BOTH native and
        emulated paths run, report both — the ratio is the only real local measurement
        of the emulation penalty and it replaces a borrowed 3-10x rule of thumb
      - whether the unbound symbols SearchBegin / SearchStep / SearchEnd / SearchRelease /
        AllCard / AllAttack are reachable, since a BC-as-prior hybrid needs them
    Because the engine now runs natively, treat Rung 2 round-robin as AFFORDABLE and
    self-play (F5) as ON THE TABLE unless the measured steps/sec says otherwise.
    If it is NOT steppable by either route, say so plainly and STOP. Phase 4 collapses to
    Rung 1 and "the model works" becomes unfalsifiable. Do not build around a guess.

0.3 WHAT ALREADY EXISTS IN THIS REPO?
    Inventory: existing agents (names, entry points, call signature), scripts/benchmark_agents.py
    (does it exist? what interface must an agent expose to be entered in it?), any existing
    encoder, any existing submission bundler, any existing run/ or runs/ layout.
    Report the exact contract a new il_agent must satisfy. Do not refactor anything yet.

0.4 HARDWARE PROBE — PARTIALLY RESOLVED, DO NOT REDO THE MPS BENCHMARK
    ALREADY MEASURED on this laptop, 5M-parameter set-transformer, forward+backward:
      - MPS float32 is 3.2x CPU float32
      - MPS bfloat16 (direct dtype, no autocast) is 1.08x MPS float32
    CONCLUSION ALREADY DRAWN: train on MPS in float32. bf16 is below my 1.3x bar and is not
    worth the fallback complexity. Do not re-argue this on speed grounds.
    STILL TO MEASURE, and report as numbers not estimates:
      - Total + available RAM: sysctl hw.memsize, vm_stat; GB free under normal load
      - Chip and core counts: sysctl -n machdep.cpu.brand_string, sysctl hw.ncpu
      - torch.__version__, torch.backends.mps.is_available(), sw_vers, df -h .
      - THE ONE THAT MATTERS: x86-64 single-threaded CPU inference latency per decision, at
        the model size you land on, with the option-scoring batched. M-series numbers do not
        transfer — different ISA, different BLAS kernels, no MPS. Get this from a CPU-only
        Kaggle notebook or the amd64 Docker image. Everything in Phase 1.2 answers to it.

0.5 MAJORITY-CLASS BASELINE (before training anything)
    Over the training day, compute the share of decisions taken by the single most common
    action, globally and per SelectContext. If BC hits 80% and "always END" hits 75%, I have
    learned nothing. Print the table. It becomes a flat reference line in TensorBoard forever.

0.6 LABEL QUALITY — WHOSE BEHAVIOR ARE WE CLONING?
    The daily dump is every episode, not top episodes. Report the distribution of player
    strength if any score/rating field exists. Recommend and justify a filtering policy.
    Prior from Orbit Wars 49th place (the only IL-only top-50 solution there): clone
    winner-side states only; solo clones beat naive blends because style coherence matters;
    cloneability tracks a teacher's PREDICTABILITY, not their rating. Test that here, do not
    assume it.

0.7 THE SUBMISSION ENVELOPE — currently UNRECORDED and it gates Phase 7
    Read https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708810 and the
    competition rules, and report: (a) submission bundle SIZE CAP in MB, (b) whether both
    agents in a match share ONE container (which halves the 1.6 vCPU / 8 GB), (c) whether
    network egress is blocked, (d) whether cold-start/import time counts against the 600 s
    overage. If any is not stated publicly, say "unknown" and design for the pessimistic case.
    A parameter budget with no file-size number attached is not a budget.

=== ###################################################################### ===
=== PHASE 0 HARD STOP — MANDATORY HUMAN REVIEW GATE. THE RUN ENDS HERE.   ===
=== ###################################################################### ===

This is a HARD STOP, not a checkpoint you may pass with a note in the log. Phase 0 is the
entire scope of this session. Everything below this block is a SPECIFICATION YOU ARE READING
FOR CONTEXT so your Phase 0 recommendations are informed — it is NOT a work order yet.

DO NOT, under any framing:
  - write, scaffold, or stub anything under models/, train/, data/preprocess.py,
    data/dataset.py, agents/il_agent.py, utils/device.py, or export/
  - begin Phase 1, or "just start Phase 1 since it is only prose"
  - refactor, rename, or move any existing repo file
  - launch any training run, of any length, including a 5-minute "smoke" run
  - prepare scaffolding "while waiting for approval"
  - infer approval from anything other than the explicit token in the APPROVAL PROTOCOL below

WHAT YOU MAY WRITE IN THIS SESSION, and nothing else:
  - notes/phase0_discovery_report.md  (the deliverable)
  - throwaway measurement scripts under scripts/probe/  — read-only, no repo mutations
  - the charts required by 0.5, 0.6, and Q21, into reports/figures/
  - notes/phase0_open_items.md if the answer list outgrows the report

WHY THE STOP EXISTS. I have 15 days and 5 submissions/day. The expensive failures in this
project are all upstream decisions — an encoder that leaks opponent-private state (Q2), a
parameter budget that eats the search budget (1.2 / Q28), a data split that leaks
within-match state (2.2), a label filter that clones the average losing ladder player (Q1).
Each of those is cheap to fix at the Phase 0 boundary and expensive to fix after Phase 2 has
been written on top of it. One review cycle here costs me twenty minutes. Catching any one of
those four after a training run costs me a day.

--- THE REPORT MUST OPEN WITH THIS SIGN-OFF BLOCK, FILLED IN ---

# PHASE 0 SIGN-OFF — awaiting approval

**One-sentence training-time estimate:** <the Phase 6 sentence, at the very top, per Deliverable 1>

**Recommendation in one sentence:** <GO / GO WITH CHANGES / DO NOT PROCEED, and why>

## A. Blocking questions — all must be RESOLVED or explicitly ESCALATED
| ID  | Question (short) | Answer | Rests on | Status |
|-----|------------------|--------|----------|--------|
| Q1  | Winners / all / score-filtered | | | RESOLVED \| UNKNOWN \| NEEDS RAMI |
| Q2  | Does the replay leak opponent-private info | | | |
| Q7  | Share of trivial decisions | | | |
| Q18 | Majority-class baseline | | | |
| Q24 | Matches needed for Rung 2 + ladder days-to-confidence | | | |
| Q28 | Fallback if BC < heuristic; BC-as-search-prior viable? | | | |
| Q31 | Bundle size cap, in MB | | | |
| Q38 | Rung-2 win-rate of the QUANTIZED model (plan, not number, at Phase 0) | | | |

Status legend: RESOLVED = measured, with the measurement cited. UNKNOWN = I looked and it is
not knowable yet; state the pessimistic design assumption you will proceed under.
NEEDS RAMI = it is my call, not yours; it appears in section C below.

## B. Stop conditions — state each as TRIPPED or CLEAR, with the evidence
| Stop condition | State | Evidence |
|----------------|-------|----------|
| cabt not steppable (native or emulated) | | sim.lib path, steps/sec |
| Replay JSON insufficient for first-person reconstruction | | |
| Any OPPONENT-PRIVATE field reaches the encoder (Q2) | | field classification table |
| Any projected training run > 1 hour | | Phase 6 arithmetic |
| Bundle cap smaller than fp32 checkpoint (Q31) | | cap MB vs checkpoint MB |
| Any need to download data or hit the network | | |
If ANY condition is TRIPPED, the recommendation line is DO NOT PROCEED and the report says
what would clear it. Do not route around a tripped condition.

## C. DECISIONS I AM BEING ASKED TO APPROVE
For each, give: the recommendation, the one-line reason, the strongest counterargument, and
what you would build if I said the opposite. Maximum one short paragraph each. These are the
items where my answer changes what you write in Phase 2:
  C1  Architecture: attention/DeepSets vs flat MLP; edge attention in or out (1.1)
  C2  Parameter budget: the number, with the x86 latency and the surviving search budget (1.2)
  C3  Masking at TRAINING time: masked loss vs unmasked (1.3, and the OW-1st counter-evidence)
  C4  Deck policy: filter to the frozen control deck vs condition on a deck embedding (1.4)
  C5  Label filter: plain vs winners-only vs advantage-weighted (Q1/Q5, E2)
  C6  History: none vs hand-engineered summary vs attention over logs (Q17, E3)
  C7  Standalone BC agent vs BC-as-action-ordering-prior inside the existing search (Q28)
  C8  Checkpoint selector, and the per-epoch proxy standing in for Rung 2 (Phase 4, Q42)
  C9  Any place your recommendation drifts toward a hand-written scorer (see ROLE) — flag it
      loudly here or not at all
Where a decision is genuinely mine and the evidence does not settle it, say "NEEDS RAMI" and
give me the two options with the trade, rather than picking quietly and burying it in prose.

## D. WHAT I WILL BUILD IF YOU SAY GO
The Phase 2 file list, one line each, with the order and a rough time estimate. This is the
thing I am approving. If it differs from Phase 2 as written below, say where and why.

## E. WHAT WOULD CHANGE MY MIND
The part (c) of the three-part format, aggregated: the two or three measurements that, if
they came out differently, would most change the plan. This tells me where to spend my review.

--- APPROVAL PROTOCOL ---

After writing the report, print exactly:

    PHASE 0 COMPLETE — STOPPED FOR REVIEW. Report: notes/phase0_discovery_report.md
    Awaiting: "APPROVED — proceed to Phase 1" plus rulings on C1-C9.

Then end the turn. Do not continue. Do not ask a follow-up question that invites me to say
"yes go ahead" in passing — the only thing that resumes this pipeline is the literal string
APPROVED — proceed to Phase 1, accompanied by my rulings on the section C items.

Partial approval is normal and expected: I may approve C1-C4 and send C5 back for more
measurement. In that case do exactly the re-measurement asked and stop again. Do not treat a
partial approval as a general licence to proceed.

If I answer ambiguously, ask. An ambiguous approval is not an approval.

=== ###################################################################### ===
=== EVERYTHING BELOW IS CONTEXT FOR THE PHASE 0 REPORT, NOT A WORK ORDER. ===
=== It becomes the work order the moment I write the approval string.     ===
=== ###################################################################### ===

=== PHASE 1 - DECISIONS. State each in one paragraph with the trade-off, then commit. ===
1.1 ARCHITECTURE — argue this, do not just agree with me.
    My prior: a transformer, because it is the most capable model per parameter and because
    people use it to get history. I believe both halves may be wrong here. Address:
      - The real reason attention fits is NOT history. It is that the observation is a set of
        variable-length permutation-invariant zones (hand, bench, discard, prizes) and the
        ACTION SPACE is a variable-length list of legal options. A flat MLP over concatenated
        features (CleanRL / PufferLib style) needs a fixed slot layout and imposes a false
        ordering on those sets. Argue attention/DeepSets vs flat MLP on THOSE grounds, with
        the parameter count and x86 CPU latency of each.
      - Consider EDGE ATTENTION before plain self-attention. Orbit Wars 6th place got his
        single biggest gain from injecting a pairwise source->target feature vector into both
        the attention scores and the output: scores = q@k + q_edge@lin(edge). The PTCG
        analogue is attacker->target: damage, is-it-a-OHKO, weakness/resistance, energy cost,
        prize value. These are exactly the quantities a scoring head otherwise has to
        rediscover from card IDs. Cost it and recommend for or against.
      - Is history actually needed? obs_dict hides the opponent's hand and both deck orders,
        so a memoryless policy is fitting a belief-free approximation. Options: (a) feedforward
        on obs["current"] only, (b) hand-engineered history summary features (cards revealed,
        energy attached so far, opponent archetype signature), (c) attention over obs["logs"].
        Recommend one, state what the others cost. Do not add an LSTM.
      - Do I actually need HuggingFace transformers? For a <20M-param custom set encoder over
        card embeddings, a hand-rolled nn.TransformerEncoder or a ~100-line attention block is
        likely smaller, faster on CPU, and lighter in the bundle. If you recommend HF anyway,
        justify it against bundle size (Phase 7) and CPU cold-start (0.7d).

1.2 PARAMETER BUDGET — a hard constraint derived from a known envelope, not a target.
    THE EVALUATION MACHINE IS: CPU only, no GPU, 1.6 vCPU, 8 GB RAM, 600 s shared overage per
    match, 2000 s runTimeout across both players. Design to these numbers, not to my laptop.
      - Budget 2M-15M total parameters, hard ceiling 50M. At ~200 decisions/match and ~20
        options per decision, 15M burns ~2% of the match clock; 300M burns ~40% and ~1.2 GB
        to load. My original "under 300M" was off by more than an order of magnitude.
      - 8 GB RAM is NOT the binding constraint — latency and bundle size are. Do not use RAM
        headroom to justify a bigger model.
      - THE POINT OF STAYING SMALL IS WHAT THE HEADROOM BUYS. The other 98% of the clock is
        exactly the budget a determinized-MCTS or one-ply search layer needs, with this policy
        as the action-ordering prior. Four independent signals in my notes say search wins here.
        Treat any parameter count that eats the search budget as disqualified regardless of
        offline accuracy. See Q28.
      - Corroborating datapoint at my scale: Orbit Wars 8th place (top-10 for under $200) used
        a 4-layer transformer at d_model=192 and found explicitly that BIGGER MODELS LEARNED
        SLOWER WITH NO BENEFIT at that compute budget.
      - 1.6 vCPU is a FRACTIONAL CGROUP QUOTA, not 1.6 cores to parallelise across. Run
        single-threaded: torch.set_num_threads(1), and export OMP_NUM_THREADS=1 and
        MKL_NUM_THREADS=1 BEFORE the torch import. Multi-threading gets throttled at CFS
        period boundaries and produces tail-latency spikes, not throughput. Sweep {1,2} to
        confirm. Never size threads from os.cpu_count() — inside the container it reports the
        HOST's cores. Ground truth is /sys/fs/cgroup/cpu.max (reads "160000 100000" here).
      - Score all N legal options in ONE batched forward pass, never N sequential ones.
      - Report vocab size x embedding dim explicitly; the card table will dominate the count,
        and that fact resurfaces in Phase 7 (quantization) and Phase 8 (Muon).
      - Benchmark on x86, single-threaded, fp32, with the option-scoring batched. If you want
        >15M, prove it with a measured x86 latency number AND show the search budget survives.
        An argument is not sufficient.

1.3 ACTION MASKING — mandatory at inference, and a gradient question during training.
    Read Huang & Ontanon, "A Closer Look at Invalid Action Masking in Policy Gradient
    Algorithms" (arXiv:2006.14171): masking changes the gradient, it is not a filter.
      Pattern A - fixed-size logit vector over a global action space, illegal logits -inf.
      Pattern B - score-each-option head: embed each legal option, score independently,
        softmax over the variable-length set.
    Recommend Pattern B unless you can show why not: the engine hands you only legal options,
    the list is variable-length, and Pattern B makes masking structural rather than an
    afterthought. Loss is cross-entropy over the legal set with the chosen index as label.
    NEVER allow a path that can emit an out-of-range index — IndexError means INVALID means
    an instant loss.
    COUNTER-EVIDENCE you should weigh rather than ignore: Orbit Wars 1st place found action
    masking made his model WORSE DURING TRAINING (his theory: it stopped the net internalising
    the physics), and he reinstated it for fine-tuning and test time only. That cuts against
    Huang & Ontanon. For PTCG, test-time masking is non-negotiable; whether the training-time
    loss should be masked is a live question — state your call and your reasoning.
    Handle maxCount > 1 (multi-select) explicitly: autoregressive picking with re-masking after
    each pick vs one combinatorial choice. Pick one and say why.
    Handle minCount == 0 (decline with []) as a real, learnable action.
    The deck-selection step (obs["select"] is None) is NOT a policy decision — route it to a
    fixed 60-ID deck constant, never to the model.

1.4 DECK POLICY — one frozen control deck, plus a registered deck experiment. Both, not either.
    CONTROL: freeze ONE 60-card deck for the main line so deck choice does not confound the
    learning curves. Start from the official Mega Lucario ex sample list. Every number in
    Phase 4 that is not explicitly labelled E1 is measured on this deck.
    EXPERIMENT E1 (see the register below): three BC models, identical config, differing only
    in training-data deck selection —
      E1a  episodes using the frozen control deck
      E1b  episodes using the MOST COMMON deck in the training day
      E1c  episodes using a RANDOMLY CHOSEN deck of comparable episode count
    Report Rung-1 accuracy and Rung-2 win-rate for each, and SHIP A CHART. The question E1
    answers is whether deck-specific data beats deck-volume data, which decides whether the
    frozen-deck control is costing me anything.
    Then produce a ranked shortlist of alternates, one line of evidence each, drawn from the
    leaderboard-meta notebook and the official rule-based samples:
      Mega Lucario ex - Dragapult ex - Iono's Bellibolt ex - Mega Abomasnow ex - Greninja -
      N's Zoroark - Raging Bolt - Mega Lopunny ex - Mega Absol - Archaludon - Starmie
    Also state: does the model SEE which deck it is piloting as an input? If training data
    spans many decks and deck is not conditioned on, the policy averages incompatible
    strategies. Recommend (a) filter to the frozen deck, or (b) condition on a deck embedding,
    with the data-volume cost of (a). Note that Orbit Wars 49th place found CONDITIONAL BC
    (one net, teacher one-hot + recency scalar) matched or beat every dedicated solo clone on
    roughly a quarter of the data each — that is the strongest available prior for (b).

1.5 TRAINING LENGTH — a floor, not a budget.
    Train a MINIMUM of 2 full epochs. Do not early-stop on validation loss alone, and do not
    report a 1-epoch model as a result. Reason: this checkpoint is a PPO initialisation
    (stage 3). An undertrained BC policy produces a weak, badly-calibrated init, and the
    damage is invisible in offline metrics — it shows up as PPO failing to improve later,
    at which point I will blame PPO. If 2 epochs is unaffordable per Phase 6, that is a
    finding to report and a reason to shrink the model, not a reason to train for one.
    State the epoch count you will run and what would make you run more.

=== PHASE 2 - BUILD. One module per file, each independently testable. ===
Constraint that shapes everything: the dataset is ~40 GB across two days (train 2026-07-26:
4554 episode JSONs; held-out eval 2026-07-27: 4430 JSONs), stored as a DIRECTORY OF
PER-EPISODE JSON FILES — not parquet, not a single jsonl. It does not fit in RAM. Do not
write a loop that assumes one in-memory tensor.

2.1 data/preprocess.py — ONE-TIME pass: walk the episode JSONs, reconstruct each acting
    player's first-person view at every decision point, emit
    (encoded_obs, encoded_legal_options, chosen_index, context, game_id, deck_id, seat,
     outcome, turn_index) into memory-mapped shards (np.memmap or Arrow/Parquet). Encoding
    must NOT be redone every epoch. Report rows produced and bytes on disk.
2.2 data/dataset.py — IterableDataset or memmap-backed Dataset + a collate_fn that pads the
    variable-length option sets and returns the padding mask alongside. Split by game_id
    within a day, and hold out the ENTIRE 2026-07-27 day as the eval set — a random row split
    leaks within-match state and will inflate accuracy. Justify the split explicitly against
    the alternatives (random row, random game, temporal-within-day, whole-day holdout) and
    say which one you would use if E-series experiments need a faster inner loop.
2.3 models/encoder.py — versioned encode_obs (v1, v2, ...) so learning curves stay comparable.
    A shared card-embedding table reused across hand/bench/discard/active (same card -> same
    vector everywhere) plus a learned zone embedding. Per-Pokemon features riding alongside
    the card ID: HP normalised, energy by type, status, tool attached. Scalars: prizes both
    sides, deck counts, turn number, phase flags, seat. Freeze and ship the card vocab; define
    behaviour for an unseen card ID at inference (an OOV row, never a crash).
2.4 models/policy.py — the encoder plus the option-scoring head from 1.3. The head must expose
    CALIBRATED PROBABILITIES over the option set, not just an argmax — this is what a search
    layer needs from it (Q28) and it is free to specify now.
2.5 train/train_bc.py — AdamW, warmup + cosine, gradient clipping, checkpoint-on-best-<metric
    named in Phase 4>. Deterministic seed. Device from 2.7 — never hardcode "mps" or "cuda".
    Minimum 2 epochs per 1.5. Optimizer choice: AdamW for stage 1, see Phase 8.1 for why NOT
    Muon here.
2.6 agents/il_agent.py — the inference wrapper satisfying the contract found in 0.3. Never-crash
    contract: wrap everything in try/except with a legal fallback ([0] for a battle step, the
    fixed 60-ID deck for the deck step). Load the checkpoint once at import, run float32 on
    CPU, cap its own thinking with time.monotonic().
    UPGRADE THE FALLBACK FROM try/except TO A TIME-TIERED LADDER. Orbit Wars 1st place played
    normally until 1 s of overage remained, then switched to a much faster small model to
    finish; it converted 100% of its winning positions. The PTCG analogue with a 600 s shared
    overage: normal path -> reduced-search path -> raw-policy-argmax path -> legal fallback,
    each triggered by remaining overage. Implement the tiers even if the top tier is the only
    one used today, because the tiering is what makes a later search layer safe to add.
2.7 utils/device.py — a single resolve_device(override=None) helper, and the ONLY place in the
    codebase that decides a device. Rule: MPS if torch.backends.mps.is_available() else CPU.
    There is no CUDA path — this laptop is Apple Silicon and the evaluator is CPU-only, so a
    cuda branch is dead code that will rot. Requirements:
      - The same source tree must run unmodified in both places. No environment sniffing beyond
        is_available(), no "if kaggle" flags.
      - Overridable by CLI flag / env var so I can force CPU on the laptop and time exactly
        what the evaluator will do.
      - Log resolved device, dtype, torch version once at startup into the run directory.
      - Checkpoints saved device-agnostically (state_dict moved to CPU before torch.save) and
        loaded with map_location. An MPS-tagged checkpoint that fails to load on a CPU-only
        evaluator is a silent submission failure.
      - agents/il_agent.py calls this too and resolves to CPU on the evaluator. Assert it: the
        inference path must be exercised under a forced-CPU run before any submission.
2.8 export/bundle.py — NEW. Takes a checkpoint and produces the exact artifact that gets
    submitted: quantized or not per Phase 7, vocab frozen, thread env vars set before the torch
    import, and a printed manifest of (bytes on disk, param count, dtype, encoder version,
    git SHA, measured x86 cold-start seconds). A submission you cannot describe in one table
    is a submission you cannot debug.

=== PHASE 3 - PRECISION AND DEVICE (resolved; do not re-derive) ===
  - TWO EXECUTION CONTEXTS, ONE CODEBASE. Training on my laptop on MPS. Inference on the
    Kaggle evaluator on CPU, no GPU. The same source must be right in both without edits —
    that is what 2.7 exists for. "Works on MPS" and "works on CPU" are two tests, not one.
  - The Apple Neural Engine is NOT reachable from PyTorch. MPS targets the GPU only; the ANE
    needs CoreML/ExecuTorch export. Moot regardless — eval is CPU-only.
  - MEASURED, 5M-param set-transformer, forward+backward: MPS fp32 = 3.2x CPU fp32.
    MPS bf16 (direct dtype, no autocast) = 1.08x MPS fp32. DECISION: TRAIN IN FP32 ON MPS.
    bf16 is below my 1.3x bar; the fallback complexity is not worth 8%.
  - MY OWN COUNTERARGUMENT, AND THE ANSWER. I noted that even at 1.08x, bf16 halves activation
    memory and so buys a bigger batch or a bigger model, which might train better. That is a
    MEMORY argument, and it only pays if I am memory-bound. At 2-15M parameters on unified
    memory I am not: the binding constraint is dataloader throughput over 40 GB of per-episode
    JSON, which bf16 does nothing for. TEST IT RATHER THAN ARGUING IT — report GPU/unified
    memory at the largest fp32 batch that fits. If fp32 tops out below batch 4096, revisit;
    if not, the counterargument is dead and bf16 stays off. Note also that a larger batch is
    not free accuracy — it changes the effective learning rate and can hurt at this data scale.
  - MPS is a training-only convenience. Nothing about correctness may depend on it: no MPS-only
    op, no MPS-only dtype in a saved artifact, no code path CPU cannot take. If an op silently
    falls back to CPU, note it — that is a perf cliff.
  - Whatever I train in, the SUBMITTED model runs on CPU. Benchmark inference in that
    configuration, not the training one.

=== PHASE 4 - EVALUATION. Three rungs. Rung 1 alone is not evidence the model works. ===
  Rung 1 (offline) — top-1 and top-3 action-match on the held-out DAY, reported per
    SelectContext, ALWAYS printed next to the 0.5 majority-class baseline. An aggregate number
    with no baseline beside it is not a result.
    Break out accuracy by len(options) == 1 / 2-4 / 5+ separately, always.
    Calibration reference from Orbit Wars 2nd place's IL phase: launch AP 83.80, target
    acc@1 82.12, acc@2 94.99 on 45k held-out samples — good enough for top-10 there, and
    still beaten outright by from-scratch RL. Use it to judge whether my numbers are in a
    sane range, not as a target.
  Rung 2 (behavioural) — wire il_agent into scripts/benchmark_agents.py and run a round-robin
    against every existing agent in the repo through the cabt environment. Report win-rate with
    a confidence interval AND the number of matches. This is the number I care about, and I
    expect it to disagree with Rung 1. The engine runs natively on macOS as of 2026-06-30,
    so the match count should be affordable; if the measured steps/sec from 0.2 says
    otherwise, say so with the arithmetic (Q24) rather than quietly running too few.
  Rung 3 (sanity) — play 5 full matches with verbose logging and READ the transcripts. Does it
    ever decline an optional selection (minCount==0)? Does it retreat? Does it attack when it
    should? A policy that never crashes and never wins is the specific failure mode here.
  CHECKPOINT SELECTION — the rule, not a suggestion:
    Offline action-match is a PIPELINE-CORRECTNESS CHECK, NOT A SELECTION METRIC. Orbit Wars
    49th place, the only IL-only top-50 solution there: "Better imitation metrics do not mean
    more wins. Fidelity and playing strength diverged every time I tested them." He also found
    his higher-local-eval checkpoint did WORSE on the ladder.
    So: select on Rung-2 win-rate if the match budget allows it. If it does not, say so
    explicitly and name the second-best selector you are falling back to and its known bias.
    Never select on val loss alone. Log all three and show where they disagree — that
    disagreement is itself a reportable result and belongs in the Strategy writeup.
  ALTERNATIVE PER-EPOCH SIGNALS (I asked for these; pick at least two and justify):
    - a fixed 50-match gauntlet vs one frozen opponent, run every epoch (cheap, noisy, but a
      real behavioural signal)
    - top-1 accuracy restricted to len(options) >= 5, i.e. the decisions that actually decide
    - expected calibration error over the option set (matters for the Q28 search-prior use)
    - policy entropy per SelectContext (an entropy collapse is an early warning for stage 3)
    - agreement rate with the existing search agent's top choice (cheap proxy for strength)
  Every comparison in this phase ships a chart. See CONSTRAINTS.

=== PHASE 5 - LOGGING ===
  TensorBoard now (SummaryWriter, log dir under runs/), behind a thin logger interface with
  one implementation, so a Weights & Biases backend can be swapped in later at the same call
  site. No wandb import today. Log at a configurable step interval:
  train loss, val loss, top-1 accuracy overall and per SelectContext, top-1 accuracy for
  len(options) >= 5, learning rate, grad norm, steps/sec, policy entropy, and the
  majority-class baseline as a flat reference line.
  Every run directory records: git SHA, full config, encoder version, resolved device+dtype,
  seed, and the Phase 6 projection vs the actual wall clock.

=== PHASE 6 - TIME AND COST. Answer with numbers before starting a long run. ===
  From the 0.4 numbers and the row count from 2.1, project wall-clock hours per epoch and for
  the full >=2-epoch run. State it as a range with the assumption that dominates it.
  Anchor: a competitor finished top-60 with pure IL on ~21k games in 3-4 h on a single H200.
  I have ~9k games and no H200, but a far smaller model.
  If the projection exceeds 8 hours per epoch, THE DATALOADER IS THE BOTTLENECK, NOT THE MODEL
  — profile and fix that before training. Report the split of time between data loading and
  compute; if data loading is over 50%, that is the finding.
  Report the projection and STOP for my approval before launching anything over 1 hour. This
  is a SECOND hard stop, independent of the Phase 0 gate, and it survives any approval I gave
  at Phase 0 — approving the plan is not approving a specific long run.
  Tell me, in one sentence at the top of the report, how long this training run will take.

=== PHASE 7 - QUANTIZATION AND THE SUBMISSION BUNDLE (new in v2) ===
  Gated on 0.7. If the bundle size cap is unknown, design for the pessimistic case and say so.
  7.1 Measure first. Report fp32 checkpoint bytes on disk and what fraction of the cap it is.
      If it is under ~25% of the cap, QUANTIZATION IS NOT NEEDED FOR SIZE and you should say
      so plainly rather than doing it because it is interesting. At 15M params fp32 that is
      ~60 MB, which may already fit.
  7.2 The other reason to quantize is SPEED, not size. int8 on linear layers is a real CPU
      inference win and is separable from weight-size compression. Evaluate the two motives
      separately and report which (if either) applies here.
  7.3 Tooling: torchao (pytorch/ao), the PyTorch-native quantization library — it supports
      post-training quantization and QAT and composes with torch.compile. Confirm it installs
      and runs inside the Kaggle submission environment BEFORE depending on it; an extra wheel
      in the bundle has a size cost of its own.
  7.4 PTQ first, QAT only if PTQ loses strength. Post-training quantization costs nothing to
      try. QAT costs a retrain. Do not start with QAT.
  7.5 THE bf16 -> int8 QUESTION I ASKED. The honest state of the literature: bf16's advantage
      is dynamic range and training stability, and int8 inference reaches near-full-precision
      accuracy through CALIBRATION or QAT — not through having trained in bf16. There is no
      established result that bf16 training specifically improves int8 deployment robustness.
      The mechanism that does work is QAT, where the model trains with simulated quantization
      in the loop. Since Phase 3 already settled on fp32 training, this is moot for me — but
      state it in the report so the question is closed rather than recurring.
  7.6 Precedent worth reading, not copying: Orbit Wars 1st place fit 200M params into a 100 MiB
      cap with 4-bit NormalFloat, group size 128, one fp16 scale per group, retaining ~40%
      head-to-head win-rate against the unquantized model; int8 on linears separately for
      speed; 3-bit was net negative. That is a 200M-param problem. At 15M I should not need
      any of it, and if I find myself reaching for 4-bit the real error was upstream in 1.2.
  7.7 NO QUANTIZED MODEL SHIPS WITHOUT A RUNG-2 WIN-RATE. Size and latency are not evidence of
      strength. See Q38.

=== PHASE 8 - RL-READINESS: what stage 1 must not foreclose (new in v2) ===
  Do NOT build stages 2 or 3. Do state, in one paragraph each, what stage 1 owes them.
  8.1 OPTIMIZER. Use AdamW for BC. Muon is a stage-3 question.
      CORRECTED 2026-08-01 — the first version of this section gave the WRONG REASON. It said
      my parameter count is dominated by the card-embedding table, which Muon does not touch.
      The arithmetic does not support that. Muon covers the 2D hidden matrices, 12*L*d^2, while
      the embedding table is only V*d, so the Muon-covered fraction is 12*L*d^2 / (12*L*d^2 +
      V*d + ...). At V = 631 (a hard floor: the highest card ID in cabt.py's own default 60-card
      deck) through V = 3000, Muon covers 73-95% of the model, not a minority. The card table
      only dominates above V of roughly 10,000. See adamw-vs-muon-ptcg-bc.png.
      THE REASONS THAT SURVIVE, in order of weight:
        - No evidence base. Muon's documented wins are LLM pretraining and RL; I found none for
          behavior cloning from replays. That is not proof it fails, it is absence of a reason
          to spend the risk budget here.
        - The gain is sweep-conditional. PufferLib's own account is that Adam->Muon "initially
          didn't seem to make much of a difference" and the step-change only appeared once they
          ran FULL SWEEPS. I have 15 days and no sweep budget, so I would be buying the part of
          Muon that does not pay without a sweep.
        - It adds hyperparameters (LR ratio between the Muon and AdamW groups, Newton-Schulz
          step count, the param-group split itself) and per-step orthogonalization cost, on a
          schedule where every new knob is a place to lose a day.
      STILL REPORT the embedding fraction (Q34) — it is load-bearing for quantization and for
      the Q12 ID-vs-name choice even though it no longer decides the optimizer.
  8.2 ENTROPY AND CALIBRATION. A BC policy trained to hard argmax with no label smoothing is a
      near-deterministic init that PPO cannot explore out of. Decide label smoothing /
      temperature NOW and log per-context entropy from epoch 1. Related: Orbit Wars 8th place's
      biggest single win was a KL PENALTY TO A HAND-SET NON-UNIFORM PRIOR instead of an entropy
      bonus — and a BC policy is precisely such a prior. That is the concrete stage-3 use of
      this checkpoint beyond weight initialisation; make sure the checkpoint can serve it
      (frozen copy loadable independently, calibrated probabilities, same encoder version).
  8.3 HYPERPARAMETER SEARCH. Not now. When stage 3 arrives, the tool is Protein, PufferLib's
      cost-aware sweep algorithm (a heavily modified CARBS: it models score AND wall-clock cost
      as Gaussian processes and searches the Pareto frontier). It is the right tool precisely
      because my budget is a laptop. What stage 1 owes it: a config object that is fully
      specifiable from a flat dict, and a single scalar objective that a sweep can read from
      the run directory. Build those two things now; they cost nothing.
  8.4 REWARD SIGN, recorded now so it is not rediscovered later. cabt ships -1/0/1. Orbit Wars
      8th place argues for 0/1/2 instead, because negative terminal rewards under gamma < 1
      incentivise DELAYING a loss, which shows up as pathologically long games and a throughput
      collapse. All four Orbit Wars RL solutions hit some version of this. Not a stage-1
      concern; note it in the report so stage 3 starts from it.
  8.5 COMPOUNDING ERROR. BC only ever sees states the demonstrated policy reached. Name whether
      DAgger or self-play fine-tuning is in scope, and if out of scope for now, say what the
      accepted cost is. See Q30.

=== EXPERIMENT REGISTER — every experiment gets an ID, a hypothesis, and a chart ===
  E1  Deck: control deck vs most-common deck vs random deck (1.4). Hypothesis, three runs,
      one chart of Rung-1 and Rung-2 side by side.
  E2  Label filter: plain BC vs winners-only BC vs advantage-weighted BC (Q5). Pick the two
      most different, not all three.
  E3  History: none vs hand-engineered summary features (1.1). Chart accuracy and x86 latency
      on the same axes — the point is the trade, not either number alone.
  E4  Quantization: fp32 vs int8, size / x86 latency / Rung-2 win-rate (Phase 7).
  Each experiment: one hypothesis sentence, one config diff, one chart, one verdict sentence.
  Anything that is not on this register is not an experiment, it is a run.

=== CONSTRAINTS ===
  - EVERY COMPARISON SHIPS A CHART. Any time you put two or more numbers side by side to
    support a conclusion — model sizes, epochs, decks, precisions, agents, accuracy vs
    win-rate — produce a matplotlib figure saved to reports/figures/<experiment_id>_<name>.png
    and reference it by filename in the report. A table in a terminal is not a chart. This is
    not decoration: the Strategy track is 40% of the prize pool and it wants the narrative.
  - EVERY EXCITING NUMBER NEEDS A CONTROL. Orbit Wars 49th place's coding agent declared an
    agent "best" off a 12-game sample and called a submission a champion-beater without
    checking the opponents were 100+ ELO weaker. State the control beside every claim.
  - Never re-download the Kaggle data. Both days are on disk under data/episodes/splits/.
  - Never read, print, or reconstruct ~/.kaggle/kaggle.json.
  - The simulator's behaviour is definitionally correct for this competition; official paper
    rules are advisory. Do not "fix" the sim.
  - No per-move wall clock, but there is a 600 s shared overage per match and a 2000 s total
    runTimeout across both players. Budget against those, not against a 1-second-per-move
    figure — that number belongs to a different competition and is wrong here by ~600x.
  - Every artifact versioned: encoder version, checkpoint, config, git SHA in the run directory.
  - No rule-based agent as the deliverable. See ROLE.
  - The Phase 0 hard stop outranks every other instruction in this prompt. If any line below it
    seems to authorise starting work, it does not; it is describing what happens after approval.

=== OPEN QUESTIONS — answer these IN WRITING in the Phase 0 report ===
Format for each: (a) the answer, (b) the evidence or measurement it rests on, (c) what you
would have to see to change your mind. "Unknown" is an acceptable answer; a guess presented
as a finding is not. Questions marked [BLOCKING] must be resolved before Phase 2 — which now
means: before I will approve the gate, not before you privately decide to continue.

--- A. Label quality: what are we actually cloning? ---
  Q1  [BLOCKING] Winners only, both players, or a score-filtered subset?
      Why: the daily dump is EVERY episode on a ladder of mixed skill. Cloning the
      unconditional action distribution fits the average ladder player, who loses ~50% of the
      time by construction.
      How: report any per-player score/rating field and its distribution; compare candidate
      policies on rows retained and on the Rung-1 baseline gap.
  Q2  [BLOCKING] Does the replay log leak information the live agent will not have?
      Why: a replay is written by an omniscient observer. Reconstruct from the log rather than
      from the acting player's obs_dict and you may hand the model the opponent's hand, deck
      order, or prize contents. It will use it, score brilliantly offline, and collapse on the
      ladder — and no offline metric will tell you. Highest-probability failure here.
      How: classify every field you feed the encoder as PUBLIC / OWN-PRIVATE / OPPONENT-PRIVATE.
      Assert in code that no OPPONENT-PRIVATE field reaches the encoder.
  Q3  Are episodes ending in INVALID, crash, timeout, or concession present in the dump?
      Why: the final actions of a crashed episode are artifacts, not decisions. Training on
      them teaches the model to imitate a bug.
  Q4  How concentrated is the dump by submitting agent?
      Why: if one prolific agent contributed 20% of episodes you are largely cloning that agent,
      including the specific weaknesses the ladder already knows how to beat.
  Q5  Weight rows by outcome, or filter by it?
      Why: filtered BC throws away half the data; advantage-weighted BC keeps it and downweights
      losing trajectories; weighting by within-episode signal (prize differential over the next
      k turns) is finer-grained than a terminal label. Argue plain vs filtered vs
      advantage-weighted on THIS dataset size, pick one, register it as E2. Do not build all three.
  Q6  How much of the row count is actually distinct?
      Why: ladder episodes with a frozen deck produce enormous numbers of near-identical
      early-game states. 1.8M rows may be 100k distinct situations, which changes Q19 and Q22.
      How: hash the encoded observation; report unique-row fraction, split at turn <= 3 vs > 3.

--- B. Decision structure: is one flat policy the right shape? ---
  Q7  [BLOCKING] What share of decisions is trivial, and does the loss reflect that?
      Why: if 85% of decisions are near-forced, cross-entropy is dominated by decisions that do
      not decide games and aggregate accuracy measures how well you predict the boring 85%.
      How: distribution of len(options). Accuracy for len==1, 2-4, 5+. Recommend whether to drop
      len==1 rows entirely and whether to reweight the loss per SelectContext.
  Q8  One shared model, or a shared trunk with per-SelectContext heads?
      Why: SETUP_BENCH_POKEMON and a mid-game attack choice are different problems sharing an
      observation. The strongest public agents here are per-context scoring functions.
      How: per-context row counts — any context too sparse to support its own head settles it.
  Q9  How are maxCount > 1 multi-select steps handled, and how common are they?
      Why: they break "one softmax over options". At 1% of decisions, a cheap heuristic; at 15%,
      a real autoregressive head, which changes the architecture.
  Q10 How often is minCount == 0 (declining legal), and how often do experts decline?
      Why: "return []" is a real action and the one a naive implementation is most likely to make
      unreachable. A policy that can never decline over-commits resources every game.
  Q11 Does seat matter — is going first an input feature?
      Why: first-player advantage is large in TCG. Without seat as an input the model averages
      two different policies.

--- C. Representation ---
  Q12 Card embeddings keyed by card ID or card NAME?
      Why: the same card exists at multiple IDs and deck legality is 4-per-NAME. Tying by name
      shrinks the vocab and shares strength across reprints; ID preserves ID-specific sim
      behaviour. Pick one, say what you lose.
  Q13 Does the option-scoring head share the card-embedding table with the observation encoder?
      Why: if not, the head scores an option without knowing the card in it is the same card it
      just encoded in hand. Sharing is nearly free and is the main reason Pattern B works.
  Q14 How are numeric quantities encoded — HP, damage, energy, prizes?
      Why: small models often do better with binned/embedded integers than raw scalars, because
      the boundaries here are sharp (is this a KO) rather than smooth. Argue; give bin edges.
  Q15 Is bench position semantically meaningful in this simulator?
      Why: decides whether the bench encoder is permutation-invariant or order-aware. Orbit Wars
      6th place got a real gain from RELATIVE-ONLY inputs with no absolute positions and no
      player IDs — build the invariance into the architecture rather than augmenting for it.
      How: check whether any option or effect references a bench index.
  Q16 OOV behaviour for a card ID unseen at training time?
      Why: frozen vocab + unseen card = KeyError = INVALID = instant loss. Define an OOV row and
      test it.
  Q17 Is there history in the input, and exactly what?
      Why: this is what my transformer instinct was really about. State the concrete choice and
      the parameter and x86-latency cost of the one you did not take. Register as E3.

--- D. Training dynamics ---
  Q18 [BLOCKING] Majority-class baseline, globally and per SelectContext?
      Why: the only number that makes an accuracy figure interpretable. Report it beside every
      accuracy metric, forever, including in TensorBoard.
  Q19 Does the row count justify the model size chosen in 1.2?
      Why: ~9k games is not a lot and Q6 may show the effective dataset is far smaller than the
      row count. Give params-per-distinct-row and say whether you are overparameterised.
  Q20 How much entropy should the final policy retain?
      Why: hard argmax with no label smoothing gives a near-deterministic init that stage-3 PPO
      cannot explore out of. Decide now; it constrains a later phase silently. See 8.2.
  Q21 Is the held-out day the same metagame as the training day?
      Why: the ladder meta shifts daily. If archetype frequencies differ between 2026-07-26 and
      2026-07-27, a held-out accuracy drop is distribution shift, not overfitting — and the
      correct response is opposite in each case.
      How: archetype/deck-composition frequency histogram for both days, side by side, as a chart.
  Q22 If held-out accuracy is far below train accuracy, how do you tell overfitting from
      metagame shift? Name the diagnostic before you need it.
  Q23 Is training reproducible on MPS?
      Why: MPS has known nondeterminism. If two same-seed runs diverge, every later A/B is noise.
      How: same seed, two runs, report the final-loss delta.

--- E. Evaluation and what happens after ---
  Q24 [BLOCKING] How many matches does Rung 2 need before a win-rate difference means anything?
      Why: win-rate at n=100 has a CI around +/-10pp. Most agent comparisons reported in this
      competition are inside their own noise floor.
      How: compute matches needed to detect 5pp at 95%, then say whether that is affordable at
      the steps/sec measured in 0.2 — which as of 2026-06-30 is a NATIVE macOS number, not an
      emulated one. If it is not affordable, that is a real finding and it caps how much Rung 2
      can be trusted; say so rather than running too few and reporting the number anyway.
      ALSO report the LADDER-side budget, which this question has never had: the hosts target
      48 matches per day per submission, with a 10% chance of drawing a random opponent, and
      staff state ratings only normalize "over hundreds of games". Convert that into days-to-
      confidence and say explicitly how late a checkpoint can be submitted and still be
      evaluated before the ladder closes 2026-08-16. A rating read at <100 games is not a
      rating; do not compare a fresh submission's score against a multi-day-old one.
      Two mechanics that belong in the eval design: an agent stuck in an infinite loop now
      LOSES BY TIMEOUT rather than drawing; and rating change is asymmetric — beating a much
      weaker opponent can pay +0 while losing to one costs heavily, which combined with the
      10% random matchmaking is a live downside risk once the agent climbs.
  Q25 Is the opponent pool in benchmark_agents.py representative of the ladder?
      Why: beating my own three baselines is not evidence about the ladder. Report what the pool
      has and what it lacks (notably strong search agents, and the archetypes in 1.4).
  Q26 Argmax or sampling at inference?
      Why: argmax is stronger in expectation but perfectly deterministic, so a bad line repeats
      identically against the same opponent. Low-temperature sampling trades strength for
      unexploitability. Interacts with Q20.
  Q27 Measured inference latency per decision on x86 CPU, and does batching the option-scoring
      forward pass help? Why: scoring N options is naturally one batched forward, not N. Getting
      this wrong is an easy 10x on the latency that has to fit the 600 s overage.
  Q28 [BLOCKING] What is the fallback if the BC agent scores BELOW the existing heuristic?
      Why: this is the documented outcome for the closest comparable attempt here — a solo
      competitor's RL+MCTS agent regressed to 580.6 mu while his own hand-written SearchScorer
      hit 660.5 mu on the same deck. Four independent signals in my notes say search wins here.
      The question I want answered: can the BC policy be an ACTION-ORDERING PRIOR or a leaf
      evaluator inside the existing search agent, rather than a standalone agent? That is
      plausibly the highest-value use of this model and it should be designed for now. State
      what the policy must expose — calibrated probabilities over the option set, not an argmax.
      Note also Orbit Wars 6th place: a greedy 2-step rollout beat principled CFR under a CPU
      budget by +30-40 LB points. Shallow targeted search is the shape to aim at, not deep search.
  Q29 Submission bundle size limit, and what fraction the checkpoint consumes? See Q31.
  Q30 BC only sees states the demonstrated policy reached. What happens the first time the agent
      drifts into a state no episode contains?
      Why: compounding error is the defining failure of behavior cloning and shows up as "plays
      well for six turns then does something insane."
      How: name whether DAgger or self-play fine-tuning is in scope, and if out, the accepted cost.

--- F. Deployment, precision, and the road to RL (new in v2) ---
  Q31 [BLOCKING] What is the submission bundle size cap, in MB, and what is in the bundle?
      Why: it caps the model, the vocab table, and any extra wheel (torchao, HF transformers).
      A parameter budget with no file-size number is not a budget. Supersedes Q29 as the
      blocking form of the same question.
      How: 0.7. If unstated publicly, design for the pessimistic case and say which case.
  Q32 Do both agents in a match share one container?
      Why: if yes, the 1.6 vCPU and 8 GB are HALVED and every latency number in Phase 1.2
      doubles against a budget half the size. This single unknown can invalidate the whole
      parameter budget.
  Q33 Does cold-start (process spawn + import torch + checkpoint load) count against the 600 s
      overage? Why: on 1.6 vCPU an import torch is seconds, not milliseconds. If it counts, a
      large dependency tree is a direct tax on thinking time, and that is an argument against
      HuggingFace transformers independent of accuracy.
      How: measure import-to-first-decision seconds in the amd64 container; report it.
  Q34 What fraction of parameters is the card-embedding table?
      Why: it decides three separate things — whether quantization helps (embeddings quantize
      well), whether Muon is worth anything later (it does not touch embeddings), and whether
      Q12's ID-vs-name choice is a rounding error or the main lever on model size.
  Q35 Is quantization needed at all, for size or for speed?
      Why: at 15M params fp32 the checkpoint is ~60 MB and may already fit. Doing it because it
      is interesting is how a working submission becomes a broken one. Answer size and speed
      separately (7.1, 7.2) and be willing to answer "no".
  Q36 Is torchao installable and functional in the Kaggle submission environment?
      Why: a quantization plan that depends on a wheel the evaluator will not install is not a
      plan, and the wheel itself eats bundle budget.
  Q37 Does training in low precision help deploying in low precision?
      Why: I asked this directly. My read of the literature is no — bf16 buys dynamic range and
      training stability, while int8 accuracy comes from calibration or QAT, and I found no
      established result that bf16 training specifically improves int8 robustness. Moot anyway
      since Phase 3 settled on fp32. Confirm or correct, cite a source, and CLOSE the question.
  Q38 [BLOCKING] What is the Rung-2 win-rate of the QUANTIZED model, not the fp32 one?
      Why: size and latency are not evidence of strength. Orbit Wars 1st place measured his
      quantized model at ~40% head-to-head against his unquantized one and shipped it anyway
      because the size cap forced it — but he KNEW the number. Never ship a quantized model
      whose win-rate you have not measured.
      At Phase 0 this is answerable only as a PLAN — say how and when the number gets measured
      and what it gates.
  Q39 What is the largest fp32 batch that fits in unified memory, and is the trainer
      memory-bound or dataloader-bound?
      Why: this is the empirical test of my own bf16-buys-a-bigger-batch counterargument
      (Phase 3). If fp32 already reaches a batch beyond where returns flatten, the argument is
      dead and bf16 stays off permanently.
  Q40 What fraction of training wall-clock is dataloading vs compute?
      Why: decides whether Phase 6 optimisation effort goes into the model or the pipeline. Over
      50% dataloading means the model size discussion is irrelevant to training time.
  Q41 Does the >= 2 epoch floor actually change anything measurable?
      Why: I asserted it from the PPO-init argument, not from evidence here. Report the metric
      deltas between epoch 1, 2, and 3 and tell me whether my floor is right, too low, or
      superstition. Chart it.
  Q42 Which per-epoch signal (Phase 4 alternatives) correlates best with Rung-2 win-rate?
      Why: I need a cheap epoch-level proxy for strength, since Rung 2 is too expensive to run
      every epoch. Whichever correlates best becomes the checkpoint selector when the match
      budget is short. This is a small measurement with a large downstream payoff.
  Q43 For E1, does deck-specific data beat deck-volume data?
      Why: it decides whether the frozen-deck control is costing me accuracy, and it is the
      cheapest test of whether to condition on a deck embedding instead of filtering.
  Q44 What does the existing search agent's top choice agree with the BC policy on, and where
      do they diverge?
      Why: the divergence set is the interesting one. If BC agrees with search on the easy 85%
      and diverges on the decisions that matter, that is either the model's value or its failure
      and I want to see the cases, not the aggregate. Directly informs Q28.
  Q45 What single scalar objective should a later Protein sweep optimise?
      Why: 8.3. A sweep needs one number written to the run directory. Naming it now costs
      nothing and naming it wrong later costs a sweep. It should almost certainly not be val loss.

Questions answerable only AFTER training (Q39-Q44 in part, Q41, Q42) get, at Phase 0, a
one-line MEASUREMENT PLAN instead of an answer: what you will measure, when, and what it
decides. A measurement plan is a valid Phase 0 answer. A guess dressed as a finding is not.

=== DELIVERABLES ===
  1. THE PHASE 0 DISCOVERY REPORT (markdown, notes/phase0_discovery_report.md), opening with
     the sign-off block above, with the one-sentence training-time estimate at the very top and
     every open question answered in the three-part format. THEN STOP. This is the entire
     deliverable of session A. Items 2-6 do not exist until I approve.
  2. After I approve: the modules in Phase 2, each with a smoke test.
  3. A README section documenting the discovered obs schema, the encoder version, and how to
     reproduce a run.
  4. The Phase 4 three-rung evaluation results table, with charts in reports/figures/.
  5. The experiment register with a verdict line per experiment.
  6. A submission bundle produced by export/bundle.py with its printed manifest, plus the
     forced-CPU inference run that proves it loads.

=== STOP CONDITIONS — halt and report, do not guess ===
  - THE PHASE 0 GATE ITSELF. Completing Phase 0 is a stop condition, not a milestone. It fires
    even when everything went well and every question resolved cleanly. "Nothing blocking came
    up" is not a reason to continue; it is the normal case.
  - The Phase 6 projection gate: any training run projected to exceed 1 hour, even after the
    Phase 0 approval.
  - The cabt simulator is not steppable locally — neither natively via libcg.dylib nor
    under amd64 emulation (0.2).
  - The replay JSON does not contain enough to reconstruct a first-person view.
  - Any OPPONENT-PRIVATE field is found reaching the encoder (Q2).
  - The bundle size cap turns out to be smaller than the fp32 checkpoint (Q31).
  - Any need to download data or hit the network.
