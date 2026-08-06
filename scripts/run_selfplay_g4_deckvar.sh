#!/bin/zsh
# Self-play generation 4: DECK VARIATION ON BOTH SIDES.
#
# The experiment. Every RL result this project has is a policy piloting ONE deck
# against opponents piloting the SAME deck (verified: run_selfplay_g1/g2/g3.sh
# pass no deck flags, so all three generations ran the Mega Lucario ex mirror).
# Hypothesis: that is why local wins keep failing to transfer to the ladder --
# the policy learned one matchup, not the game.
#
# Two arms, EQUAL COMPUTE and EQUAL SEEDS. Deck randomization makes the task
# strictly harder, so a lower win rate in the treatment arm is unreadable
# without the control:
#
#   control    no deck pool -- single-deck mirror self-play, today's behaviour
#   treatment  --deck-pool: learner deck and opponent deck drawn INDEPENDENTLY
#              and uniformly per episode from 29 legality-verified lists
#
# BUDGET IS WALL-CLOCK, not a pinned step count (PR #55 rescinded the
# equal-steps rule, 2026-08-06). TIMESTEPS is set out of reach so MAX_SECONDS
# binds. This matters here: the treatment arm's games are ~1.3x longer, so an
# equal-STEP budget would quietly hand it ~1.3x the compute.
#
# Opponents are LINEAGE-ONLY in both arms: the live mirror + our own frozen
# checkpoints. Self-play means the agent plays its own lineage. The mix's
# last share (externals) is always 0 and the driver hard-errors on anything
# else; --field-pool must be empty; --league entries must be real saved
# checkpoints. Externals are never training opponents, which is what keeps
# them unseen and the local evaluation pool a genuine out-of-sample filter.
#
# Runs arms SEQUENTIALLY: never two MPS jobs at once.
#
#   scripts/run_selfplay_g4_deckvar.sh                 # both arms, all seeds
#   ARMS=treatment SEEDS=42 scripts/run_selfplay_g4_deckvar.sh
set -u
cd "$(dirname "$0")/.."
MAIN=/Users/ramiismael/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026
PY=$MAIN/.venv-ppo/bin/python

: ${SEEDS:="42 43 44"}
: ${ARMS:="control treatment"}
: ${TIMESTEPS:=100000000}     # out of reach on purpose; MAX_SECONDS is the budget
: ${MAX_SECONDS:=9000}          # per arm-seed wall-clock budget (2.5h; 6 runs = 15h)
: ${POOL:='@configs/deck_pools/legal_decks.txt'}
: ${RETRY_HOURS:=3}
: ${INIT:=models/il_agent}
: ${LEAGUE:=models/s2v2/e0_s43,models/il_alldays_0804}

mkdir -p runs
for seed in ${=SEEDS}; do
  for arm in ${=ARMS}; do
    tag="g4_${arm}_s${seed}"
    out="models/${tag}"
    log="runs/${tag}.log"
    [ -f "$out/policy_full.pt" ] && { echo "$tag already finished -- skipping"; continue; }

    deck_args=()
    [ "$arm" = "treatment" ] && deck_args=(--deck-pool "$POOL")

    echo "=== $tag  (wall-clock budget ${MAX_SECONDS}s; steps uncapped)"
    # Preflight refuses to launch into a contended machine, and it is RIGHT to
    # -- contention invalidates the run's throughput arithmetic and can starve
    # env workers into timeout losses that poison the rollouts. Other worktree
    # sessions come and go, so a refusal is a "wait", not a "give up": retry on
    # a fast failure for up to RETRY_HOURS, and only then abort the sweep.
    t_arm=$(date +%s)
    while :; do
      start=$(date +%s)
      nice -n 5 $PY scripts/train_ppo_puffer.py \
        --init-from "$INIT" --kl-prior "$INIT" \
        --league "$LEAGUE" \
        --mix 0.625,0.375,0 \
        --seed "$seed" \
        --total-timesteps "$TIMESTEPS" --max-seconds "$MAX_SECONDS" \
        --run-tag "$tag" --out "$out" \
        "${deck_args[@]}" >>"$log" 2>&1
      rc=$?
      ran=$(( $(date +%s) - start ))
      [ $rc -eq 0 ] && break
      waited=$(( $(date +%s) - t_arm ))
      if [ $ran -lt 120 ] && [ $waited -lt $((RETRY_HOURS * 3600)) ]; then
        echo "$tag: launch failed fast (rc=$rc after ${ran}s) -- preflight or "\
             "startup; retrying in 300s (waited ${waited}s)" | tee -a "$log"
        sleep 300
        continue
      fi
      echo "ABORTING: $tag failed rc=$rc after ${ran}s" | tee -a "$log"
      exit $rc
    done
    echo "$tag rc=0" | tee -a "$log"
  done
done

echo
echo "All arms done. Next, in order:"
echo "  1. uv run python scripts/fallback_diagnostic.py --agent <ckpt>  (per checkpoint)"
echo "  2. uv run python scripts/report_deck_arms.py models/g4_*  (per-deck / worst matchup)"
echo "  3. local Glicko vs the anchored pool  -- go/no-go for a submission slot"
echo "  4. refresh the ledger, submit, read the ladder"
