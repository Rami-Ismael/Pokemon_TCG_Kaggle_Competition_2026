# Deterministic-future features for il_agent — Step 1 candidate list

Inspired by the Orbit Wars 2nd-place writeup: feed the network the
"do-nothing future" so it doesn't have to learn game arithmetic. We cannot
roll the simulator forward at inference (the cg search API needs
determinized hidden state and a time budget we don't want to spend), so our
version is **hand-computed deterministic futures from the visible
observation** — the quantities a strong player computes mentally before
deciding: damage-race turns, energy timelines, prize arithmetic.

## Shared inputs, and why they are legal

Every feature below is computed from exactly two sources:

1. **The POV-filtered `observation.current`** — the engine already nulls
   the opponent's hand, both deck contents, and facedown prizes
   (pinned by `tests/test_privacy_no_leak.py`). Everything we touch is
   what a human player sees across the table: both sides' in-play Pokémon
   with card ids, current/max HP, **attached energies** (`Pokemon.energies`
   is populated for both players — attached energy is face-up public
   state), tools, special-condition flags, prize/deck/hand *counts*, and
   discard piles.
2. **The static card database** — `cg.api.all_card_data()` /
   `cg.api.all_attack()`: per-card HP, weakness, resistance, retreat cost,
   ex/megaEx flags, attack list; per-attack printed damage and energy cost.
   This is the rulebook/card-text every player has memorized — public by
   definition, and shipped inside the evaluator's own `cg` library (the
   same lib the observation parser already loads), so no new bundle weight.

Cost: every feature is a handful of dict lookups and integer ops per
decision (worst case ~a dozen attacks × ~10 energy symbols). The evaluator
envelope is ~1.6 vCPU / ~198 MiB; measured encode time is dominated by
tensor construction, not this arithmetic. No feature loops over more than
(2 players × 9 in-play slots × ~4 attacks).

### Damage model (approximation, stated once)

`dmg(attack, attacker, defender) = printed damage; ×2 if defender.weakness
== attacker.energyType; −30 (floor 0) if defender.resistance ==
attacker.energyType`. Weakness ×2 and base-damage 1:1 were verified
empirically from visible episode logs (ATTACK→HP_CHANGE pairs, modal
ratios 2.0 and 1.0); the −30 resistance rule is the standard-format rule
but our log sample was too sparse to confirm it (22 samples, polluted by
effect-bonus attacks) — treated as an approximation. Variable-damage
attacks (coin flips, scaling effects) contribute their printed base
(often 0); the feature is a lower bound there, and the model still sees
the attack-id embedding to correct for it. "Payable" means an attack whose
energy cost is satisfiable by the Pokémon's currently attached energies
under greedy matching (colored first, RAINBOW wildcard, TEAM_ROCKET
matches {P,D}, COLORLESS consumes leftovers).

## Candidates

Marked **[arm]** = gets its own ablation arm now; **[deferred]** = passes
both filters but is not in the first ablation wave (reason given).

### Group `ko_race` [arm] — the two seeded KO-race features (tightly coupled)

1. **turns-until-my-active-is-KO'd** = ceil(my_active.hp / best payable
   dmg of opp active vs my active), capped at 10, ∞→cap when they can't
   attack yet. *Visibility:* opp active card id, its attached energies,
   my active HP — all face-up. *Cost:* ≤4 attacks × deficit check.
2. **turns-until-opponent-active-is-KO'd** = same with roles swapped
   (opp active current HP is visible in `Pokemon.hp`).
3. **race sign** = 1 if (2) ≤ (1) (I win the current exchange; it is my
   decision so I move first), else 0. Pure derivation of 1+2.
4. **my best payable damage / 300** and **opp best payable damage / 300**
   — the raw exchange rates the two turn counts are built from; included
   in the same arm because they are the same computation.

### Group `prize_race` [arm] — prize arithmetic at current exchange rate

5. **prize value of KOing opp active** (1/2/3 via ex/megaEx flags of the
   visible card id, ÷3) and **prize value of my active being KO'd**.
   *Visibility:* both actives are face-up; ex/megaEx is printed on the
   card. *Cost:* two dict lookups.
6. **"KOing opp active wins me the game"** = prize value ≥ my remaining
   prize count (`len(me.prize)` is public — the pile size is on the
   table). Symmetric **"losing my active loses me the game"**.
   Together with the baseline's existing prize counts and the `ko_race`
   turns, this is who-wins-the-race-at-current-exchange-rate arithmetic.

### Group `energy_deficit` [arm] — energy-attachments-still-needed (seeded)

7. **My active's minimum energy deficit** = min over its attacks of
   (unmatched cost symbols given attached energies), ÷4: "attachments
   until I can attack at all". *Visibility:* own board. *Cost:* greedy
   matcher over ≤10 symbols.
8. **Min deficit across my bench** (÷4): how far my next attacker is from
   coming online. Same computation over ≤8 visible bench Pokémon.
9. **Opp active's minimum energy deficit** (÷4): their attached energy is
   face-up public state, so their timeline is computable too — "how many
   turns before the threat goes live".

### Group `status_conditions` [arm] — not a future, but an un-encoded present

10. **The 10 special-condition booleans** (`poisoned/burned/asleep/
    paralyzed/confused` × both players) from `PlayerState`. Candidate
    because the current encoder *drops them entirely*, and they gate the
    KO race (asleep/paralyzed actives can't attack; poison shifts the
    damage race). *Visibility:* status markers are physically on the
    table. *Cost:* 10 field reads. Included in the wave because any
    KO-race gain could be confounded by missing status info.

### Group `attack_tactical` [arm] — option-level: "this ATTACK option KOs now"

11. **Per ATTACK option: dmg vs opp active ÷300**, **KOs-now** binary
    (dmg ≥ opp active hp), **KO-wins-game** binary (KOs-now AND prize
    value ≥ my remaining prizes). Option-level version of the race
    arithmetic — the strongest form of "the network shouldn't have to do
    arithmetic to know this button wins". *Visibility:* attackId is in
    the option; defender is the visible opp active. *Cost:* one lookup +
    two comparisons per ATTACK option.

### Group `attach_enable` [arm] — option-level energy timeline

12. **Per ATTACH option: target's post-attach minimum deficit ÷4** and
    **enables-an-attack** binary (deficit hits 0 with this attachment).
    Distinguishes "this attachment turns my attacker on" from "this
    parks energy". The attached card's energy type comes from the static
    DB (special energies approximated as wildcard). *Visibility:* own
    hand card + own board target. *Cost:* one matcher call per ATTACH
    option.

### Group `retreat_switch` [arm] — option-level: who should be active

13. **Per switch-target option (CARD options in SWITCH/TO_ACTIVE
    contexts, and the RETREAT option's implied bench)**: candidate's
    survival turns vs opp active's best payable attack ÷10, and
    candidate's best payable damage vs opp active ÷300. "If this Pokémon
    becomes active, does it die and can it fight back." *Visibility:*
    own bench + visible opp active. *Cost:* one damage table per option.
    Implemented for CARD options via their ref; RETREAT itself carries no
    target ref, so it stays feature-zero (the switch happens in a
    follow-up select where the features do fire).

### [deferred] candidates — pass both filters, not in wave 1

14. **Opp bench threat**: max payable damage among opp bench vs my active
    (their bench + energies are face-up). Deferred: overlaps `ko_race`
    only after a switch decision by the opponent, which is *their* policy
    choice — weaker deterministic claim, and wave 1 already has 7 arms
    (~27 training runs at 3 seeds).
15. **Deck-out race**: my deckCount == 0 → I lose at my draw; turns of
    draw remaining ÷60. Public counts, trivially cheap. Deferred:
    deckCount/60 for both players is already a baseline global scalar;
    the added step function only matters in rare mill endgames.
16. **Retreat-cost payability**: my active's retreat cost vs attached
    energy count (can I even retreat without an effect card). Public +
    static DB. Deferred: the engine only offers RETREAT when legal, so
    the legality half is already encoded by the option's presence; only
    the "energy spent by retreating" nuance remains, which overlaps 13.
17. **Weakness alignment flags** (my active weak to opp type / vice
    versa). Public + static DB. Deferred: strictly implied by the
    ko_race damage numbers, which already fold weakness in — a separate
    arm would double-test the same signal.

## What is deliberately NOT here (fails filter b)

Anything derived from `steps[0][0]["visualize"]`, opponent hand contents,
either deck's contents or order, facedown prizes, or opponent discard
*probabilities* conditioned on hidden zones. Also anything needing the
simulator rolled forward (`search_begin` requires determinized hidden
state — that's the part of the Orbit Wars trick we explicitly cannot
port).

## Ablation protocol (Step 3 summary)

One group per arm, each arm = baseline encoder + that group only; 3 seeds
(42/43/44); identical data (same split, same per-seed stream order) and
equal `--total-steps` for every arm; metric = eval_rung1 top-1/top-3 on
the held-out day vs the majority-class baseline on the same rows. Accept
iff mean top-1 improvement over the *baseline arm* exceeds the across-seed
spread (max−min) of both arms' runs. Finish with one combined arm of all
accepted groups. Accuracy only gates; only
`scripts/benchmark_agents.py` (mirrored pairs, win rate ± σ) decides
whether the combined feature set ships.
