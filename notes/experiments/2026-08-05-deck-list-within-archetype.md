# Does the exact 60-card list matter within an archetype? (controlled comparison)

- **Hypothesis:** Holding the BC checkpoint fixed, swapping our *modal-reconstructed*
  Marnie's Grimmsnarl ex list for a *real top-player* Grimmsnarl list raises field win
  rate by ≥10pp — because the policy is not deck-conditioned, so it has learned lines
  that only an internally coherent, actually-played list can execute. Our list is the
  modal one of **15 distinct exact 60-card multisets** across 4,672 corpus
  deck-instances, so it may be a deck nobody piloted.
- **Design:** Controlled comparison (not an ablation, not a sweep) — the question is
  attribution of a single swap, not sensitivity or component credit.
- **Independent variable:** the 60-card list. ONE thing changes.
  - arm A: `configs/deck_lists/marnies_grimmsnarl_ex.csv` (modal reconstruction)
  - arm B: `agents/wmh_grimmsnarl/deck.csv` (documented top-player list)
- **Held fixed:** checkpoint `models/il_alldays_0804` (sha 1d67d1acdbb0), the pool,
  games/cell, PTCG_DEVICE=cpu, seat mirroring.
- **Baseline:** arm A = the exact build submitted as 55270787 (ladder 311.3).
- **Metric & protocol:** field win% ± σ vs pool v3 (42 validated opponents),
  10 mirrored pairs/cell = 20 games/cell, ~840 games/arm. Star topology (`--focus`).
- **Pre-registered decision:**
  - adopt (list matters) if arm B − arm A **≥ +10pp**
  - drop if |Δ| **< 5pp**
  - inconclusive if 5–10pp; would need ~440 games/arm to resolve at 80% power
- **Why this is the cheapest falsification:** no retraining, no HF streaming, both
  lists already on disk. If the effect is real it should be large and visible here.
- **Known limitation, stated up front:** the local pool overrates our agents 2.4–6.0×
  vs the ladder and ranked Grimmsnarl ABOVE Lucario when the ladder ranked it below.
  So this screens for a LARGE effect only; a positive result needs a ladder slot to
  confirm, and a null result is the more trustworthy outcome here.
- **Prior work checked:** `reports/deck_selection.md` (modal-list choice; Spidops modal
  covered only 25% of its archetype's instances and was called the least-consensus of
  the four corpus-only decks, implying Grimmsnarl's modal coverage is higher but it was
  never recorded); `reports/il_model_deck_selection.md` (deck axis, ladder falsification);
  memory `deck-matters-more-than-checkpoint` (now ladder-disputed).
- **Cost estimate:** ~2 arms x ~25 min at nice 10, CPU-only, alongside the pool
  round-robin. No MPS contention.

## Result

- **Observed** (840 games/arm vs pool v3's 42 validated opponents, star topology):

  | arm | 60-card list | field win% | Wilson 95% | Glicko |
  |---|---|---:|---|---:|
  | A | modal reconstruction (`marnies_grimmsnarl_ex`) | **86.2 ± 1.2%** (724/840) | [83.7, 88.4] | 1874.9 |
  | B | documented top-player list (`grimmsnarl_toplayer`) | 84.4 ± 1.3% (709/840) | [81.8, 86.7] | 1856.4 |

  **delta (B − A) = −1.8pp ± 1.7, 95% CI [−5.2, +1.6]** — CIs overlap, not separated.
  The real list is if anything *slightly worse*, and the difference is inside noise.

  Command:
  ```
  bash scripts/run_testbed.sh decklist_exp 10 \
      il_alldays_3ep@grimmsnarl_toplayer il_alldays_3ep@marnies_grimmsnarl_ex
  uv run python scripts/merge_il_sweep.py --dirs reports/il_sweep/decklist_exp \
      reports/il_sweep/poolv3_rr --out reports/il_sweep/decklist_merged.json
  ```

- **Decision: DROPPED** by the pre-registered rule (|Δ| 1.8pp < 5pp). The exact
  60-card list within an archetype is **not** the source of the error. Our modal
  reconstruction is not a broken deck, and swapping to a real top-player list buys
  nothing measurable.

- **What we learned:** "pure IL only works with the right deck format" is falsified in
  both of its forms. The strong form (IL needs the deck it was trained on) was already
  contradicted by the ladder — Grimmsnarl (3488 corpus episodes) scored 311.3 while
  Mega Lucario ex (1 episode) scored 418.0. The refined form (our reconstructed list is
  incoherent) is now falsified locally at n=840/arm, where a 10pp effect would have been
  trivially visible. Deck *format* is excluded as a major term.

  Transferable lesson: this is why the pre-registered threshold matters. A −1.8pp result
  is easy to narrate either way after the fact ("the real list is worse, interesting!"),
  and committing to |Δ|<5pp ⇒ drop *before* seeing it removes that freedom. Note also
  that the null is the trustworthy direction here: the pool overrates our agents 2.4–6.0x
  and could not be trusted to CONFIRM a deck effect, but a biased instrument can still
  falsify one.

  Caveat retained: measured in a regime where the agent wins ~85%. A list difference that
  only matters against opponents that actually beat us would not show up here — the same
  blind spot that limits the compounding-error test.
