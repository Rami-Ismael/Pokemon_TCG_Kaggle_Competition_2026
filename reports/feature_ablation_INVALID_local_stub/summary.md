# Feature ablation (accuracy gate)

Equal steps per arm: 4000; seeds [42, 43, 44]; eval on first 120 held-out-day episodes via eval_rung1.

| arm | top1 by seed (%) | mean top1 | paired d vs baseline (pp) | mean d | spread(d) | verdict |
|---|---|---|---|---|---|---|
| baseline | 58.0, 58.5, 56.3 | 57.60 | — | — | — | reference |
| ko_race | 55.5, 55.3, 55.1 | 55.30 | -2.50, -3.20, -1.20 | -2.30 | 2.00 | drop |
| prize_race | 56.4, 57.1, 56.8 | 56.77 | -1.60, -1.40, +0.50 | -0.83 | 2.10 | drop |
| energy_deficit | 54.4, 57.1, 56.8 | 56.10 | -3.60, -1.40, +0.50 | -1.50 | 4.10 | drop |
| status_conditions | 57.3, 57.5, 58.9 | 57.90 | -0.70, -1.00, +2.60 | +0.30 | 3.60 | drop |
| attack_tactical | 57.2, 56.7, 57.7 | 57.20 | -0.80, -1.80, +1.40 | -0.40 | 3.20 | drop |
| attach_enable | 56.7, 57.1, 56.8 | 56.87 | -1.30, -1.40, +0.50 | -0.73 | 1.90 | drop |
| retreat_switch | 56.2, 57.5, 55.1 | 56.27 | -1.80, -1.00, -1.20 | -1.33 | 0.80 | drop |

Accepted: NONE

Accuracy only gates features -- the ship decision belongs to scripts/benchmark_agents.py (see Step 4).
