# Feature ablation (accuracy gate)

Equal steps per arm: 4000; seeds [42, 43, 44]; eval on first 150 held-out-day episodes via eval_rung1.

| arm | top1 by seed (%) | mean top1 | paired d vs baseline (pp) | mean d | spread(d) | verdict |
|---|---|---|---|---|---|---|
| baseline | 69.5, 69.6, 70.9 | 70.00 | — | — | — | reference |
| ko_race | 68.0, 68.7, 69.1 | 68.60 | -1.50, -0.90, -1.80 | -1.40 | 0.90 | drop |
| prize_race | 70.9, 69.9, 70.8 | 70.53 | +1.40, +0.30, -0.10 | +0.53 | 1.50 | drop |
| energy_deficit | 67.9, 69.4, 66.4 | 67.90 | -1.60, -0.20, -4.50 | -2.10 | 4.30 | drop |
| status_conditions | 68.0, 69.1, 69.3 | 68.80 | -1.50, -0.50, -1.60 | -1.20 | 1.10 | drop |
| attack_tactical | 68.6, 69.5, 70.2 | 69.43 | -0.90, -0.10, -0.70 | -0.57 | 0.80 | drop |
| attach_enable | 68.7, 68.6, 69.3 | 68.87 | -0.80, -1.00, -1.60 | -1.13 | 0.80 | drop |
| retreat_switch | 68.1, 68.8, 68.0 | 68.30 | -1.40, -0.80, -2.90 | -1.70 | 2.10 | drop |

Accepted: NONE

Accuracy only gates features -- the ship decision belongs to scripts/benchmark_agents.py (see Step 4).
