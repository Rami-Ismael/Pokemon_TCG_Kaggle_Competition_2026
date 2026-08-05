# Skill-filtered demonstrations (accuracy gate)

Equal steps 4000; seeds [42, 43, 44]; eval 150 episodes (high-skill subset: eval-day min_score >= 1120).

| arm | high-skill top1 by seed (%) | mean | paired d (pp) | mean d | spread | all-eval mean d | verdict |
|---|---|---|---|---|---|---|---|
| unfiltered_scored | 66.5, 68.4, 68.9 | 67.93 | — | — | — | — | reference |
| top50_wd | 66.3, 66.9, 68.6 | 67.27 | -0.20, -1.50, -0.30 | -0.67 | 1.30 | -1.40 | drop |
| top25_wd | 62.5, 65.8, 61.6 | 63.30 | -4.00, -2.60, -7.30 | -4.63 | 4.70 | -5.60 | drop |

Accepted: NONE

Accuracy only gates. A full-budget retrain + benchmark_agents decides shipping (see the driver docstring).
