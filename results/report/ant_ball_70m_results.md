# ant_ball_70m results

Reward = per-seed mean over last 20 eval checkpoints, then mean ± std (ddof=1) across 4 seeds. Success rate shown as fraction × 100 (percentage).

| Variant | Reward | Success rate (%) |
|---|---|---|
| Vanilla CRL | 85.86 ± 17.30 | 20.3 ± 3.5 |
| CRL+EMA | 98.10 ± 21.24 | 23.0 ± 3.1 |
| **CRL+EMA+Goal (reg=1e-7)** | **108.12 ± 6.75** | **19.3 ± 1.4** |
