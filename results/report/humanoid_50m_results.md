# humanoid_50m results

Reward = per-seed mean over last 20 eval checkpoints, then mean ± std (ddof=1) across 4 seeds. Success rate shown as fraction × 100 (percentage).

| Variant | Reward | Success rate (%) |
|---|---|---|
| Vanilla CRL | 8.18 ± 0.78 | 29.7 ± 2.4 |
| **CRL+EMA** | **9.07 ± 1.58** | **31.1 ± 4.2** |
| CRL+EMA+Goal (reg=1e-7) | 8.35 ± 0.61 | 30.5 ± 1.4 |
