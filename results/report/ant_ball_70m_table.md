# Ant_ball 70M: eval reward across training

Reward = mean over the last 20 eval checkpoints per seed. `mean` / `std` are across the 4 seeds (sample std, ddof=1).

| Variant                 |   seed 0 |   seed 1 |   seed 2 |   seed 3 |   mean |   std |
|:------------------------|---------:|---------:|---------:|---------:|-------:|------:|
| Vanilla CRL             |    73.04 |   109.91 |    87.03 |    73.47 |  85.86 | 17.3  |
| CRL+EMA                 |    89.56 |    93.7  |    80.3  |   128.82 |  98.1  | 21.24 |
| CRL+EMA+Goal (reg=1e-7) |   111.58 |    98.73 |   108.01 |   114.18 | 108.12 |  6.75 |
