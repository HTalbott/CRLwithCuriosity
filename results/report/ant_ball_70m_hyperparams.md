# Hyperparameters for ant_ball 70M ablation


**Environment & schedule**

| Parameter | Value | Applies to |
|---|---|---|
| Environment | ant_ball | all |
| Total env steps | 7.0093×10^7 | all |
| Episode length | 1001 | all |
| Prefill steps | 2.56×10^5 | all |
| Parallel envs (train / eval) | 256 / 256 | all |
| Eval checkpoints | 200 | all |
| Seeds | {0,1,2,3} | all |

**Optimization & replay**

| Parameter | Value | Applies to |
|---|---|---|
| Batch size | 256 | all |
| Discount γ | 0.99 | all |
| Policy LR | 3×10^-4 | all |
| Critic LR | 3×10^-4 | all |
| Alpha (entropy) LR | 3×10^-4 | all |
| Optimizer | Adam | all |
| Unroll length | 62 | all |
| Replay buffer (max / min) | 10^4 / 10^3 traj. | all |
| UTD ratio | ≈0.063 | all |

**Network architecture**

| Parameter | Value | Applies to |
|---|---|---|
| Hidden width | 256 | all |
| Hidden layers | 2 | all |
| Skip-connection period | 4 | all |
| Activation | Swish | all |
| LayerNorm | off | all |
| Representation dim | 64 | all |

**Contrastive loss (CRL)**

| Parameter | Value | Applies to |
|---|---|---|
| Contrastive loss | fwd_InfoNCE | all |
| Energy function | -‖φ(s,a)−ψ(g)‖_2 | all |
| Logsumexp penalty λ | 0.1 | all |

**Target-critic EMA**

| Parameter | Value | Applies to |
|---|---|---|
| EMA τ (CRL target critic) | 0.005 | EMA, Goal |

**Goal critic**

| Parameter | Value | Applies to |
|---|---|---|
| Goal critic LR | 3×10^-4 | Goal |
| Goal critic coefficient | 1.0 | Goal |
| Goal-critic step warmup | 5×10^6 | Goal |
| Goal-critic anneal end | 1.5×10^7 | Goal |
| Perf. warmup metric | `eval/episode_success` | Goal |
| Perf. warmup threshold | 50.0 | Goal |
| Goal reach threshold | 0.5 | Goal |
| Positive-class weight cap | 4.0 | Goal |
| Goal-logit clamp (symmetric) | ±4.0 | Goal |
| Goal-logit EMA norm decay | 0.99 | Goal |
| Goal-logit L2 penalty | 1×10^-7 | Goal |
