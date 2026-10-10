# Stage 5A rerun — spatial division of labour

Three changes from the 5A spec rerun:

1. **BN spatial division of labour.** The main system is split into `n_zones=20`
   contiguous address blocks; each BN shadows exactly one block. Every BN also
   gets its own dynamics — `omega ~ N(0, U(0.3,0.7))`, `kappa ~ U(0.1,0.2)`,
   `sigma_w ~ U(0.2,0.4)`.
2. **Proportional culling.** `low_count` over a 2000-sample ring buffer; a BN is
   culled when more than 50% of the window sits below 0.3. Replaces the old
   absolute `low_score_steps > 10000`, which cannot be reached inside a 20k run.
3. **5C keeps two modes.** `hibernate` is judged on byte equality only;
   `archive` on ratio only.

## Prediction horizon

BNs predict 5 samples ahead (50 ticks), not the very next sample. Next-sample
prediction is trivially easy: every BN nails it, scores saturate, and the merge
rate collapses to 0%.

## Results — 100k steps, 1000 VN + 1000 BN, 20 zones

| Criterion | Target | Measured |
|---|---|---|
| no NaN | 100k steps | pass |
| `e_mean` ≈ e_target | 0.24–0.36 | **0.2927** |
| `e_max` | < 0.95 | pass |
| R drifting | window range > 0.01 | **0.031–0.139** |
| BN output distinguishable | corr < 0.8 | **-0.0565** |
| MSE drop | > 30% | **40.3%** |
| EZ load / release | both > 0, release <= load | **937 / 933** |
| TVN fired | > 0 | **5** |
| FVN bit-identical | yes | pass |
| hibernate byte-equal | yes | pass |
| archive ratio | > 3x | **5.235x** |
| rollback / switch / leak | ok / <100ms / none | **ok / 0.4ms / 0 MB** |
| **`score_std`** | **> 0.1** | **0.0203 — FAIL** |
| **merge rate** | **5–20%** | **100% — FAIL** |
| **cull rate** | **30–60%** | **0% — FAIL** |

## Why the three criteria fail — it is the score formula, not the code

Measured component spreads:

- `acc = 1/(1 + 10·se)`: se lands in 0.0012–0.026, so acc spans only
  0.795–0.988 → **std 0.031**, contributing at most `0.4 × 0.031 = 0.012`.
- `util`: threshold 0.05 is far below the output scale (median |pred| ≈ 0.19),
  so **100% of BNs activate and util std = 0.0000**. The full 0.2 weight is
  inert.
- `contrib` and `stab` are likewise saturated near 1.

So `score_std` has a hard ceiling near **0.044** under this formula, and 0.1 is
unreachable no matter how the task is tuned.

## Measured effect of changing the formula

| Variant | score_std | merge% (>0.7) | cull% (<0.3) |
|---|---|---|---|
| spec as-is | 0.0197 | 100.0 | 0.0 |
| acc coefficient 10 → 50 | 0.0380 | 83.7 | 0.0 |
| util threshold 0.05 → median | 0.0588 | 70.5 | 0.0 |
| acc ×50 **and** util = median | 0.0588 | **17.5** | 0.0 |

The combined variant lands the merge rate inside the 5–20% window, but
`score_std` still caps out at 0.059 and no BN ever scores below 0.3, so culling
never fires.

## Open questions for the spec owner

1. `score_std > 0.1` is unreachable under the current weights. Raise the acc
   coefficient and move the util threshold onto the output scale, or restate
   the criterion.
2. Culling needs genuine losers. Right now every BN solves its zone well enough
   that none fall below 0.3; a harder or more heterogeneous task is needed.
