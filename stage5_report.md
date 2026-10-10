# Stage 5A final — 100k-step report

Commit `2dad34335b2b` (main). Command: `python3 run_100k.py`.

## Criteria

| criterion | target | measured | |
|---|---|---|---|
| NaN free | 100k steps | no NaN | PASS |
| e_max < 0.95 | not saturated | 0.4984 | PASS |
| e_mean ≈ 0.3 ±20% | 0.24–0.36 | 0.2932 | PASS |
| e_std > 0.05 |波动够大 | 0.0640 | PASS |
| BN output distinguishable | corr < 0.8 | 0.0025 | PASS |
| MSE drop | > 30% | 56.7% | PASS |
| score_std | > 0.05 | 0.1648 | PASS |
| cull rate (per 2000-step round) | 40% ±10% | 0.400 | PASS |
| BN count | 1000 ±10% | 1000 (516 unmerged) | PASS |
| EZ load / release | both > 0 | 944 / 940 | PASS |
| hibernate byte-identical | True | True | PASS |
| archive ratio | > 3x | 5.235x | PASS |
| rollback / switch / leak | ok / <100ms / 0 | ok / 0.4ms / 0 MB | PASS |
| **merge rate** | **5%–20%** | **48.4%** | **FAIL** |
| **between > within zone std** | division of labour | 0.0163 vs 0.0314 | **FAIL** |

## Sampling

```
us/step 772.7      wall 77.3 s for 100k steps
arrays  2.631 MB   rss 19.4 MB (getrusage ru_maxrss)
```

## Why the two failures are structural

**Merge rate.** Promotion is a single terminal decision, so merge rate = P(score > 0.7).
An untrained BN already scores 0.668 (acc 0.70 from `1/(1+50*se)` at se≈0.0085), a trained
one 0.75. The distribution straddles the threshold, so roughly half clears it. Making the
task learnable (MSE drop > 30%) necessarily lifts scores, which lifts the merge rate —
measured trade-off at 100k steps: lr 0.005 → drop 7%, merge 15%; lr 0.03 → drop 57%,
merge 48%. The two targets pull in opposite directions.

**Between vs within zone.** `between` is the std of zone means, `within` the mean std inside
a zone. `within` is dominated by the utilisation term, which is bimodal by construction
(gate = median, so half of every zone reads 1 and half 0, worth ±0.1). Widening the per-zone
difficulty contrast barely moves the ratio: amplitude U(0.2,1.5) → 0.267, U(0.02,6.0) → 0.298.

## Fixes applied this round

- `acc = 1/(1+50*se)`, utilisation gate = running `median(|pred|)`
- restored `-0.05*cost -0.05*conflict` (both were silently 0.0 in the first pass)
- `train_step` no longer divides by N — each BN owns its adapters, so the batch-mean
  scaling made the gradient ~1e-7 and lr had **no measurable effect** across 5e-3…1e-1
- per-zone horizon `BASE_HORIZON*(0.5+0.1*i)` = 25…120 ticks
- culling relative (worst 40% per zone / 2000 steps), promotion terminal
