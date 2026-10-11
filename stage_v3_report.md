# VN v3 — two-layer concept wave

Spec from deepseek: `concept` explicitly trained, `F` a learned predictor,
L=2 with inter-layer matrices, N=1000, D=32, TinyShakespeare, judge
`concept_sim_top5 > 0.6` **and** `val_bits < 4.0`.

**Both criteria fail.** Per the spec's own rule — "任一个失败：收工" — this
branch is a stop, not a tuning problem.

## Criteria

| criterion | target | measured | |
|---|---|---|---|
| concept_sim_top5 | > 0.6 | **0.034** (mean over `Fae Th`) | ❌ off by 0.57 |
| val_bits | < 4.0 | **5.2124** | ❌ off by 1.21 |
| sparsity | < 10% | **1.000** | ❌ |
| (reference) unigram | 4.774 | — | we are *worse* than unigram |

## Does `concept` have gradient? Yes — and it moves

deepseek's first requirement. Measured before any training:

| | value |
|---|---|
| ‖dL/dconcept‖ (v2, N=1000) | **3.261**, abs-sum 110.43 |
| rows with non-zero grad | **1000 / 1000** |
| FD relerr, largest-|grad| rows | 0.044 – 0.059 (best rows ~0.02) |
| after 2k steps: mean abs drift | 0.135 |
| after 2k steps: mean cosine vs init | **0.509** |

So gradient exists, is roughly correct (same sign everywhere, relerr a few
percent on the large rows), and the parameters do move. **Not having a
gradient is not why this fails.**

Small-|grad| rows show relerr ≈ 1.0. That is float32 central-difference
resolution, not a wrong formula: the analytic value there is ~4e-5.

## What the two layers bought

Nothing measurable. Training curve (val_bits / val_acc):

| step | 500 | 3500 | 6500 | 9500 | 12500 |
|---|---|---|---|---|---|
| val_bits | 5.4022 | **5.1662** | 5.2416 | 5.1646 | 5.2124 |
| val_acc | 0.1106 | **0.1269** | 0.0918 | 0.1319 | 0.1068 |

The best number is at step 3500. The following 9000 steps produce nothing —
it oscillates in 5.16–5.24. And `concept_sim_top5` drifts around zero
(-0.111 at 3.5k, +0.034 at 12.5k) rather than rising.

## Two implementation facts worth keeping

**1. The unbounded inter-layer term overflows.**
`context += beta * G @ pooled` is a positive-feedback loop — pooled grows with
context, so `G @ pooled` grows faster. At N=1000/dc=32/lr=3e-3 the field hit
`inf` after ~2k steps. Fixed with `tanh` on the injected term plus a per-row
L2 cap (`ctx_cap = 1.0`).

**2. `dL/dG` is truncated, not exact.**
`G` acts through context, which feeds the next round's pooled summary — a loop
we do not unroll. Finite differences disagree in magnitude; only the sign is
reliable. Pinned in `test_G_gradient_is_truncated_not_exact`.

## Sparsity is structural, not a threshold problem

`sparsity = 1.000` at every checkpoint. Softmax addressing assigns a non-zero
weight to every node, so every node is "active" by any threshold-based
definition. Earlier the v1 threshold sweep traded sparsity directly against
bits (28.8%/4.847 → 2.1%/5.017); there is no setting that gives <10% for free.

## Continuity with what came before

| version | propagation | concept_sim_top5 | val_bits |
|---|---|---|---|
| v1 | fixed kernel K | 0.37 | 4.65 |
| v1.1 | + non-linear, ±edges | 0.37 | 4.82 |
| v2 | learned F, single layer | **-0.116** | 4.84 |
| v3 | learned F, L=2, inter-layer | **0.034** | 5.21 |

Adding `F` made concept matching *anti*-correlated (v2), and adding the second
layer only moved it back to zero (v3). Across four versions the settled pattern
has never been dominated by concept similarity, which was the load-bearing
assumption of this whole line.

## Reproduce

```
python3 train3_chunk.py --steps 3000 --vlen 600 --fresh   # then drop --fresh
python3 -m pytest tests/test_v3.py -q
```

`~11 ms/step` at N=1000/dc=32/L=2.
