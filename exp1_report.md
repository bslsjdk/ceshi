# Exp 1-3: VN v1 vs same-parameter Transformer

Fairness controls, all enforced in `exp1.py`:
1. **Parameters**: VN 9368 vs TF 9448 (0.85% apart)
2. **Token budget**: both consume exactly 300,000 training characters
3. **Optimiser**: the *same* `vn_mobile.io.adam.Adam` class, identical settings
4. **Data**: identical split, identical vocab (65 chars)

Transformer is hand-written numpy (`tf_micro.py`) because torch is absent in
this sandbox. Its backward pass is verified by finite differences against all
15 parameter tensors (worst relative error 2.2e-3, which is the float32
central-difference floor). One real bug was found and fixed this way: `G[wo]`
was using the post-projection attention output instead of the pre-projection
one (relative error 0.80 before, 2.0e-6 after).

## Exp 1 — long context (val bits, lower is better)

| model | params | tokens | val bits | val acc | wall |
|---|---|---|---|---|---|
| **VN v1** | 9368 | 300k | **4.331** | 0.2013 | 293.2 s |
| TF T=64 | 9448 | 300k | **3.6876** | 0.2570 | 3.2 s |
| TF T=256 | 9448 | 300k | **3.8407** | 0.2500 | 8.7 s |
| TF T=1024 | 9448 | 300k | **3.9845** | 0.2324 | 57.5 s |

VN loses by **0.643 bits** to the best Transformer, and is **92x slower**.

Counter-intuitive result: *longer* context is *worse* for the Transformer
(3.69 at T=64 → 3.98 at T=1024). This is an artefact of the equal-token
budget: T=1024 gets only 292 optimiser steps versus 4687 at T=64. It is
under-training, not evidence that long context hurts.

## Exp 2 — persistent state (already inside Exp 1)

VN's validation pass (`roll`) evolves the field continuously across the whole
3000-character validation set **without ever resetting state**. The
Transformer validates in independent windows with no cross-window carry.

So VN already had the theoretical advantage (O(1) state, unlimited context)
and still lost by 0.643 bits. Persistent state is not being converted into
predictive accuracy. This is consistent with the earlier measurement that
injected signal decays to 1/1000 within ~42 ticks.

## Exp 3 — sparse activation

| metric | measured |
|---|---|
| fraction of VNs active | **1.000** |
| energy share of top 1% | 0.0319 |
| energy share of top 10% | 0.2279 |

VN v1 is **densely** active: all 1000 neurons carry non-negligible energy at
every tick, and the top 1% hold only 3.2% of the energy (uniform would be
1.0%). There is no sparsity to exploit, so the "sparse activation saves
compute" advantage does not exist in this architecture as built.

## Verdict

All three signals fail. Under deepseek's own criterion — "any one win > 5%
means the architecture has an advantage; all three lose means it does not" —
**VN v1 has no demonstrated architectural advantage at ~9.4k parameters**.

Loss margin: 0.643 bits = 21% relative (3.688 / 4.331 = 0.852, i.e. VN
retains 85.2% of the Transformer's advantage over uniform).
