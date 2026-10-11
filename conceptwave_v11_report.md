# Concept wave v1.1 — three fixes, re-measured

Fixes: (1) signed edges (|cos| top-k keeps the sign), (2) per-node cap replaces
the global renorm, (3) complex tanh amplitude squash (phase kept).

## Fix verification

| fix | metric | v1 | v1.1 |
|---|---|---|---|
| 1 signed edges | frac_positive of edge weights | 0.5643 | **0.4995** (target 0.3-0.7) |
| 2 no global renorm | amp after 1 round, phi = 0 / pi/2 / pi | 5.0616 / 5.0616 / 5.0616 (**phase-blind**) | **40.94 / 32.61 / 0.0** |
| 3 nonlinearity | cos of settled pattern from 2 random inits | 0.9931 (random ctx) / 0.8965 (concept[0] ctx) | 0.9484 / 0.9436 (**context-dependent, no ordering**) |

Fix 1 and 2 verify. Fix 3 does not: the eigenvector lock is still ~0.94-0.99 and
its direction flips with the awake mask, so no claim is made.

## exp 1 (matched budget: 40k tokens, same seed / val slice)

| | v1 | v1.1 | target |
|---|---|---|---|
| val_bits | **4.7675** | 4.8200 | < 4.331 |
| val_acc | 0.1384 | 0.1430 | - |
| frac_active | 0.0157 | **0.0093** | < 0.10 |
| concept_sim_top5 | 0.3855 | 0.3715 | concept should dominate |

v1.1 is 0.053 bits *worse* at matched budget. Sparsity passes. Concept still
does not dominate the settled pattern (0.37, unchanged from v1's 0.17-0.39).

## exp 2 (interference = error correction)

Absolute energy on the wrong path (as originally specified):

| | v1 | v1.1 |
|---|---|---|
| 10-round decay | 0.5945 (passes >50%) | **-17.36** (grew 17x) |

The absolute metric is meaningless once the global renorm is gone, so the same
experiment measured as *share of total field energy*:

| | v1 | v1.1 |
|---|---|---|
| share decay | **0.8333** | 0.3645 |

On the fair metric v1 still wins. Removing the renorm made error correction worse.

## exp 3 (long context)

Same trained v1.1 model, context window changed at eval only:

| hist | val_bits |
|---|---|
| 100 | 4.8221 |
| 1000 | 4.8200 |

Difference 0.0021 (needs < -0.1). Not demonstrated. Note: this is eval-only,
not matched re-training — the original exp3 trains a model per hist.

## exp 4 (external feedback)

Both variants "recover in 1 round" — because the wrong injection never flips the
argmax (`argmax_after_wrong = 5` == correct). The pass is vacuous.

## Known limitation (measured, not speculation)

The settled pattern still converges to one dominant mode from arbitrary init
(cos 0.94-0.99). Three fixes did not change this. See
`test_eigenvector_lock_is_not_solved_by_v11`.
