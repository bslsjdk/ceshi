# Stage 6A — TinyShakespeare, character level

Reference points: uniform 6.022 bits, unigram **4.774 bits**, GPT-team
14,770-param model 1.088 bits / 80.71% accuracy.

## Headline

| | bits | acc |
|---|---|---|
| uniform | 6.022 | 1.5% |
| unigram | 4.774 | — |
| **VN-Mobile (ours)** | **4.382** | **19.5%** |
| sklearn on the same features | 4.276 | 25.0% |
| GPT-team target | 1.088 | 80.7% |

We clear the unigram baseline and land within 0.1 bits of what an
off-the-shelf logistic regression extracts from the identical features.
We do **not** reach the GPT-team number; see "Why it stops here".

## Two real defects found, both fixed

**1. `pop.bias` gradient had the wrong shape.** bias is shared `[DIM]`;
the code subtracted a `[N, DIM]` array. `ValueError: non-broadcastable
output operand with shape (8,) doesn't match (1000,8)`. Every preset
touching bias (`head_bias`, `all`) had therefore *never executed a single
step* — a previously reported result for those presets could not have been
measured. Fixed to `dz.sum(axis=0)`.

**2. The readout was blockwise.** `logits[v] = <pooled[v], w[v]>` gives
character v's score only block v's own activity, so no logit can see what
the rest of the field is doing. Measured effect of widening to
`[V, V*DIM]` (every logit sees the whole pattern):

| readout | val bits @20k |
|---|---|
| blockwise `[V, DIM]` | 8.497 |
| full-field `[V, V*DIM]` | **4.501** |

## Is it the optimiser or the representation? (kept separate)

Collected 12,000 `(pooled, next_char)` pairs and handed them to
sklearn's `LogisticRegression` — a solver we did not write:

- representation signal-to-noise: 16.2
- sklearn val: **4.276 bits**
- our SGD val: **4.382 bits**

Since sklearn only reaches 4.276 on these features, our optimiser is
within 0.1 bits of the ceiling the representation supports. **The
remaining gap to 1.088 bits is the representation, not the training code.**

## Learning-rate / trainable-parameter sweep (20k steps)

| preset | lr 1e-4 | 1e-3 | 1e-2 |
|---|---|---|---|
| adapter | 6.028 | 6.028 | 6.028 |
| head (blockwise) | 6.038 | 8.497 | 11.286 |
| head_bias | 6.034 | 5.993 | 28.331 |
| all | 6.034 | 5.993 | 23.210 |

`adapter` is flat to three decimals across a 100x lr range: its measured
sensitivity is ~1e-6 against the head's ~3e-4, so **"train adapters only"
is structurally unable to move the loss**, confirming the sensitivity
measurement. Above lr 1e-3 the run diverges rather than improves.

## Longer history made it worse (surprising, reported as-is)

Injecting a window of recent characters with decay 0.6:

| history | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| val bits | **4.382** | 5.233 | 7.672 | 10.136 |

Lighting several blocks at once destroys the between-character contrast
the readout depends on. One character at a time is best.

## Curve (lr 1e-3, full readout)

| step | train bits | val bits | val acc |
|---|---|---|---|
| 25k | 4.411 | 4.448 | 19.5% |
| 50k | 4.150 | 4.422 | 20.4% |
| 75k | 4.049 | 4.452 | 19.3% |
| 100k | 4.003 | **4.382** | 19.5% |

Validation flattens after ~50k while train keeps falling: the field is
being fitted, not the language.

## Honest limits

- Generated samples are not readable English (see `samples.txt`).
- 19.5% accuracy vs the 80.71% target — the gap is the representation.
- Gradients are truncated to the current tick; there is no BPTT, so the
  field cannot be trained *to remember*, only to react.
