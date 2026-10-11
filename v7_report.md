# VN v7 — does scale buy anything?

Four experiments, all aimed at one question: **does adding neurons to the
concept pool improve the model?** Every one answers *no*, and together they
explain why v6's "lossless 5x compression" was an artefact rather than a
result.

All numbers below are measured, not inferred. Environment: Python 3.10,
NumPy, CPU only. Data: TinyShakespeare, 65 symbols, unigram baseline
4.7914 bits, uniform 6.022 bits. Optimiser: shared Adam, lr 3e-3.

---

## Experiment A — pool-size scan

Same budget (100k steps, lr 3e-3), three seeds each, only `N` varies.

| N | params | mean val bits | seed std | per-seed |
|---|---|---|---|---|
| **20** | **4,505** | **3.8478** | 0.0138 | 3.8546 / 3.8285 / 3.8603 |
| 50 | 8,825 | 3.8836 | 0.0336 | 3.8459 / 3.8774 / 3.9275 |
| 105 | 16,745 | 3.9734 | 0.0401 | 3.9652 / 3.9290 / 4.0261 |
| 200 | 30,425 | 3.9105 | 0.0267 | 3.9237 / 3.8733 / 3.9346 |
| 1000 | 145,625 | 3.9087 | — | (v6 baseline, 1 seed) |

**N=20 is the best of the lot.** The gap to N=105 is 0.126 bits, which is
3–9x the seed noise (0.014–0.040). Going from 20 to 105 neurons — a 3.7x
increase in parameters — makes the model *worse*, not better.

So the v6 baseline (N=1000, 145,625 params) was not "a big model that we
compressed"; it was **an over-parameterised model whose excess capacity was
doing nothing**. Compressing it to 9,440 params and losing only 0.009 bits
is exactly what you would expect when the extra 136k parameters were never
carrying information.

The sharper form of the same point: v6's best compressed model scored
**3.9176** at 9,440 params, while simply *training a small model from
scratch* (N=50, 8,825 params) scores **3.8836**. Compression is beaten by
just training the right size in the first place.

## Experiment B — what training does to the concept geometry

Nearest-neighbour cosine (median over rows), N=200, checkpoints:

| checkpoint | nn_p50 | mean \|cos\| | pairs ≥0.99 | ≥0.95 | ≥0.90 |
|---|---|---|---|---|---|
| init | 0.2609 | 0.0806 | 0 | 0 | 0 |
| 10k | **0.8090** | 0.3138 | 0 | 0 | 3 |
| 20k | 0.7887 | 0.2907 | 0 | 0 | 0 |
| 50k | 0.7398 | 0.2481 | 0 | 0 | 0 |
| 100k | **0.6929** | 0.2217 | 0 | 0 | 0 |

Two things:

1. **Concepts first cluster, then spread apart.** From random init (0.261)
   they rush together by 10k steps (0.809), then monotonically diverge for
   the remaining 90k (0.809 → 0.693). Training is driving neurons toward
   *functional differentiation*, i.e. toward mutual orthogonality — not
   toward "similar concepts live at the same address".

2. **No duplicate concepts ever appear.** `cos ≥ 0.99` is 0 at every
   checkpoint; even `≥ 0.95` is 0. The first stage of v6 — "find identical
   concepts and fuse them losslessly" — has literally nothing to fuse. It
   removed 0 neurons at the specified 0.99 threshold.

The gate-key half and the weight half both diverge, so this is a property
of the whole concept vector, not of one component.

## Experiment C — is big-N only under-trained?

The strongest defence of scale: a bigger pool needs more optimisation
steps. Tested directly — N=1000 trained for **400k steps** (4x the budget):

| step | 66,666 | 133,332 | 199,998 | 266,664 | 333,330 | 399,996 |
|---|---|---|---|---|---|---|
| val bits | 4.0313 | 4.0252 | 3.9108 | **3.8637** | 3.9907 | 4.0594 |

Best point is 3.8637 at 266k steps. That is **still worse than N=20's
3.8478**, which cost 1/4 of the training budget and 1/32 of the
parameters. The curve also swings by 0.20 bits across checkpoints, so the
large pool is not merely unrewarded — it is *unstable*.

## Experiment D — is it a learning-rate artefact?

`pooled` is scaled by `1/sqrt(N)`, so gradients shrink as the pool grows.
If that were the whole story, `lr * sqrt(N/20)` would restore parity. It
does not — it is clearly harmful at both sizes:

| N | lr 3e-3 | lr × √(N/20) | effect |
|---|---|---|---|
| 105 | 3.9652 / 3.9290 | 4.4534 / 4.5640 | **much worse** |
| 1000 | 3.9087 | 4.9610 | **much worse** |

The "big N just needs a bigger lr" defence is dead too. (Note this also
means the 1/√N scaling is doing its intended job — the residual N
dependence is not a learning-rate phenomenon.)

---

## What this adds up to

Four assumptions behind the VN programme, each now tested:

| assumption | verdict |
|---|---|
| more neurons ⇒ better model | **false** — N=20 beats N=105/200/1000 |
| big N is merely under-trained | **false** — 4x budget still loses; lr matching hurts |
| similar concepts converge to shared addresses | **false** — they diverge after 10k steps |
| duplicate concepts exist and can be fused losslessly | **false** — zero pairs ≥0.99, ever |
| v6's 5x compression extracted structure | **false** — it removed redundancy; training small wins outright |

None of this is a claim that a spiking/wave architecture *cannot* work.
It is a claim that **on this task, at this scale, with this update rule,
the pool size is not where the value is**. The best model in the whole
sweep is the smallest one (4,505 params, 3.8478 bits), and every mechanism
layered on top of a large pool — fusion, clustering, shared bases, concept
waves — has been optimising a quantity that was flat to begin with.

## Caveats

- All runs are character-level TinyShakespeare with a 2-character context
  (`x = emb[cur] + 0.5*emb[prev]`). A richer input encoding could change
  how much capacity is usable; that is untested here.
- N=1000 baselines are 1 seed (v6 artifact) while the scan is 3 seeds.
- Experiment C's curve is single-seed; its *best* point is compared against
  a 3-seed mean, which favours C.
- `nn_p50` is measured on `concept = [key | vec(W)]`; per-component numbers
  are in `v7_disperse.json` and show the same trend.

## Reproduce

```bash
python3 v7_nscan.py   20,50,105,200      0,1,2 100000 v7_nscan_a.json
python3 v7_disperse.py 200 10000,20000,50000,100000 v7_disperse.json
python3 v7_long.py    1000 400000 0      v7_long.json
python3 v7_lr.py
python3 -m pytest tests/test_v7_scale.py -q
```
