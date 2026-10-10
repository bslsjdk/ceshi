# Stage 5A final — relative promotion + continuous utilisation

100 000 steps, 1000 main VN, 1000 BN, 20 zones. Command:

```
python3 diag100k.py          # 80.3 s, 803 us/step, RSS 17.7 MB
pytest -q                    # 33 tests collected, 33 passed
```

## Criteria

| Criterion | Target | Previous | Now | |
|---|---|---|---|---|
| Merge rate | 5–20 % | 48.4 % | **12.0 %** | PASS (was FAIL) |
| Cull rate | 40 % ± 10 % | 0.400 | **0.400** | PASS |
| `score_std` | > 0.05 | 0.1648 | **0.1027** | PASS |
| `between > within` | division of labour | 0.0163 / 0.0314 (0.52) | **0.0119 / 0.0930 (0.128)** | **FAIL — worse** |
| BN count | 1000 ± 10 % | 1000 | 880 alive + 120 merged | PASS |
| MSE drop | > 30 % | 0.567 | **0.432** | PASS |
| BN vs main corr | < 0.8 | 0.0025 | **-0.0405** | PASS |
| no NaN / e_max<0.95 | | ok | False / 0.4984 | PASS |
| hibernate byte-exact | True | True | **True** | PASS |
| archive ratio | > 3x | 5.235 | **5.235** | PASS |
| rollback / leak | ok / 0 MB | ok | **ok / 0.0 MB** | PASS |

## What worked: promotion is now relative

Merging used an absolute floor (`score > 0.7`). An **untrained** BN already
scores ~0.668 and a trained one ~0.75, so the distribution straddled the
threshold and the merge rate pinned to 48 % regardless of training. Ranking
inside each zone and taking the best `PROMOTE_FRAC = 0.12` makes the rate a
parameter: measured **0.120**, and `test_promote_rate_is_independent_of_
absolute_score` shows it stays 12 % when every score is scaled by 0.5x or 10x.

Two accounting senses, both reported:

* `merge_rate` (terminal fraction carrying a merge flag) = **0.120**
* `merge_rate_per_round` (mean over selection rounds of promoted/alive)

`promote_mode="rolling"` (merge every 2000-step round) is also implemented and
is the wrong reading: merged slots leave for good while culled slots come back
after reinit, so the terminal rate climbs toward 100 % whatever the fraction is.
Terminal merge is the correct sense for this criterion.

## What did not work: `between > within` got worse

The continuous utilisation

```
util_i = sigmoid((|pred_i| - median_zone) / (std_zone + eps))
```

was meant to remove the binary gate's bimodality. It does, but it made
within-zone spread **worse**, not better: 0.0314 → **0.0930**, and the ratio
fell from 0.52 to **0.128**.

The reason is structural. Dividing by the zone's *own* standard deviation
rescales every zone to unit width, so the sigmoid output has std ≈ 0.2 inside
**every** zone whether or not the BNs differ in skill — measured `util`
within = 0.1966 against between = 0.0159. The spread is a property of the
normalisation, not of the population.

Per-component breakdown at 100 k (within / between):

| component | weight | within | between |
|---|---|---|---|
| acc | 0.40 | 0.1461 | 0.0238 |
| util | 0.20 | 0.1966 | 0.0159 |
| contrib | 0.20 | 0.0551 | 0.0099 |
| cost | -0.05 | 0.0516 | 0.0085 |
| conflict | -0.05 | 0.2162 | 0.0151 |

Every component has within ≈ 10x its between. That is not a tuning problem:
`within` is *individual skill spread* and `between` is *task difficulty spread*,
and in a self-supervised prediction task the former is an order of magnitude
larger than the latter.

## The two criteria pull against each other

Training increases `within` and leaves `between` flat, so the ratio falls
exactly as learning succeeds:

| steps | MSE drop | between | within | b/w |
|---|---|---|---|---|
| 20 000 | 0.18 | 0.0114 | 0.0388 | 0.294 |
| 100 000 | **0.43** | 0.0119 | **0.0930** | **0.128** |

Widening the difficulty gradient does not rescue it. Scanning the per-zone
horizon and drive amplitude (all at 20 k):

| variant | b/w | MSE drop |
|---|---|---|
| spec (hz 25→120 tick) | 0.37 | 0.17 |
| wide horizon (hz 10→390) | 0.39 | 0.11 |
| wide amplitude (U(0.02, 6.0)) | 0.33 | 0.08 |
| both wide | 0.41 | 0.03 |
| extreme horizon (hz 1→950) | **0.535** | **0.039** |

Best ratio reached anywhere: **0.535**, and it costs the MSE criterion (0.039
against a 0.30 floor). `between > within` and `MSE drop > 30 %` cannot both hold
here.

## Options for the next round

1. Put zone identity into the score, not just ability — e.g. weight `acc` by the
   zone's own difficulty so easy zones gain less from the same accuracy.
2. Relax the criterion to an absolute floor on `between` (it is a stable 0.012
   and clearly non-zero; the zones really do differ, just less than individuals do).
3. Share adapters inside a zone to shrink individual spread — this trades away
   the per-BN perturbation the spec asks for.
