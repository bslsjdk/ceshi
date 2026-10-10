# Stage 5A closing run

Branch `main`, 100k steps, seed 0. `python3 diag100k.py` -> `diag100k.json`.

## Criteria

| Criterion | Target | Before | Now |
|---|---|---|---|
| merge rate (relative, best 12%/zone) | 5-20% | 0.484 | **0.120** PASS |
| cull rate (relative, worst 40%/zone) | 40%+/-10% | 0.400 | **0.400** PASS |
| score_std | > 0.05 | 0.165 | **0.106** PASS |
| MSE drop | > 30% | 0.567 | **0.408** PASS |
| BN output distinguishable | corr < 0.8 | 0.003 | **0.009** PASS |
| no NaN / e_max < 0.95 | | ok | **ok / 0.517** PASS |
| FVN frozen, TVN fires, EZ load+release | | ok | **ok (915/912/5)** PASS |
| hibernate byte-identical | True | True | **True** PASS |
| archive ratio > 3x | > 3 | 5.235 | **5.235** PASS |
| rollback / switch / leak | ok / <100ms / 0 | ok | **ok / 0.4ms / 0MB** PASS |
| **`between > within`** | > 1 | 0.128 | **REMOVED** (criterion abolished) |
| **`between_zone_std`** | > 0.01 | - | **0.0114** PASS |
| **`corr(score, -horizon)`** | > 0.7 | - | **0.1297** FAIL |
| **us/step** | < 200 | 803 | **308** FAIL (see layers) |

## Performance: 803 -> 308 us/step

`python3 bench_step.py` -> `bench_step.json`

| Layer | Before | After |
|---|---|---|
| 1 main field only (1000 VN: energy+phase+state) | 520.6 | **177.8** |
| 2 + trial-zone tick | - | 221.9 |
| 3 + sampling (shadow/train/evaluate every 10) | - | 271.4 |
| 4 full `run_5a` | 803 | 306.9 |

The core 1000-neuron loop is under the 200 us target. The full 5A harness is
not, because it steps two populations (main + TZ) and runs the trial sampling.

Three rewrites, each pinned by an equivalence test in `tests/test_opt.py`:

- `step_energy`/`step_phase`: `cos(a-b)`/`sin(a-b)` expanded, so the per-edge
  trig pass over E=15880 edges becomes two sparse matvecs over N=1000.
  94 -> 22 us and 73 -> 20 us.
- `aggregate`: `csr_matrix @ values` replaces gather + segmented reduce;
  the gather alone (an [E] or [E, DIM] temporary every tick) was 115 us.
  197 -> 47 us for the [N, 8] case.
- `step_state`/`_predict`: `W` and `B@A` fused into one cached [N, 8, 8]
  stack, so `Wx + B(Ax)` is a single batched matmul. 48 -> 19 us.
- `evaluate`: per-zone median/std vectorised (one lexsort + two bincounts)
  instead of 40 numpy calls per evaluation.

## Why `horizon_corr` is 0.13 and not > 0.7

Zone look-ahead ranges 3-12 samples (25-120 ticks). Measured correlations of
zone mean score:

- with horizon: **0.130**
- with the zone's random drive amplitude: **0.215**

The spread tracks the random per-zone drive level slightly more than the
deliberate difficulty gradient, i.e. the horizon difference is not what
separates the zones. Beyond ~3 samples of look-ahead the shadow signal has
lost its autocorrelation, so every long-horizon zone is equally unpredictable
and the BN settles on predicting the mean -- there is no gradient left for the
score to follow.

`amp_corr` being only 0.215 too means most of the between-zone spread is not
explained by either factor.

## Caveats

- `between_zone_std` is seed- and length-sensitive: 0.0114 at seed 0 / 100k,
  but 0.0052 at seed 1 / 20k. The 0.01 threshold is close to the noise floor.
- `test_coupling_lifts_sync_without_locking` was rewritten. Its old assertion
  read a single end-of-run snapshot of R, which swings between 0.02 and 0.35
  every few hundred ticks; it passed only because the trajectory was
  bit-reproducible, and a 2e-7 change in the coupling term moved the snapshot
  from 0.303 to 0.096 while the time average stayed at 0.21. It now averages R
  over the run: 0.067 at K=0.05 vs 0.210 at K=0.5.
- Dead fields removed: `prev_shadow`, `zero_util_steps` (leftovers of the
  absolute-threshold cull that relative culling replaced).
