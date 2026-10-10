import numpy as np
import pytest

from vn_mobile.core.group import build_blocks, hot_blocks
from vn_mobile.core.neuron import (DIM, MODE_BACKGROUND, RANK, VNPopulation,
                                   build_population, dequantise_u8,
                                   quantise_u8)
from vn_mobile.core.update import (ENERGY_CAP, E_TARGET, aggregate,
                                   neighbour_mean, normalise_energy,
                                   read_output, step_energy, step_phase,
                                   step_state, sync_order_parameter)
from vn_mobile.core.field import Field


def make_pop(n=64, degree=8, seed=1, weight_sigma=0.1):
    return build_population(n=n, degree=degree, seed=seed,
                            weight_sigma=weight_sigma)


def test_field_bit_widths():
    pop = make_pop()
    assert pop.addr.dtype == np.uint32
    assert pop.route_key.dtype == np.uint32
    assert pop.adapter_ref.dtype == np.uint32
    assert pop.mode.dtype == np.uint8 and pop.mode.max() <= 7
    assert pop.health.dtype == np.uint8 and pop.health.max() <= 31
    assert pop.mode.dtype.itemsize == 1
    assert pop.direction.shape == (pop.n, DIM)


def test_quantise_roundtrip():
    x = np.linspace(0.0, 6.28, 257, dtype=np.float32)
    q = quantise_u8(x, 0.0, 6.28)
    assert q.dtype == np.uint8
    back = dequantise_u8(q, 0.0, 6.28)
    assert np.abs(back - x).max() <= 6.28 / 255.0 + 1e-6


def test_graph_is_undirected_and_loop_free():
    pop = make_pop()
    assert (pop.src != pop.dst).all()
    forward = set(zip(pop.src.tolist(), pop.dst.tolist()))
    backward = set(zip(pop.dst.tolist(), pop.src.tolist()))
    assert forward == backward


def test_csr_ptr_matches_edges():
    pop = make_pop()
    assert pop.ptr[0] == 0 and pop.ptr[-1] == pop.n_edges
    counts = np.diff(pop.ptr)
    assert (counts == pop.deg.astype(np.int64)).all()


def test_aggregate_matches_manual():
    pop = make_pop(n=32, seed=3)
    vals = np.arange(pop.n, dtype=np.float32) * 0.5
    got = aggregate(pop, vals)
    want = np.zeros(pop.n, dtype=np.float32)
    for s, d in zip(pop.src.tolist(), pop.dst.tolist()):
        want[s] += vals[d]
    assert np.allclose(got, want, atol=1e-5)
    assert np.allclose(neighbour_mean(pop, vals), want / pop.deg, atol=1e-5)


def test_weight_mean_is_zero():
    pop = make_pop()
    assert abs(float(pop.weight.mean())) < 1e-6


def test_single_step_is_finite():
    pop = make_pop(seed=5)
    rng = np.random.default_rng(0)
    inj = np.zeros((pop.n, DIM), dtype=np.float32)
    step_energy(pop, rng)
    step_state(pop, inj)
    assert np.isfinite(pop.energy).all()
    assert np.isfinite(pop.state).all()


def test_normalise_scales_total_only_when_over():
    pop = make_pop(n=32, seed=7)
    pop.energy[:] = 0.01
    assert normalise_energy(pop) is False
    pop.energy[:] = 5.0
    assert normalise_energy(pop) is True
    assert float(pop.energy.sum()) <= ENERGY_CAP * pop.n + 1e-4


def test_output_scales_with_energy_and_state():
    pop = make_pop(n=16, seed=9)
    pop.state[:] = 0.5
    pop.energy[:] = 0.0
    assert np.abs(read_output(pop)).max() == 0.0
    pop.energy[:] = 2.0
    assert np.allclose(read_output(pop), 1.0)


def test_blocks_cover_every_neuron():
    pop = make_pop(n=100, seed=11)
    layout = build_blocks(pop, block_size=64)
    assert sorted(layout.order.tolist()) == list(range(pop.n))
    assert layout.n_blocks == 2
    hot = hot_blocks(pop, layout, top_k=2)
    assert len(hot) == 2


def test_short_run_has_no_nan_and_stays_bounded():
    f = Field(make_pop(n=128, seed=13), seed=13)
    st = f.run(1000, log_every=0)
    assert st.nan is False
    assert st.step == 1000
    assert st.energy_max < 0.95
    assert st.top_frac < 0.01
    assert np.isfinite(f.pop.state).all()


# ---------------------------------------------------------------- stage 2

def test_phase_stays_wrapped():
    pop = make_pop(n=64, seed=21)
    for _ in range(500):
        step_phase(pop)
    assert (pop.phase >= 0.0).all()
    assert (pop.phase < 2.0 * np.pi).all()


def test_sync_order_parameter_in_unit_range():
    pop = make_pop(n=64, seed=23)
    pop.phase[:] = 0.0
    assert sync_order_parameter(pop) == pytest.approx(1.0)
    pop.phase[:] = np.linspace(0.0, 2.0 * np.pi, pop.n, endpoint=False)
    assert sync_order_parameter(pop) == pytest.approx(0.0, abs=1e-6)


def test_energy_fixed_point_sits_at_target():
    """The restoring term pins the mean at e_target, with no bifurcation."""
    for sigma in (0.1, 1.0, 4.0):
        f = Field(make_pop(n=256, seed=31, weight_sigma=sigma), seed=31)
        st = f.run(3000)
        assert st.nan is False
        # spec allows +/-20%; strong drive pulls the mean slightly below
        # e_target because mean(tanh(g)) is not exactly zero
        assert abs(st.energy_mean - E_TARGET) / E_TARGET < 0.20, sigma


def test_coupling_lifts_sync_without_locking():
    """R rises with K but stops short of locking.

    Under uniform omega_std=0.1 this used to assert R>0.9 at K=0.5.  Stage 4
    layered omega (80% @0.3 + 20% @1.0) so the bulk keeps a sync frame while a
    minority keeps drifting: measured R is ~0.30 at K=0.5, ~0.04 at K=0.05.
    Locking is exactly what the layering exists to prevent.
    """
    rs = {}
    for k in (0.05, 0.5):
        f = Field(make_pop(n=256, seed=37), seed=37, k_coupling=k)
        rs[k] = f.run(4000).sync_r
    assert rs[0.05] < 0.3
    assert rs[0.5] > 0.25
    assert rs[0.5] < 0.9


def test_weak_drive_leaves_field_flat():
    """Documents why sigma=0.1 is not viable: e_std collapses to ~2% of mean."""
    f = Field(make_pop(n=256, seed=41), seed=41)
    st = f.run(3000)
    assert st.energy_std < 0.02
    assert st.neg_frac == 0.0


def test_wave_spreads_outward_from_seed():
    """Seed decays while outer shells light up -> the disturbance travels."""
    from diffusion import bfs_layers, run_diffusion
    res = run_diffusion(n=400, degree=8, seed=0, sigma=0.5, k_coupling=0.5,
                        warmup=1500, observe=60, amp=0.5, max_layers=4)
    trace = res["trace"]
    seed_layer, outer = trace[:, 0], trace[:, 2]
    assert seed_layer[0] > 0.2                 # seed is disturbed
    assert seed_layer[-1] < seed_layer[0]      # and decays
    assert outer[-1] > outer[0] * 2            # while the shell lights up


def test_encoder_input_does_not_move_the_energy_field():
    """Known gap: e and h are decoupled, so a text input cannot steer the wave.

    The causal effect is measured against a control run because the field
    drifts on its own; comparing against its own baseline would be meaningless.
    """
    from diffusion import run_diffusion
    res = run_diffusion(n=200, degree=8, seed=0, sigma=4.0, k_coupling=0.2,
                        warmup=800, observe=30, amp=0.5, max_layers=3)
    assert res["encoder_h_after"] > 0.1        # the state layer does respond
    assert res["encoder_e_shift"] < 1e-6       # but the energy field does not
