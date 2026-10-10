import numpy as np
import pytest

from vn_mobile.core.group import build_blocks, hot_blocks
from vn_mobile.core.neuron import (
    DIM,
    MODE_BACKGROUND,
    RANK,
    VNPopulation,
    build_population,
    dequantise_u8,
    quantise_u8,
)
from vn_mobile.core.update import (
    aggregate,
    neighbour_mean,
    normalise_energy,
    read_output,
    step_energy,
    step_state,
)
from vn_mobile.core.field import Field


def make_pop(n=64, degree=8, seed=1):
    return build_population(n=n, degree=degree, seed=seed)


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
    assert float(pop.energy.sum()) <= 0.3 * pop.n + 1e-4


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
