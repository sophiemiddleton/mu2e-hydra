import numpy as np
import pytest

from histcompare import synthetic as syn
from hydra_mu2e import training_data as td
from hydra_mu2e.render import SIZE, render

W = syn.detector_map(0)


def rng(seed=0):
    return np.random.default_rng(seed)


def test_every_family_has_good_and_bad():
    for fam, kinds in td.KINDS.items():
        labels = {label for _, label, _ in kinds}
        assert labels == {td.GOOD, td.BAD}, fam


def test_no_led_drops_only_led():
    for name in td.HISTOGRAMS:
        full = [k[0] for k in td.kinds(name)]
        no_led = [k[0] for k in td.kinds(name, include_led=False)]
        assert "led" in full and "led" not in no_led
        assert set(full) - set(no_led) == {"led"}


def test_masked_channels_always_empty():
    for kind, _, _ in td.KINDS["digis"]:
        c = td.sample(syn.DIGIS, kind, rng(), W)
        assert c.shape == (syn.N_CHANNELS,)
        assert (c[list(syn.MASKED)] == 0).all(), kind


def test_dead_feb_is_a_whole_empty_feb():
    c = td.sample(syn.DIGIS, "dead_feb", rng(1), W)
    febs = c.reshape(-1, syn.FEB_SIZE).sum(axis=1)
    assert (febs == 0).sum() == 1


def test_dead_block_is_at_least_8_adjacent_empty_channels():
    c = td.sample(syn.DIGIS, "dead_block", rng(2), W)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[1, (c == 0).astype(int), 1]) != 0))[::2]
    assert runs.max() >= 8


def test_isolated_dead_adds_at_most_two_empty_channels():
    for s in range(20):
        c = td.sample(syn.DIGIS, "isolated_dead", rng(s), W)
        assert 1 <= (c == 0).sum() - len(syn.MASKED) <= 2


def test_hot_channel_stands_out():
    c = td.sample(syn.DIGIS, "hot_channel", rng(3), W)
    live = c[c > 0]
    assert c.max() > 4 * np.median(live)


def test_gain_drop_moves_the_peak_down():
    peak = lambda c: syn.PE_EDGES[np.argmax(c)]
    normal = [peak(td.sample(syn.PES[0], "normal", rng(s), W)) for s in range(10)]
    drop = [peak(td.sample(syn.PES[0], "gain_drop", rng(s), W)) for s in range(10)]
    assert max(drop) < min(normal)


def test_low_pe_noise_fills_the_low_bins():
    low = syn.PE_EDGES[:-1] < 12
    frac = lambda c: c[low].sum() / c.sum()
    assert frac(td.sample(syn.PES[1], "low_pe_noise", rng(4), W)) > \
        3 * frac(td.sample(syn.PES[1], "normal", rng(4), W))


def test_sampling_is_deterministic():
    a = td.sample(syn.DIGIS, "hot_channel", rng(7), W)
    b = td.sample(syn.DIGIS, "hot_channel", rng(7), W)
    assert np.array_equal(a, b)


def test_render_shape_and_range():
    img = render(td.sample(syn.PES[0], "normal", rng(), W), syn.PE_EDGES)
    assert img.shape == (SIZE, SIZE) and img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0 and img.mean() > 0.01


def test_render_empty_histogram():
    img = render(np.zeros(100), syn.PE_EDGES)
    assert np.isfinite(img).all()


@pytest.mark.parametrize("name", td.HISTOGRAMS)
def test_make_set_labels_match_kinds(name):
    X, y, k = td.make_set(name, 2, seed=0)
    assert X.shape[1:] == (1, SIZE, SIZE) and len(X) == len(y) == len(k)
    label = {kind: lab for kind, lab, _ in td.kinds(name)}
    assert all(y[i] == label[k[i]] for i in range(len(k)))


def test_training_never_copies_the_benchmark_runs():
    """Different seeds and random fault parameters: no training image equals a benchmark one."""
    X, _, _ = td.make_set(syn.PES[0], 3, seed=1000)
    rng_b = np.random.default_rng(105004)  # the benchmark's gain-drop run seed
    bench = render(syn.pe_hist(rng_b, syn.N_PULSES, "cosmics", mpv=34.0), syn.PE_EDGES)
    assert not any(np.array_equal(x[0], bench) for x in X)
