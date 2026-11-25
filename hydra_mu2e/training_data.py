"""
Labelled training images, the part Hydra needs and a prompted vision LLM does not

Hydra trains one classifier per plot type on images that people have labelled. Mu2e has
no such labels yet, so they are generated here: the same detector as the benchmark set
(histcompare.synthetic's channel map, binning and PE shape), but with every fault drawn
at a random strength and place, and the statistics drawn log-uniformly between 4% and 3x
a normal run. No benchmark run is copied into training.

Each plot family has a list of kinds (label, relative frequency):

  pe     normal, gain_drop (BAD), low_pe_noise (BAD), led (GOOD)
  digis  normal, isolated_dead (GOOD), dead_feb (BAD), dead_block (BAD),
         hot_channel (BAD), led (GOOD)

`include_led=False` drops the LED kinds, which shows what happens to a run type the
classifier never saw (the benchmark's run 105008).
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import moyal

from histcompare import synthetic as syn

GOOD, BAD = 0, 1

KINDS: Dict[str, List[Tuple[str, int, float]]] = {
    "pe": [("normal", GOOD, 3), ("gain_drop", BAD, 1), ("low_pe_noise", BAD, 1),
           ("led", GOOD, 1)],
    "digis": [("normal", GOOD, 2), ("isolated_dead", GOOD, 1), ("dead_feb", BAD, 1),
              ("dead_block", BAD, 1), ("hot_channel", BAD, 1), ("led", GOOD, 1)],
}

HISTOGRAMS = (*syn.PES, syn.DIGIS)


def family(histogram: str) -> str:
    return "digis" if histogram == syn.DIGIS else "pe"


def edges(histogram: str) -> np.ndarray:
    return syn.CH_EDGES if family(histogram) == "digis" else syn.PE_EDGES


def kinds(histogram: str, include_led: bool = True) -> List[Tuple[str, int, float]]:
    return [k for k in KINDS[family(histogram)] if include_led or k[0] != "led"]


def _pe(rng, kind: str, n: int) -> np.ndarray:
    if kind == "led":
        pes = rng.normal(rng.uniform(50, 70), rng.uniform(4, 8), n)
    else:
        # normal gain varies a few percent; a gain drop is 12-35%
        mpv = rng.uniform(26, 35) if kind == "gain_drop" else rng.uniform(38, 42)
        pes = moyal.rvs(loc=mpv, scale=rng.uniform(5.5, 6.5), size=n, random_state=rng)
    if kind == "low_pe_noise":
        hi = rng.uniform(10, 15)
        pes = np.concatenate([pes, rng.uniform(syn.PE_THRESHOLD, hi, int(n * rng.uniform(1 / 15, 1 / 4)))])
    pes = pes[pes >= syn.PE_THRESHOLD]
    return np.histogram(pes, syn.PE_EDGES)[0].astype(float)


def _live(w: np.ndarray) -> np.ndarray:
    return np.flatnonzero(w > 0)


def _digis(rng, kind: str, n: int, weights: np.ndarray) -> np.ndarray:
    w = np.where(weights > 0, 1.0, 0.0) if kind == "led" else weights.copy()
    if kind == "isolated_dead":
        w[rng.choice(_live(w), rng.integers(1, 3), replace=False)] = 0.0
    elif kind == "dead_feb":
        feb = rng.integers(0, syn.N_CHANNELS // syn.FEB_SIZE)
        w[feb * syn.FEB_SIZE:(feb + 1) * syn.FEB_SIZE] = 0.0
    elif kind == "dead_block":
        length = rng.integers(8, 41)
        start = rng.integers(0, syn.N_CHANNELS - length)
        w[start:start + length] = 0.0
    elif kind == "hot_channel":
        w[rng.choice(_live(w))] *= rng.uniform(5, 40)
    return rng.poisson(n * w / w.sum()).astype(float)


def sample(histogram: str, kind: str, rng, weights: np.ndarray) -> np.ndarray:
    """Counts of one histogram of the given kind, at random statistics."""
    scale = np.exp(rng.uniform(np.log(0.04), np.log(3.0)))
    if family(histogram) == "pe":
        return _pe(rng, kind, int(syn.N_PULSES * scale))
    return _digis(rng, kind, int(syn.N_DIGIS * scale), weights)


def make_set(histogram: str, n_per_unit: int, seed: int, include_led: bool = True,
             render_fn=None):
    """(images [N,1,H,W], labels [N], kind names [N]) for one histogram.

    Each kind gets round(n_per_unit * its relative frequency) images.
    """
    from .render import render
    render_fn = render_fn or render
    rng = np.random.default_rng(seed)
    weights = syn.detector_map(0)  # the benchmark's detector
    e = edges(histogram)
    images, labels, names = [], [], []
    for kind, label, freq in kinds(histogram, include_led):
        for _ in range(int(round(n_per_unit * freq))):
            images.append(render_fn(sample(histogram, kind, rng, weights), e))
            labels.append(label)
            names.append(kind)
    return np.stack(images)[:, None], np.array(labels, dtype=np.float32), np.array(names)
