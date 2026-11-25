import numpy as np
import torch

from histcompare import synthetic as syn
from hydra_mu2e import model as M
from hydra_mu2e import training_data as td
from hydra_mu2e.render import render


def test_forward_shape():
    m = M.PlotCNN()
    out = m(torch.zeros(5, 1, 128, 128))
    assert out.shape == (5,)


def test_predict_is_a_probability():
    p = M.predict(M.PlotCNN(), np.random.rand(4, 1, 128, 128).astype(np.float32))
    assert p.shape == (4,) and ((0 <= p) & (p <= 1)).all()


def test_learns_dead_feb_vs_normal():
    """A small, clearly separable problem trains to high accuracy in a few epochs."""
    rng = np.random.default_rng(0)
    w = syn.detector_map(0)
    X, y = [], []
    for i in range(160):
        kind = "dead_feb" if i % 2 else "normal"
        X.append(render(td.sample(syn.DIGIS, kind, rng, w), syn.CH_EDGES))
        y.append(float(i % 2))
    X, y = np.stack(X)[:, None], np.array(y, dtype=np.float32)
    model, hist = M.train(X[:120], y[:120], epochs=10, seed=0, val_frac=0.0)
    acc = ((M.predict(model, X[120:]) > 0.5) == (y[120:] > 0.5)).mean()
    assert acc >= 0.9
    assert hist.train_loss[-1] < hist.train_loss[0]
