"""
Histogram -> the image the classifier sees

Hydra classifies the plot images shifters look at. Here a histogram is drawn as a filled
step plot with a fixed frame and no tick labels (unreadable at this size anyway), x range
fixed by the bin edges and y autoscaled to the maximum, then read back as a grayscale
array in [0, 1]. Like Hydra, the classifier gets this one image only: no reference run.
"""

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SIZE = 128  # pixels per side

_fig = _ax = None


def _axes():
    global _fig, _ax
    if _fig is None:
        _fig = plt.figure(figsize=(2, 2), dpi=SIZE // 2)
        _ax = _fig.add_axes([0.04, 0.04, 0.92, 0.92])
    return _fig, _ax


def render(counts: np.ndarray, edges: np.ndarray, png: Optional[str] = None) -> np.ndarray:
    """(SIZE, SIZE) float32 image of the histogram, 1 = ink; optionally also saved as a PNG."""
    fig, ax = _axes()
    ax.cla()
    ax.stairs(counts, edges, fill=True, color="black")
    ax.set_xlim(edges[0], edges[-1])
    top = counts.max()
    ax.set_ylim(0, 1.1 * top if top > 0 else 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = 1.0 - rgba[..., :3].mean(axis=-1) / 255.0
    if png:
        plt.imsave(png, img, cmap="gray_r", vmin=0, vmax=1)
    return img.astype(np.float32)
