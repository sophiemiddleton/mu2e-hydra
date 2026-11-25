# mu2e-hydra

(Formerly HYDRA-Mu2e. The Python package is `hydra_mu2e`.)

A Hydra-style classifier of Mu2e DQM plots, for comparison with the vision LLM in
[../mu2e-vision](../mu2e-vision/README.md) and the KS test.

[Hydra](https://arxiv.org/abs/2403.00689) (Britton, Jeske, Lawrence, Rajput; JLab, in use
since 2019) classifies DQM plot images with trained CNNs, one model per plot type. It
uses no language model and no reference run. It needs labelled training images. This
is not JLab's code. It's a small reimplementation of the idea, for Mu2e CRV plots.

| | KS | Hydra-style CNN (here) | Vision LLM (mu2e-vision) |
|---|---|---|---|
| Sees | Bin contents vs a reference run | The plot image alone | Input and reference images, shifter instructions, run type |
| Needs | A golden run, thresholds | Labelled images per plot | A YAML entry per plot, an API key |
| Gives | A p-value | P(BAD) | A verdict with a written reason |

## Running it

```bash
source mu2e-hydra/setup_env.sh       # sources ../mu2e-vision/setup_env.sh (torch is in the ana env)
cd mu2e-hydra

python -m pytest -q tests                                    # 21 tests, ~15 s
python -m hydra_mu2e.benchmark --out results/default \
    --compare ../results/opus-5-5/results.csv                # ~5 min on 4 CPU cores
python -m hydra_mu2e.benchmark --out results/no-led --no-led # LED runs left out of training
```

The benchmark generates the synthetic set in `synthetic/` if it isn't there, using
mu2e-vision's `histcompare.synthetic`, so all three methods are scored on the same 27
comparisons. `--compare` adds the KS and vision verdicts from a mu2e-vision
`results.csv`. Outputs: `summary.md`, `results.csv`, `heldout.csv`, the input images in
`images/` and the trained models in `models/`.

## How it works

| File | Role |
|---|---|
| `hydra_mu2e/training_data.py` | Labelled training images: the benchmark's detector map and binning, with each fault at a random strength and position, at 4%–3x normal statistics |
| `hydra_mu2e/render.py` | Histogram → 128×128 grayscale plot image (fixed frame, no tick labels) |
| `hydra_mu2e/model.py` | `PlotCNN`: 4 conv blocks and a dense head; training and prediction |
| `hydra_mu2e/benchmark.py` | Trains one model per histogram and scores it on the benchmark set and on a held-out set |
| `tests/` | Unit tests of the fault generator, renderer and model, and an end-to-end smoke run |

Training kinds (label):

- **PE histograms:** normal, gain drop of 12–35% (BAD), low-PE noise (BAD), LED (GOOD)
- **Digis per channel:** normal, 1–2 isolated dead channels (GOOD), a dead FEB (BAD), a dead
  block of 8–40 channels (BAD), a hot channel 5–40x (BAD), LED (GOOD)

No benchmark run is copied into training. The faults are drawn at random strengths and
positions, using different seeds.

## Results (2026-11-15)

| Method | BAD caught | BAD missed | GOOD flagged | Agreement |
|---|---|---|---|---|
| Hydra-style CNN | 4 | 0 | 0 | 27/27 |
| KS (standard) | 4 | 0 | 3 | 24/27 |
| Vision LLM (Opus 5.5) | 4 | 0 | 0 | 27/27 |

The CNN also scores 100% on the held-out set of every fault kind. Two cautions:

- **The synthetic set is too easy to rank these methods.** Every fault is far above
  threshold. A sweep of fault strength (a gain drop of 2–15%, a hot channel at 1.5–5x)
  would show where each method stops seeing it.
- **Without LED training, the CNN still calls LED runs GOOD** (P(BAD) ≈ 0.001). That's
  the right answer here, but for the wrong reason: a binary classifier puts a shape it has
  never seen on whichever side it resembles more. An unseen fault can just as easily
  pass as GOOD. The vision LLM is told what an LED run is. KS flags any change.

## Real CRV data (crvreco-006-001), 2026-11-16

`hydra_mu2e/realdata.py` trains on real shapes. It takes subrun histograms from 77–79 006-001
runs that are not among the 21 scored runs (466 subruns, bank in `bank/`, git-ignored),
sums a random number of them, and injects faults (gain shift, shoulder loss, chi2 bump or
shift, ADC shift or saturation, dead/parity-dead/low/hot FEBs, hot channel). The untouched
images are labelled GOOD without anyone checking them. Four shape histograms only:
errorBits and activeFebCount are counters, not pictures.

```bash
python -m hydra_mu2e.realdata collect --exclude ../mu2e-vision/real/006-001 --out bank/006-001.npz
python -m hydra_mu2e.realdata run --bank bank/006-001.npz --data ../mu2e-vision/real/006-001 \
    --out results/real-006-001-C --drop-runs 124528 125848 --max-run 125200 \
    --compare ../mu2e-vision/results/real-006-001-ref124228-opus55/results.csv   # ~13 min
```

| Variant | Training runs | CNN BAD | Same verdict as Opus 5.5 (ref 124228) |
|---|---|---|---|
| A | all | 11/84 | 56/80 |
| B | minus 124528 and 125848 (same dead blocks as 125846) | 17/84 | 60/80 |
| C | B, and only runs ≤ 125200 (the reference's era) | 30/84 | 73/80 |

The vision LLM called 33/80 BAD; KS called 80/80. h1_peakAdc is GOOD for every run by both.

- In A the CNN passed 125846 (P(BAD) = 0.05), because two training runs had the same dead
  channels and were taught to it as GOOD. Hydra is only as good as its labels.
- A and B call the late-period PE shoulder (~25 → ~36 PE) GOOD, because late runs are in
  training. C calls it BAD, like the vision model. The CNN is just learning "normal", and
  normal is whatever you train it on. Whether the shift is a fault is a CRV question.
- The digi-map CNN false-alarms on 22–27% of held-out normal images (PEr is 26% in C), so
  some of its agreement on crvDigisPerChannel is coincidence.
