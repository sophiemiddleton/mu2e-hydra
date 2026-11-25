"""
Hydra-style CNNs on real CRV DQM runs (crvreco-006-001), side by side with the vision LLM

Hydra needs labelled plot images. These runs have no labels, so the training images are
built from real data: subrun histograms from 006-001 runs that are NOT among the scored runs,
summed over a random number of subruns (so the statistics span one subrun to a whole run),
then left as they are (label GOOD) or given one injected fault (label BAD). The faults are
the ones the draft shifter instructions (mu2e-vision instructions/crv_real_draft.yaml)
ask a shifter to report.

The "GOOD" training images are whatever those runs looked like. Nobody checked them, so
any real problem in them, and every detector configuration in the dataset, is taught to the
CNN as normal. Hydra's real training labels come from people.

  collect  read the training runs' subrun histograms into a bank (npz)
  run      train one CNN per histogram, score the runs in a mu2e-vision data directory,
           and put the KS and vision verdicts beside the CNN's

    python -m hydra_mu2e.realdata collect --exclude ../mu2e-vision/real/006-001 \
        --out bank/006-001.npz
    python -m hydra_mu2e.realdata run --bank bank/006-001.npz \
        --data ../mu2e-vision/real/006-001 --out results/real-006-001 \
        --compare ../mu2e-vision/results/real-006-001-ref124228-opus55/results.csv
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import uproot

from . import model as M
from .render import render

HISTOGRAMS = {
    "PEr": "crvRecoDQM/PEr",
    "logchi2": "crvRecoDQM/logchi2",
    "crvDigisPerChannel": "crvDigiDQM/crvDigisPerChannel",
    "h1_peakAdc": "crvDigiDQM/h1_peakAdc",
}
GOOD, BAD = 0, 1

# (kind, label, relative frequency) per histogram
KINDS = {
    "PEr": [("normal", GOOD, 4), ("gain_down", BAD, 1), ("gain_up", BAD, 1),
            ("shoulder_loss", BAD, 1), ("low_pe_excess", BAD, 1)],
    "logchi2": [("normal", GOOD, 3), ("shift", BAD, 1), ("high_bump", BAD, 1), ("broaden", BAD, 1)],
    "h1_peakAdc": [("normal", GOOD, 3), ("shift", BAD, 1), ("saturation", BAD, 1),
                   ("below_baseline", BAD, 1)],
    "crvDigisPerChannel": [("normal", GOOD, 3), ("isolated_dead", GOOD, 1), ("dead_febs", BAD, 1),
                           ("parity_dead", BAD, 1), ("low_block", BAD, 1), ("hot_block", BAD, 1),
                           ("hot_channel", BAD, 1)],
}
FEB = 64


# ---------------------------------------------------------------- the training bank

def training_runs(survey: Path, exclude: set, min_live: int = 1850, min_digis_per_event: float = 20):
    """Runs of the survey with a populated detector, minus the excluded (scored) runs."""
    out = {}
    for r in csv.DictReader(open(survey)):
        if r["run"] in exclude or int(r["events"]) <= 0:
            continue
        if int(r["live_channels"]) >= min_live and int(r["digis"]) / int(r["events"]) > min_digis_per_event:
            out[r["run"]] = int(r["files"])
    return out


def _read(path):
    with uproot.open(path) as f:
        return {k: f[v].to_numpy() for k, v in HISTOGRAMS.items() if v in f}


def collect(survey: Path, exclude_dir: Path, out: Path, max_files: int, dataset: str, workers: int = 16):
    from histcompare.realdata import run_files
    exclude = set(json.loads((exclude_dir / "runs.json").read_text())["runs"])
    runs = training_runs(survey, exclude)
    files = run_files(dataset)
    rng = np.random.default_rng(0)
    todo = []
    for run in runs:
        paths = files.get(run, [])
        if len(paths) > max_files:
            paths = [paths[i] for i in sorted(rng.choice(len(paths), max_files, replace=False))]
        todo += [(run, p) for p in paths]
    print(f"{len(runs)} training runs, {len(todo)} subrun files", flush=True)
    counts, run_of, edges = defaultdict(list), [], {}
    with ThreadPoolExecutor(workers) as pool:
        for (run, _), h in zip(todo, pool.map(lambda t: _read(t[1]), todo)):
            if set(h) != set(HISTOGRAMS) or h["crvDigisPerChannel"][0].sum() <= 0:
                continue
            for k, (c, e) in h.items():
                counts[k].append(c.astype(np.float32))
                edges[k] = e
            run_of.append(int(run))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, run=np.array(run_of),
                        **{f"counts_{k}": np.stack(v) for k, v in counts.items()},
                        **{f"edges_{k}": v for k, v in edges.items()})
    print(f"✓ {len(run_of)} subruns from {len(set(run_of))} runs saved to {out}")


class Bank:
    def __init__(self, path: Path):
        z = np.load(path)
        self.run = z["run"]
        self.counts = {k: z[f"counts_{k}"] for k in HISTOGRAMS}
        self.edges = {k: z[f"edges_{k}"] for k in HISTOGRAMS}
        self.rows_of = defaultdict(list)
        for i, r in enumerate(self.run):
            self.rows_of[int(r)].append(i)

    def runs(self):
        return sorted(self.rows_of)

    def base(self, rng, hist: str, runs) -> np.ndarray:
        """The sum of a random number (log-uniform, 1..all) of one random run's subruns."""
        rows = self.rows_of[int(rng.choice(runs))]
        k = int(round(np.exp(rng.uniform(0, np.log(len(rows))))))
        pick = rng.choice(rows, max(1, k), replace=False)
        return self.counts[hist][pick].sum(axis=0).astype(float)


# ---------------------------------------------------------------- fault injection

def remap(c, e, inverse):
    """Counts after moving every entry x to x' (given as x = inverse(x')), by the cumulative."""
    cum = np.concatenate([[0.0], np.cumsum(c)])
    new = np.interp(inverse(e), e, cum, left=0.0, right=cum[-1])
    return np.clip(np.diff(new), 0, None)


def _centers(e):
    return 0.5 * (e[1:] + e[:-1])


def _gauss(e, mu, sig):
    from scipy.stats import norm
    return np.diff(norm.cdf(e, mu, sig))


def inject(hist: str, kind: str, c: np.ndarray, e: np.ndarray, rng) -> np.ndarray:
    c = c.copy()
    x, tot = _centers(e), c.sum()
    if kind == "normal":
        return c
    if hist == "PEr":
        if kind == "gain_down":
            g = rng.uniform(0.65, 0.85)
            return remap(c, e, lambda y: y / g)
        if kind == "gain_up":
            g = rng.uniform(1.18, 1.45)
            return remap(c, e, lambda y: y / g)
        if kind == "shoulder_loss":
            c[x > rng.uniform(16, 22)] *= rng.uniform(0.15, 0.5)
            return c
        if kind == "low_pe_excess":
            c[x < rng.uniform(12, 18)] *= rng.uniform(2, 5)
            return c
    if hist == "logchi2":
        if kind == "shift":
            d = rng.choice([-1, 1]) * rng.uniform(0.25, 0.6)
            return remap(c, e, lambda y: y - d)
        if kind == "high_bump":
            return c + tot * rng.uniform(0.02, 0.1) * _gauss(e, rng.uniform(3.6, 4.5), rng.uniform(0.15, 0.4))
        if kind == "broaden":
            m, s = x[np.argmax(c)], rng.uniform(1.3, 1.8)
            return remap(c, e, lambda y: m + (y - m) / s)
    if hist == "h1_peakAdc":
        if kind == "shift":
            d = rng.choice([-1, 1]) * rng.uniform(30, 150)
            return remap(c, e, lambda y: y - d)
        if kind == "saturation":
            c[-1] = tot * rng.uniform(0.03, 0.15)
            return c
        if kind == "below_baseline":
            return c + tot * rng.uniform(0.02, 0.1) * _gauss(e, rng.uniform(500, 1800), rng.uniform(50, 400))
    if hist == "crvDigisPerChannel":
        n = len(c)
        blocks = [b for b in range(n // FEB) if c[b * FEB:(b + 1) * FEB].mean() > 0.2 * np.median(c[c > 0])]
        live = np.flatnonzero(c > 0)
        if kind == "isolated_dead":
            c[rng.choice(live, rng.integers(1, 4), replace=False)] = 0
        elif kind == "dead_febs":
            k = int(rng.integers(1, 9))
            b0 = blocks[rng.integers(0, len(blocks))]
            c[b0 * FEB:min(n, (b0 + k) * FEB)] = 0
        elif kind == "parity_dead":
            start = 128 * int(rng.integers(0, n // 128))
            c[start + int(rng.integers(0, 2)):start + 128 * int(rng.integers(1, 3)):2] = 0
        elif kind == "low_block":
            b0, k = blocks[rng.integers(0, len(blocks))], int(rng.integers(1, 5))
            c[b0 * FEB:(b0 + k) * FEB] *= rng.uniform(0.1, 0.4)
        elif kind == "hot_block":
            b0 = blocks[rng.integers(0, len(blocks))]
            c[b0 * FEB:(b0 + 1) * FEB] *= rng.uniform(3, 10)
        elif kind == "hot_channel":
            c[rng.choice(live)] = np.median(c[live]) * rng.uniform(10, 50)
        return c
    raise ValueError(f"{hist}: unknown kind {kind}")


def make_set(bank: Bank, hist: str, runs, n_per_unit: int, seed: int):
    rng = np.random.default_rng(seed)
    e = bank.edges[hist]
    X, y, names = [], [], []
    for kind, label, freq in KINDS[hist]:
        for _ in range(int(round(n_per_unit * freq))):
            X.append(render(inject(hist, kind, bank.base(rng, hist, runs), e, rng), e))
            y.append(label)
            names.append(kind)
    return np.stack(X)[:, None], np.array(y, dtype=np.float32), np.array(names)


# ---------------------------------------------------------------- train, score, compare

def run(args):
    bank = Bank(Path(args.bank))
    data, out = Path(args.data), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    meta = json.loads((data / "runs.json").read_text())
    scored = sorted(meta["runs"])
    rng = np.random.default_rng(args.seed)
    runs = [r for r in bank.runs() if r not in set(args.drop_runs) and (not args.max_run or r <= args.max_run)]
    val_runs = set(rng.choice(runs, max(1, len(runs) // 5), replace=False).tolist())
    tr_runs = [r for r in runs if r not in val_runs]
    print(f"bank: {len(bank.run)} subruns, {len(runs)} runs ({len(tr_runs)} train, {len(val_runs)} held out)")

    other = {}
    if args.compare:
        other = {(r["run"], r["histogram"]): r for r in csv.DictReader(open(args.compare))}

    rows, heldout = [], []
    for h_i, hist in enumerate(HISTOGRAMS):
        t0 = time.time()
        X, y, _ = make_set(bank, hist, tr_runs, args.n_train, seed=100 * args.seed + h_i)
        model, h = M.train(X, y, epochs=args.epochs, seed=args.seed)
        torch.save(model.state_dict(), out / "models" / f"{hist}.pt")
        Xh, yh, kh = make_set(bank, hist, sorted(val_runs), args.n_heldout, seed=9000 + 100 * args.seed + h_i)
        ph = M.predict(model, Xh)
        for kind in dict.fromkeys(kh):
            sel = kh == kind
            heldout.append({"histogram": hist, "kind": kind, "truth": "BAD" if yh[sel][0] else "GOOD",
                            "n": int(sel.sum()),
                            "accuracy": float(((ph[sel] > args.threshold) == (yh[sel] > 0.5)).mean()),
                            "mean_p_bad": float(ph[sel].mean())})
        for r in scored:
            with uproot.open(data / f"dqm.mu2e.cosmics_crv.run{r}.root") as f:
                c, e = f[HISTOGRAMS[hist]].to_numpy()
            img = render(c, e, png=str(out / "images" / f"{r}_{hist}.png"))
            p = float(M.predict(model, img[None, None])[0])
            o = other.get((r, hist), {})
            rows.append({"run": r, "histogram": hist, "p_bad": round(p, 4),
                         "hydra": "BAD" if p > args.threshold else "GOOD",
                         "ks": o.get("standard", ""), "vision": o.get("ml", "")})
        print(f"  {hist}: {len(X)} images, val acc {h.val_acc[-1]:.3f}, "
              f"held-out runs acc {np.mean((ph > args.threshold) == (yh > 0.5)):.3f}, {time.time() - t0:.0f}s",
              flush=True)

    rows.sort(key=lambda r: (r["run"], list(HISTOGRAMS).index(r["histogram"])))
    for name, rr in (("results.csv", rows), ("heldout.csv", heldout)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    write_summary(out, rows, heldout, meta, args)
    print(f"✓ see {out / 'summary.md'}")


def agreement(rows, a, b):
    pairs = [(r[a], r[b]) for r in rows if r[a] and r[b]]
    both_bad = sum(x == y == "BAD" for x, y in pairs)
    both_good = sum(x == y == "GOOD" for x, y in pairs)
    return {"n": len(pairs), "agree": both_bad + both_good, "both BAD": both_bad, "both GOOD": both_good,
            f"{a} only BAD": sum(x == "BAD" != y for x, y in pairs),
            f"{b} only BAD": sum(y == "BAD" != x for x, y in pairs)}


def write_summary(out, rows, heldout, meta, args):
    ref = meta["reference"]
    L = ["# mu2e-hydra on real CRV runs", "",
         f"Data: `{args.data}` (reference for KS and the vision LLM: {ref}; the CNN uses no reference). "
         f"Compared with: `{args.compare or '-'}`. Threshold P(BAD) > {args.threshold}; "
         f"{args.n_train} training images per unit frequency, {args.epochs} epochs. "
         f"Training runs dropped: {args.drop_runs or 'none'}; latest training run: {args.max_run or 'any'}.", "",
         "## Agreement per histogram", "",
         "| Histogram | CNN BAD | Vision BAD | KS BAD | CNN = vision | CNN only BAD | Vision only BAD |",
         "|---|---|---|---|---|---|---|"]
    for hist in [*HISTOGRAMS, None]:
        rr = [r for r in rows if hist is None or r["histogram"] == hist]
        a = agreement(rr, "hydra", "vision")
        nb = lambda k: sum(r[k] == "BAD" for r in rr)
        L.append(f"| {hist or '**all**'} | {nb('hydra')}/{len(rr)} | {nb('vision')} | {nb('ks')} | "
                 f"{a['agree']}/{a['n']} | {a['hydra only BAD']} | {a['vision only BAD']} |")
    L += ["", "## Every run", "",
          "| Run | " + " | ".join(HISTOGRAMS) + " |", "|---|" + "---|" * len(HISTOGRAMS)]
    by = {(r["run"], r["histogram"]): r for r in rows}
    for run in sorted({r["run"] for r in rows}):
        cells = []
        for hist in HISTOGRAMS:
            r = by[(run, hist)]
            v = r["vision"][:1] if r["vision"] else "-"
            cell = f"{r['p_bad']:.2f} {r['hydra'][:1]} / {v}"
            cells.append(f"**{cell}**" if r["vision"] and r["vision"] != r["hydra"] else cell)
        L.append(f"| {run} | " + " | ".join(cells) + " |")
    L += ["", "Cell: CNN P(BAD), CNN verdict / vision verdict (G or B). Bold: they disagree.", "",
          "## Held-out runs (training runs kept out of training, with injected faults)", "",
          "| Histogram | Kind | Truth | N | Accuracy | Mean P(BAD) |", "|---|---|---|---|---|---|"]
    for h in heldout:
        L.append(f"| {h['histogram']} | {h['kind']} | {h['truth']} | {h['n']} | {h['accuracy']:.3f} | "
                 f"{h['mean_p_bad']:.3f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--survey", default=str(Path(__file__).parents[2] / "mu2e-vision/real/survey.csv"))
    c.add_argument("--exclude", required=True, help="a mu2e-vision data directory; its runs are kept out")
    c.add_argument("--dataset", default="crvreco-006-001-file")
    c.add_argument("--max-files", type=int, default=30, help="subrun files read per training run")
    c.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bank", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--compare", help="a mu2e-vision results.csv (columns standard, ml)")
    r.add_argument("--n-train", type=int, default=300)
    r.add_argument("--n-heldout", type=int, default=40)
    r.add_argument("--epochs", type=int, default=15)
    r.add_argument("--threshold", type=float, default=0.5)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--drop-runs", type=int, nargs="*", default=[],
                   help="training runs to leave out (as if a shifter had labelled them BAD)")
    r.add_argument("--max-run", type=int, help="train only on runs up to this one")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect(Path(args.survey), Path(args.exclude), Path(args.out), args.max_files, args.dataset)
    else:
        torch.set_num_threads(max(1, torch.get_num_threads()))
        run(args)


if __name__ == "__main__":
    main()
