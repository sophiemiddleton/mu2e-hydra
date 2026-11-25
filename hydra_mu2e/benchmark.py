"""
Train the Hydra-style classifiers and score them

For each of the three histograms: generate labelled training images, train a PlotCNN, then
score it on
  1. the benchmark set, the same 9 synthetic runs x 3 histograms that KS and the vision LLM
     are scored on in mu2e-vision (generated there by histcompare.synthetic), and
  2. a held-out set of fresh random faults (another seed), broken down by fault kind.

    python -m hydra_mu2e.benchmark --out results/default
    python -m hydra_mu2e.benchmark --out results/no-led --no-led
    python -m hydra_mu2e.benchmark --out results/default \
        --compare ../results/opus-5-5/results.csv     # add the KS and vision columns
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import uproot

from histcompare import synthetic as syn

from . import model as M
from . import training_data as td
from .render import render

FIELDS = ["run", "run_type", "histogram", "truth", "truth_note", "p_bad", "hydra"]


def verdict(p: float, threshold: float = 0.5) -> str:
    return "BAD" if p > threshold else "GOOD"


def score(rows, key: str) -> dict:
    """Counts of the four outcomes of `key` against the truth, and the agreement."""
    c = defaultdict(int)
    for r in rows:
        v, t = r.get(key), r["truth"]
        if not v or not t:
            continue
        c[{("BAD", "BAD"): "BAD caught", ("BAD", "GOOD"): "BAD missed",
           ("GOOD", "BAD"): "GOOD flagged", ("GOOD", "GOOD"): "GOOD passed"}[(t, v)]] += 1
    n = sum(c.values())
    agree = c["BAD caught"] + c["GOOD passed"]
    return {k: c[k] for k in ("BAD caught", "BAD missed", "GOOD flagged", "GOOD passed")} | {
        "agreement": f"{agree}/{n}" if n else "-"}


def load_benchmark(data: Path):
    """[(run, run_type, histogram, counts, edges, truth, note)] for every non-reference run."""
    if not (data / "runs.json").exists():
        syn.generate(str(data))
    meta = json.loads((data / "runs.json").read_text())
    truth = json.loads((data / "truth.json").read_text())
    refs = {str(r) for r in meta["reference"].values()}
    out = []
    for run, run_type in meta["runs"].items():
        if run in refs:
            continue
        with uproot.open(data / f"dqm.mu2e.synthetic_crv.run{run}.root") as f:
            for name in td.HISTOGRAMS:
                counts, edges = f[f"{syn.DIR}/{name}"].to_numpy()
                t = truth.get(run, {}).get(name, {})
                out.append((run, run_type, name, counts, edges, t.get("verdict"), t.get("note", "")))
    return out


def merge_compare(rows, path: Path) -> None:
    """Add the KS ('standard') and vision ('ml') verdicts from a mu2e-vision results.csv."""
    other = {(r["run"], r["histogram"]): r for r in csv.DictReader(open(path))}
    for r in rows:
        o = other.get((r["run"], r["histogram"]), {})
        r["standard"], r["ml"] = o.get("standard") or None, o.get("ml") or None


def write_summary(out: Path, rows, heldout, histories, args, compare: bool) -> None:
    mark = lambda v, t: f"**{v}**" if v and t and v != t else (v or "-")
    lines = ["# mu2e-hydra benchmark", "",
             f"Training: {args.n_train} images per unit frequency per histogram, "
             f"{args.epochs} epochs, LED runs in training: {'no' if args.no_led else 'yes'}, "
             f"threshold P(BAD) > {args.threshold}.", "",
             "## Benchmark set (the runs KS and the vision LLM are scored on)", "",
             "| Method | BAD caught | BAD missed | GOOD flagged | GOOD passed | Agreement |",
             "|---|---|---|---|---|---|"]
    methods = [("Hydra-style CNN", "hydra")]
    if compare:
        methods += [("KS (standard)", "standard"), ("Vision LLM", "ml")]
    for label, key in methods:
        s = score(rows, key)
        lines.append(f"| {label} | " + " | ".join(str(s[k]) for k in s) + " |")
    head = "| Run | Type | Histogram | Truth | P(BAD) | Hydra |" + (" KS | Vision |" if compare else "")
    lines += ["", head + " Note |", "|" + "---|" * (head.count("|") ) ]
    for r in rows:
        extra = f" {mark(r['standard'], r['truth'])} | {mark(r['ml'], r['truth'])} |" if compare else ""
        lines.append(f"| {r['run']} | {r['run_type']} | [{r['histogram']}](images/{r['run']}_{r['histogram']}.png) "
                     f"| {r['truth']} | {r['p_bad']:.3f} | {mark(r['hydra'], r['truth'])} |{extra} {r['truth_note']} |")
    lines += ["", "Bold: disagrees with the truth.", "",
              "## Held-out set (fresh random faults, another seed)", "",
              "| Histogram | Kind | Truth | N | Accuracy | Mean P(BAD) |", "|---|---|---|---|---|---|"]
    for (name, kind), h in sorted(heldout.items()):
        lines.append(f"| {name} | {kind} | {h['truth']} | {h['n']} | {h['acc']:.3f} | {h['p']:.3f} |")
    lines += ["", "## Training", "", "| Histogram | Final train loss | Validation accuracy |", "|---|---|---|"]
    for name, h in histories.items():
        lines.append(f"| {name} | {h.train_loss[-1]:.4f} | {h.val_acc[-1] if h.val_acc else float('nan'):.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="synthetic",
                    help="benchmark set from histcompare.synthetic (generated if missing)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=200,
                    help="training images per unit of a kind's relative frequency")
    ap.add_argument("--n-heldout", type=int, default=50, help="held-out images per kind")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--no-led", action="store_true", help="leave LED runs out of training")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", help="a mu2e-vision results.csv to show KS and vision beside")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    bench = load_benchmark(Path(args.data))
    include_led = not args.no_led

    rows, heldout, histories = [], {}, {}
    for h_i, name in enumerate(td.HISTOGRAMS):
        t0 = time.time()
        X, y, _ = td.make_set(name, args.n_train, seed=1000 * (args.seed + 1) + h_i, include_led=include_led)
        model, hist = M.train(X, y, epochs=args.epochs, seed=args.seed)
        histories[name] = hist
        torch.save(model.state_dict(), out / "models" / f"{name}.pt")
        print(f"  {name}: {len(X)} images, val acc {hist.val_acc[-1] if hist.val_acc else float('nan'):.3f}, "
              f"{time.time() - t0:.0f}s")

        # held-out: every kind, LED included even if it was left out of training
        Xh, yh, kh = td.make_set(name, args.n_heldout, seed=9000 + 1000 * args.seed + h_i, include_led=True)
        ph = M.predict(model, Xh)
        for kind in dict.fromkeys(kh):
            sel = kh == kind
            heldout[(name, kind)] = {"truth": "BAD" if yh[sel][0] else "GOOD", "n": int(sel.sum()),
                                     "acc": float(((ph[sel] > args.threshold) == (yh[sel] > 0.5)).mean()),
                                     "p": float(ph[sel].mean())}

        for run, run_type, hname, counts, edges, truth, note in bench:
            if hname != name:
                continue
            img = render(counts, edges, png=str(out / "images" / f"{run}_{name}.png"))
            p = float(M.predict(model, img[None, None])[0])
            rows.append({"run": run, "run_type": run_type, "histogram": name, "truth": truth,
                         "truth_note": note, "p_bad": p, "hydra": verdict(p, args.threshold)})

    rows.sort(key=lambda r: (r["run"], td.HISTOGRAMS.index(r["histogram"])))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(out / "heldout.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["histogram", "kind", "truth", "n", "accuracy", "mean_p_bad"])
        for (name, kind), h in sorted(heldout.items()):
            w.writerow([name, kind, h["truth"], h["n"], f"{h['acc']:.4f}", f"{h['p']:.4f}"])
    if args.compare:
        merge_compare(rows, Path(args.compare))
    write_summary(out, rows, heldout, histories, args, bool(args.compare))
    print("hydra   ", score(rows, "hydra"))
    print(f"✓ {len(rows)} benchmark comparisons; see {out / 'summary.md'}")


if __name__ == "__main__":
    main()
