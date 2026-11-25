import csv

from hydra_mu2e import benchmark


def test_score_counts():
    rows = [{"truth": "BAD", "x": "BAD"}, {"truth": "BAD", "x": "GOOD"},
            {"truth": "GOOD", "x": "BAD"}, {"truth": "GOOD", "x": "GOOD"},
            {"truth": "GOOD", "x": None}]
    s = benchmark.score(rows, "x")
    assert s == {"BAD caught": 1, "BAD missed": 1, "GOOD flagged": 1, "GOOD passed": 1,
                 "agreement": "2/4"}


def test_end_to_end_smoke(tmp_path):
    """Tiny training run over the real benchmark set: every output is written, 27 rows."""
    data, out = tmp_path / "synthetic", tmp_path / "out"
    benchmark.main(["--data", str(data), "--out", str(out),
                    "--n-train", "4", "--n-heldout", "2", "--epochs", "1"])
    rows = list(csv.DictReader(open(out / "results.csv")))
    assert len(rows) == 27
    assert {r["hydra"] for r in rows} <= {"GOOD", "BAD"}
    assert all(r["truth"] in ("GOOD", "BAD") for r in rows)
    assert (out / "summary.md").exists() and (out / "heldout.csv").exists()
    assert len(list((out / "models").glob("*.pt"))) == 3
    assert len(list((out / "images").glob("*.png"))) == 27
