#!/usr/bin/env python3
"""Rebuild the paper's figures and tables.

    python experiments/v2/reproduce.py

reads experiments/v2/results/summary.json and query_level.json. It does not
download data and does not refit clusters. query_level.json holds the
Holm-Bonferroni decisions and the 5-fold cross-validated budgets.

    python experiments/v2/reproduce.py --full

reruns the released MiniLM and BGE-small grid, then rebuilds query_level.json
from the per-query arrays. e5-small-v2, extra seeds, and ScaNN are not part
of that grid.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid"]
RELEASED_EMBEDDERS = ["minilm", "bge-small"]


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def rebuild_artifacts():
    run([sys.executable, str(ROOT / "export_latex.py")])
    run([sys.executable, str(ROOT / "make_figures.py")])


def full_grid():
    run([sys.executable, str(ROOT / "prepare_data.py"), *DATASETS])
    for emb in RELEASED_EMBEDDERS:
        run([sys.executable, str(ROOT / "embed.py"), emb, *DATASETS])
    for emb in RELEASED_EMBEDDERS:
        for ds in DATASETS:
            run([sys.executable, str(ROOT / "run_routing.py"), ds, emb])
            run([sys.executable, str(ROOT / "run_cheap.py"), ds, emb])
            run([sys.executable, str(ROOT / "run_baselines.py"), ds, emb])
    run([sys.executable, str(ROOT / "aggregate.py")])
    run([sys.executable, str(ROOT / "query_level.py"), str(ROOT / "results")])
    rebuild_artifacts()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Rerun the released MiniLM and BGE-small grid from scratch (long).",
    )
    args = parser.parse_args()
    if args.full:
        full_grid()
    else:
        rebuild_artifacts()


if __name__ == "__main__":
    main()
