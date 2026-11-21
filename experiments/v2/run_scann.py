#!/usr/bin/env python3
"""ScaNN and same-vector FAISS HNSW on one encoded corpus.

Usage: python run_scann.py <dataset> <embedder>

Writes results_scann/scann__<dataset>__<embedder>.json. This is not part of
the released 12-pair grid. Search latency is one thread, batch size 1,
median of three passes. Neighbor recall is against exact inner-product top-10
on these vectors.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import scann

from common import BASE, gains_matrix, load, ndcg_at10

faiss.omp_set_num_threads(1)
OUT = Path(BASE) / "results_scann"
OUT.mkdir(parents=True, exist_ok=True)


def timed(fn, nq, reps=3):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for i in range(nq):
            fn(i)
        samples.append((time.perf_counter() - t0) / nq * 1e3)
    return float(np.median(samples))


def recall_at_10(I, truth):
    hit = 0.0
    for i in range(len(I)):
        hit += len(set(I[i].tolist()) & set(truth[i].tolist())) / 10.0
    return hit / len(I)


def main():
    ds, emb = sys.argv[1], sys.argv[2]
    d, X, Q = load(ds, emb)
    X = np.ascontiguousarray(X, dtype=np.float32)
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    N, m = X.shape
    nq = Q.shape[0]
    gains = gains_matrix(d)
    sims = Q @ X.T
    truth = np.argpartition(-sims, 9, axis=1)[:, :10]
    rows = []

    def add(kind, params, I, ms):
        nd = ndcg_at10(I, gains)
        rows.append({
            "kind": kind,
            **params,
            "recall": float(recall_at_10(I, truth)),
            "ndcg": float(nd.mean()),
            "ms": ms,
            "N": int(N),
            "m": int(m),
            "nq": int(nq),
        })
        print(kind, params, f"R {rows[-1]['recall']:.3f}", f"nDCG {rows[-1]['ndcg']:.3f}", f"{ms:.3f} ms", flush=True)

    # Exact
    flat = faiss.IndexFlatIP(m)
    flat.add(X)
    _, I = flat.search(Q, 10)
    add("faiss_flat", {}, I, timed(lambda i: flat.search(Q[i:i + 1], 10), nq))

    # HNSW on the same vectors
    hnsw = faiss.IndexHNSWFlat(m, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(X)
    for ef in (16, 32, 64):
        hnsw.hnsw.efSearch = ef
        _, I = hnsw.search(Q, 10)
        add("faiss_hnsw", {"M": 32, "efSearch": ef}, I, timed(lambda i, ef=ef: hnsw.search(Q[i:i + 1], 10), nq))

    # ScaNN: a tree with brute-force leaves (IVF-style) and the anisotropic hash from Guo et al.
    train_n = int(min(N, 100000))
    for leaves in (32, 128, 256):
        if leaves >= N:
            continue
        for probe in (4, 8, 16):
            if probe > leaves:
                continue
            brute = (
                scann.scann_ops_pybind.builder(X, 10, "dot_product")
                .tree(num_leaves=leaves, num_leaves_to_search=probe, training_sample_size=train_n)
                .score_brute_force(True)
                .set_n_training_threads(1)
                .build()
            )
            I, _ = brute.search_batched(Q, leaves_to_search=probe)
            add(
                "scann_tree_brute",
                {"num_leaves": leaves, "num_leaves_to_search": probe},
                I,
                timed(lambda i, s=brute, p=probe: s.search_batched(Q[i:i + 1], leaves_to_search=p), nq),
            )
            hashed = (
                scann.scann_ops_pybind.builder(X, 10, "dot_product")
                .tree(num_leaves=leaves, num_leaves_to_search=probe, training_sample_size=train_n)
                .score_ah(2, anisotropic_quantization_threshold=0.2)
                .reorder(100)
                .set_n_training_threads(1)
                .build()
            )
            I, _ = hashed.search_batched(Q, leaves_to_search=probe)
            add(
                "scann_ah",
                {"num_leaves": leaves, "num_leaves_to_search": probe, "dims_per_block": 2, "reorder": 100, "threshold": 0.2},
                I,
                timed(lambda i, s=hashed, p=probe: s.search_batched(Q[i:i + 1], leaves_to_search=p), nq),
            )

    out = {
        "ds": ds,
        "emb": emb,
        "N": int(N),
        "m": int(m),
        "nq": int(nq),
        "threads": 1,
        "note": "Same-vector comparison on this machine. Not part of the released FAISS table.",
        "rows": rows,
    }
    path = OUT / f"scann__{ds}__{emb}.json"
    json.dump(out, open(path, "w"))
    print("wrote", path)


if __name__ == "__main__":
    main()
