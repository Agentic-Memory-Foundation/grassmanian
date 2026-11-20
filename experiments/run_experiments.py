#!/usr/bin/env python3
"""Run Grassmannian-vs-IVF indexing experiments and subspace-structure analysis."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data import load_or_build_embeddings, load_scifact, synthetic_subspace_corpus
from grassmann import (
    effective_rank,
    fit_subspace,
    mean_pairwise_geodesic,
    participation_ratio,
    principal_angles,
)
from index import TwoLevelIndex, brute_topk, evaluate_index, relevance_recall


OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def analyze_structure(X: np.ndarray, labels: np.ndarray, k: int = 8) -> dict:
    """Measure intra-cluster rank / angles vs random Gaussian baseline."""
    n_clusters = int(labels.max()) + 1
    ranks90 = []
    parts = []
    bases = []
    intra_angles = []
    sizes = []
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) < max(k + 1, 5):
            continue
        sizes.append(int(len(idx)))
        ranks90.append(effective_rank(X[idx], energy=0.9))
        parts.append(participation_ratio(X[idx]))
        U = fit_subspace(X[idx], min(k, len(idx)))
        bases.append(U)
        # angle between two random halves of the cluster
        rng = np.random.default_rng(c)
        perm = rng.permutation(len(idx))
        mid = len(idx) // 2
        if mid >= 3:
            U1 = fit_subspace(X[idx[perm[:mid]]], min(k, mid))
            U2 = fit_subspace(X[idx[perm[mid:]]], min(k, len(idx) - mid))
            th = principal_angles(U1, U2)
            intra_angles.append(float(np.mean(th)))

    inter = mean_pairwise_geodesic(bases) if len(bases) >= 2 else 0.0

    # random baseline matched to same sizes
    rng = np.random.default_rng(42)
    rand_ranks = []
    rand_parts = []
    rand_bases = []
    for n in sizes:
        R = rng.normal(size=(n, X.shape[1])).astype(np.float32)
        R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
        rand_ranks.append(effective_rank(R, energy=0.9))
        rand_parts.append(participation_ratio(R))
        rand_bases.append(fit_subspace(R, min(k, n)))
    rand_inter = mean_pairwise_geodesic(rand_bases) if len(rand_bases) >= 2 else 0.0
    rand_intra = []
    for n in sizes:
        if n < 8:
            continue
        R = rng.normal(size=(n, X.shape[1])).astype(np.float32)
        R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
        mid = n // 2
        U1 = fit_subspace(R[:mid], min(k, mid))
        U2 = fit_subspace(R[mid:], min(k, n - mid))
        rand_intra.append(float(np.mean(principal_angles(U1, U2))))

    return {
        "n_clusters_used": len(sizes),
        "mean_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
        "mean_rank90": float(np.mean(ranks90)) if ranks90 else 0.0,
        "mean_participation": float(np.mean(parts)) if parts else 0.0,
        "mean_intra_angle": float(np.mean(intra_angles)) if intra_angles else 0.0,
        "mean_inter_geodesic": float(inter),
        "random_mean_rank90": float(np.mean(rand_ranks)) if rand_ranks else 0.0,
        "random_mean_participation": float(np.mean(rand_parts)) if rand_parts else 0.0,
        "random_mean_intra_angle": float(np.mean(rand_intra)) if rand_intra else 0.0,
        "random_mean_inter_geodesic": float(rand_inter),
        "rank_ratio_vs_random": (
            float(np.mean(ranks90) / (np.mean(rand_ranks) + 1e-12)) if ranks90 and rand_ranks else 1.0
        ),
    }


def _eval_config(X, Q, truth, routing, n_clusters, k, n_probe, topk, meta, labels=None, tag="", index=None):
    if index is None:
        index = TwoLevelIndex(
            X, n_clusters=n_clusters, subspace_k=k, routing=routing, labels=labels
        )
    stats = evaluate_index(index, Q, truth, n_probe=n_probe, topk=topk)
    row = {
        "dataset": tag or routing,
        "routing": routing,
        "partition": "oracle" if labels is not None else "kmeans",
        "n_clusters": stats.n_clusters,
        "subspace_k": k,
        "n_probe": n_probe,
        "neighbor_recall@10": stats.recall_at_k,
        "probed_frac": stats.mean_probed_frac,
        "query_ms": stats.mean_query_ms,
        "build_s": stats.build_s,
    }
    # Relevance recall only for the paper's primary probe settings (keeps the grid fast).
    if meta is not None and "qrels" in meta and n_probe in (2, 4) and k in (4, 8):
        preds = []
        for i in range(Q.shape[0]):
            idx, _ = index.search(Q[i], topk=topk, n_probe=n_probe)
            row_idx = np.zeros(topk, dtype=np.int64)
            row_idx[: min(len(idx), topk)] = idx[:topk]
            preds.append(row_idx)
        row["relevance_recall@10"] = relevance_recall(
            np.vstack(preds), meta["query_ids"], meta["doc_ids"], meta["qrels"], k=topk
        )
    print(
        f"  {row['partition']:7s}/{routing:9s} C={stats.n_clusters:3d} k={k:2d} P={n_probe:2d} "
        f"R@10={stats.recall_at_k:.3f} probed={stats.mean_probed_frac:.3f} "
        f"ms={stats.mean_query_ms:.2f}"
        + (f" rel={row['relevance_recall@10']:.3f}" if "relevance_recall@10" in row else ""),
        flush=True,
    )
    return row


def run_retrieval_suite(
    name: str,
    X: np.ndarray,
    Q: np.ndarray,
    meta: dict | None = None,
    oracle_labels: np.ndarray | None = None,
) -> dict:
    print(f"\n=== Retrieval suite: {name} (n={X.shape[0]}, d={X.shape[1]}, q={Q.shape[0]}) ===")
    topk = 10
    truth = brute_topk(X, Q, k=topk)

    # Keep the grid tight enough for interactive runs; primary paper config is C=32,k=8,P=4.
    cluster_grid = [32] if X.shape[0] >= 500 else [8, 16]
    if name == "synthetic":
        cluster_grid = [16, 20]
    k_grid = [4, 8] if name != "synthetic" else [4, 8]
    probe_grid = [1, 2, 4, 8]

    rows = []
    for n_clusters in cluster_grid:
        if n_clusters * 5 > X.shape[0]:
            continue
        for routing in ("centroid", "grassmann", "affine_grassmann"):
            for k in k_grid:
                if k > X.shape[1]:
                    continue
                index = TwoLevelIndex(X, n_clusters=n_clusters, subspace_k=k, routing=routing)
                for n_probe in probe_grid:
                    if n_probe > index.n_clusters:
                        continue
                    rows.append(
                        _eval_config(
                            X,
                            Q,
                            truth,
                            routing,
                            n_clusters,
                            k,
                            n_probe,
                            topk,
                            meta,
                            tag=name,
                            index=index,
                        )
                    )

    # Oracle partition (planted / true labels): upper bound on routing quality
    if oracle_labels is not None:
        n_c = int(oracle_labels.max()) + 1
        for routing in ("centroid", "grassmann", "affine_grassmann"):
            for k in (4, 8):
                index = TwoLevelIndex(
                    X, n_clusters=n_c, subspace_k=k, routing=routing, labels=oracle_labels
                )
                for n_probe in (1, 2, 4):
                    rows.append(
                        _eval_config(
                            X,
                            Q,
                            truth,
                            routing,
                            n_c,
                            k,
                            n_probe,
                            topk,
                            meta,
                            labels=oracle_labels,
                            tag=name,
                            index=index,
                        )
                    )

    # structure analysis on a mid-size clustering
    n_c = 32 if X.shape[0] >= 400 else max(4, X.shape[0] // 40)
    index_s = TwoLevelIndex(X, n_clusters=n_c, subspace_k=8, routing="grassmann")
    structure = analyze_structure(X, index_s.labels, k=8)
    print("  structure:", json.dumps({kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in structure.items()}))

    # flat brute latency reference
    t0 = time.perf_counter()
    _ = brute_topk(X, Q[: min(50, Q.shape[0])], k=topk)
    brute_ms = (time.perf_counter() - t0) * 1000.0 / max(min(50, Q.shape[0]), 1)

    return {
        "dataset": name,
        "n_docs": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "n_queries": int(Q.shape[0]),
        "brute_query_ms": float(brute_ms),
        "structure": structure,
        "rows": rows,
    }


def make_figures(results: dict) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    # Fig 1: structure comparison SciFact vs synthetic vs random
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    datasets = []
    for key in ("scifact", "synthetic"):
        if key in results:
            datasets.append(key)
    metrics = ["mean_rank90", "mean_participation", "mean_intra_angle"]
    labels = ["Rank@90% energy", "Participation ratio", "Intra-cluster angle"]
    x = np.arange(len(labels))
    width = 0.35
    for ax_i, ds in enumerate(datasets):
        st = results[ds]["structure"]
        real = [st["mean_rank90"], st["mean_participation"], st["mean_intra_angle"]]
        rand = [
            st["random_mean_rank90"],
            st["random_mean_participation"],
            st["random_mean_intra_angle"],
        ]
        # normalize each metric for display side by side using twin? Better separate bars per panel.
    # redraw as grouped bars for rank and participation only on left; angles on right
    ax = axes[0]
    names = []
    rank_r, rank_g = [], []
    part_r, part_g = [], []
    for ds in datasets:
        st = results[ds]["structure"]
        names.append(ds)
        rank_r.append(st["mean_rank90"])
        rank_g.append(st["random_mean_rank90"])
        part_r.append(st["mean_participation"])
        part_g.append(st["random_mean_participation"])
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.2, rank_r, 0.2, label="Data rank@90%", color="#4a90c8")
    ax.bar(xpos, rank_g, 0.2, label="Random rank@90%", color="#c8c8c8")
    ax.bar(xpos + 0.2, part_r, 0.2, label="Data participation", color="#70a86e")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names)
    ax.set_ylabel("Effective dimension")
    ax.set_title("Cluster rank structure vs random")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    intra_r, intra_g, inter = [], [], []
    for ds in datasets:
        st = results[ds]["structure"]
        intra_r.append(st["mean_intra_angle"])
        intra_g.append(st["random_mean_intra_angle"])
        inter.append(st["mean_inter_geodesic"])
    ax.bar(xpos - 0.2, intra_r, 0.2, label="Intra angle (data)", color="#e08a3c")
    ax.bar(xpos, intra_g, 0.2, label="Intra angle (random)", color="#c8c8c8")
    ax.bar(xpos + 0.2, inter, 0.2, label="Inter geodesic", color="#9b59b6")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names)
    ax.set_ylabel("Radians")
    ax.set_title("Subspace angles")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "fig_structure.png")
    fig.savefig(FIG / "fig_structure.pdf")
    plt.close(fig)

    # Fig 2: recall vs probed fraction for SciFact
    if "scifact" in results:
        rows = results["scifact"]["rows"]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for routing, color, marker in (("centroid", "#4a90c8", "o"), ("grassmann", "#c4564e", "s"), ("affine_grassmann", "#70a86e", "^")):
            pts = [r for r in rows if r["routing"] == routing and r["n_clusters"] == 32 and r["subspace_k"] == 8]
            if not pts:
                pts = [r for r in rows if r["routing"] == routing]
            pts = sorted(pts, key=lambda r: r["probed_frac"])
            ax.plot(
                [r["probed_frac"] for r in pts],
                [r["neighbor_recall@10"] for r in pts],
                marker=marker,
                color=color,
                label=f"{routing}",
                linewidth=1.8,
            )
        ax.set_xlabel("Fraction of corpus probed")
        ax.set_ylabel("Neighbor recall@10 (vs exact cosine)")
        ax.set_title("SciFact: routing quality vs scan cost")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "fig_recall_vs_probe.png")
        fig.savefig(FIG / "fig_recall_vs_probe.pdf")
        plt.close(fig)

        # Fig 3: ablation over k
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for routing, color in (("centroid", "#4a90c8"), ("grassmann", "#c4564e"), ("affine_grassmann", "#70a86e")):
            xs, ys = [], []
            for k in sorted({r["subspace_k"] for r in rows}):
                cand = [
                    r
                    for r in rows
                    if r["routing"] == routing and r["subspace_k"] == k and r["n_probe"] == 4 and r["n_clusters"] == 32
                ]
                if not cand:
                    continue
                xs.append(k)
                ys.append(cand[0]["neighbor_recall@10"])
            if xs:
                ax.plot(xs, ys, marker="o", color=color, label=routing, linewidth=1.8)
        ax.set_xlabel("Subspace dimension $k$")
        ax.set_ylabel("Neighbor recall@10")
        ax.set_title("SciFact: effect of subspace dimension (C=32, P=4)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "fig_k_ablation.png")
        fig.savefig(FIG / "fig_k_ablation.pdf")
        plt.close(fig)

        # Fig 4: latency
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        # pick matched configs
        configs = []
        for routing in ("centroid", "grassmann", "affine_grassmann"):
            for n_probe in (1, 2, 4, 8):
                cand = [
                    r
                    for r in rows
                    if r["routing"] == routing and r["n_probe"] == n_probe and r["n_clusters"] == 32 and r["subspace_k"] == 8
                ]
                if cand:
                    configs.append(cand[0])
        labels = [f"{r['routing'][:1].upper()}P{r['n_probe']}" for r in configs]
        colors_map = {"centroid": "#4a90c8", "grassmann": "#c4564e", "affine_grassmann": "#70a86e"}
        ax.bar(range(len(configs)), [r["query_ms"] for r in configs], color=[colors_map[r["routing"]] for r in configs])
        ax.axhline(results["scifact"]["brute_query_ms"], color="gray", linestyle="--", label=f"brute ({results['scifact']['brute_query_ms']:.2f} ms)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Mean query latency (ms)")
        ax.set_title("SciFact: latency (C=32, k=8)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "fig_latency.png")
        fig.savefig(FIG / "fig_latency.pdf")
        plt.close(fig)

    # Fig 5: synthetic positive control
    if "synthetic" in results:
        rows = results["synthetic"]["rows"]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for routing, color, marker in (("centroid", "#4a90c8", "o"), ("grassmann", "#c4564e", "s"), ("affine_grassmann", "#70a86e", "^")):
            pts = [r for r in rows if r["routing"] == routing and r["subspace_k"] == 4]
            pts = sorted(pts, key=lambda r: r["probed_frac"])
            ax.plot(
                [r["probed_frac"] for r in pts],
                [r["neighbor_recall@10"] for r in pts],
                marker=marker,
                color=color,
                label=routing,
                linewidth=1.8,
            )
        ax.set_xlabel("Fraction of corpus probed")
        ax.set_ylabel("Neighbor recall@10")
        ax.set_title("Synthetic planted subspaces (positive control)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "fig_synthetic.png")
        fig.savefig(FIG / "fig_synthetic.pdf")
        plt.close(fig)

    print(f"Figures written to {FIG}")


def summarize_for_tex(results: dict) -> str:
    lines = ["% auto-generated by experiments/run_experiments.py", "\\newcommand{\\ExpGenerated}{1}"]
    if "scifact" in results:
        st = results["scifact"]["structure"]
        lines.append(f"\\newcommand{{\\SciRank}}{{{st['mean_rank90']:.1f}}}")
        lines.append(f"\\newcommand{{\\SciRandRank}}{{{st['random_mean_rank90']:.1f}}}")
        lines.append(f"\\newcommand{{\\SciPart}}{{{st['mean_participation']:.1f}}}")
        lines.append(f"\\newcommand{{\\SciIntra}}{{{st['mean_intra_angle']:.2f}}}")
        lines.append(f"\\newcommand{{\\SciInter}}{{{st['mean_inter_geodesic']:.2f}}}")
        lines.append(f"\\newcommand{{\\SciNDocs}}{{{results['scifact']['n_docs']}}}")
        lines.append(f"\\newcommand{{\\SciNQueries}}{{{results['scifact']['n_queries']}}}")
        rows = results["scifact"]["rows"]

        def pick(routing, C=32, k=8, P=4, partition="kmeans"):
            for r in rows:
                if (
                    r["routing"] == routing
                    and r.get("partition", "kmeans") == partition
                    and r["n_clusters"] == C
                    and r["subspace_k"] == k
                    and r["n_probe"] == P
                ):
                    return r
            return None

        g = pick("grassmann")
        c = pick("centroid")
        a = pick("affine_grassmann")
        if g and c and a:
            lines.append(f"\\newcommand{{\\GrassRecall}}{{{g['neighbor_recall@10']*100:.1f}}}")
            lines.append(f"\\newcommand{{\\CentRecall}}{{{c['neighbor_recall@10']*100:.1f}}}")
            lines.append(f"\\newcommand{{\\AffineRecall}}{{{a['neighbor_recall@10']*100:.1f}}}")
            lines.append(f"\\newcommand{{\\GrassProbe}}{{{g['probed_frac']*100:.1f}}}")
            lines.append(f"\\newcommand{{\\CentProbe}}{{{c['probed_frac']*100:.1f}}}")
            lines.append(f"\\newcommand{{\\AffineProbe}}{{{a['probed_frac']*100:.1f}}}")
            lines.append(f"\\newcommand{{\\GrassMs}}{{{g['query_ms']:.2f}}}")
            lines.append(f"\\newcommand{{\\CentMs}}{{{c['query_ms']:.2f}}}")
            lines.append(f"\\newcommand{{\\AffineMs}}{{{a['query_ms']:.2f}}}")
            if "relevance_recall@10" in g:
                lines.append(f"\\newcommand{{\\GrassRel}}{{{g['relevance_recall@10']*100:.1f}}}")
                lines.append(f"\\newcommand{{\\CentRel}}{{{c['relevance_recall@10']*100:.1f}}}")
                lines.append(f"\\newcommand{{\\AffineRel}}{{{a['relevance_recall@10']*100:.1f}}}")
    if "synthetic" in results:
        st = results["synthetic"]["structure"]
        lines.append(f"\\newcommand{{\\SynRank}}{{{st['mean_rank90']:.1f}}}")
        lines.append(f"\\newcommand{{\\SynRandRank}}{{{st['random_mean_rank90']:.1f}}}")
        rows = results["synthetic"]["rows"]
        g = next((r for r in rows if r["routing"] == "grassmann" and r.get("partition", "kmeans") == "kmeans" and r["n_probe"] == 2 and r["subspace_k"] == 4), None)
        c = next((r for r in rows if r["routing"] == "centroid" and r.get("partition", "kmeans") == "kmeans" and r["n_probe"] == 2 and r["subspace_k"] == 4), None)
        a = next((r for r in rows if r["routing"] == "affine_grassmann" and r.get("partition", "kmeans") == "kmeans" and r["n_probe"] == 2 and r["subspace_k"] == 4), None)
        if g and c and a:
            lines.append(f"\\newcommand{{\\SynGrassRecall}}{{{g['neighbor_recall@10']*100:.1f}}}")
            lines.append(f"\\newcommand{{\\SynCentRecall}}{{{c['neighbor_recall@10']*100:.1f}}}")
            lines.append(f"\\newcommand{{\\SynAffineRecall}}{{{a['neighbor_recall@10']*100:.1f}}}")
        og = next((r for r in rows if r["routing"] == "grassmann" and r.get("partition") == "oracle" and r["n_probe"] == 2 and r["subspace_k"] == 4), None)
        oc = next((r for r in rows if r["routing"] == "centroid" and r.get("partition") == "oracle" and r["n_probe"] == 2 and r["subspace_k"] == 4), None)
        oa = next((r for r in rows if r["routing"] == "affine_grassmann" and r.get("partition") == "oracle" and r["n_probe"] == 2 and r["subspace_k"] == 4), None)
        if og:
            lines.append(f"\\newcommand{{\\SynOracleGrass}}{{{og['neighbor_recall@10']*100:.1f}}}")
        if oc:
            lines.append(f"\\newcommand{{\\SynOracleCent}}{{{oc['neighbor_recall@10']*100:.1f}}}")
        if oa:
            lines.append(f"\\newcommand{{\\SynOracleAffine}}{{{oa['neighbor_recall@10']*100:.1f}}}")
    return "\n".join(lines) + "\n"


def main():
    results = {}

    # Positive control with planted subspaces + oracle labels
    syn = synthetic_subspace_corpus()
    results["synthetic"] = run_retrieval_suite(
        "synthetic",
        syn["doc_emb"],
        syn["query_emb"],
        meta=syn,
        oracle_labels=syn["labels"],
    )

    # Real RAG embeddings
    try:
        bundle = load_scifact(max_docs=6000, max_queries=400)
        print(f"SciFact: {len(bundle['docs'])} docs, {len(bundle['queries'])} queries")
        X, Q = load_or_build_embeddings(bundle)
        results["scifact"] = run_retrieval_suite("scifact", X, Q, meta=bundle)
    except Exception as e:
        print("SciFact load failed:", e)
        import traceback

        traceback.print_exc()
        print("Continuing with synthetic-only results.")

    out_json = OUT / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")

    tex = summarize_for_tex(results)
    tex_path = ROOT.parent / "generated"
    tex_path.mkdir(exist_ok=True)
    (tex_path / "stats.tex").write_text(tex)
    print(tex)

    make_figures(results)
    print("Done.")


if __name__ == "__main__":
    main()
