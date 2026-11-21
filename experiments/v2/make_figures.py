"""Figures for the paper, drawn only from the released JSON.

Writes PDF files under <repo>/figures/. No new measurements.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RES = ROOT / "results"
FIG = REPO / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DS_ORDER = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid"]
EMB_ORDER = ["minilm", "bge-small"]
DS_NAME = {
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "arguana": "ArguAna",
    "scidocs": "SciDocs",
    "fiqa": "FiQA",
    "trec-covid": "TREC-COVID",
}
EMB_NAME = {"minilm": "MiniLM", "bge-small": "BGE-small"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_summary():
    return json.load(open(RES / "summary.json"))


def load_routing():
    out = {}
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            path = RES / f"routing__{ds}__{emb}.json"
            out[(ds, emb)] = json.load(open(path))
    return out


def seed_mean_configs(configs):
    groups = defaultdict(list)
    for c in configs:
        groups[(c["router"], c["C"], c["P"])].append(c)
    rows = []
    for (router, C, P), cs in groups.items():
        rows.append({
            "router": router,
            "C": C,
            "P": P,
            "recall": float(np.mean([x["recall"] for x in cs])),
            "ndcg": float(np.mean([x["ndcg"] for x in cs])),
            "cand_frac": float(np.mean([x["cand_frac"] for x in cs])),
            "flops_route": float(np.mean([x["flops_route"] for x in cs])),
            "flops_scan": float(np.mean([x["flops_scan"] for x in cs])),
        })
    return rows


def fig_kcurve(summary, routing):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), sharex=True)
    colors = {16: "#0072B2", 32: "#D55E00", 64: "#009E73"}
    ks = [0, 4, 8, 16, 32]
    routers = {0: "l2_k0", 4: "kflat_k4", 8: "kflat_k8", 16: "kflat_k16", 32: "kflat_k32"}
    for ax, metric, ylab in (
        (axes[0], "d_recall", "Recall@10 gap vs cosine (points)"),
        (axes[1], "d_ndcg", "nDCG@10 gap vs cosine (points)"),
    ):
        ax.axhline(0.0, color="#444444", lw=0.8, label="cosine IVF")
        for C, color in colors.items():
            ys = []
            for k in ks:
                rec = summary["kcurve"][f"{routers[k]}|C{C}|P4"]
                ys.append(100 * rec[metric])
            ax.plot(ks, ys, marker="o", ms=3.5, color=color, lw=1.4, label=f"C={C}, mean")
            # per-pair seed means at P=4
            for ds, emb in [(a, b) for a in DS_ORDER for b in EMB_ORDER]:
                pts = []
                cfg = { (r["router"], r["C"], r["P"]): r for r in seed_mean_configs(routing[(ds, emb)]["configs"]) }
                base = cfg[("cos", C, 4)]
                key = "recall" if metric == "d_recall" else "ndcg"
                for k in ks:
                    pts.append(100 * (cfg[(routers[k], C, 4)][key] - base[key]))
                ax.plot(ks, pts, color=color, alpha=0.18, lw=0.6)
        ax.set_xticks(ks)
        ax.set_xlabel("Directions k (0 = Euclidean centroid)")
        ax.set_ylabel(ylab)
        ax.set_xlim(-1.5, 33.5)
    # Holm count for the primary comparison, from the query-level file when it exists.
    n_up = summary["kcurve"]["kflat_k16|C32|P4"]["n_sig_up"]
    ql_path = ROOT / "results" / "query_level.json"
    if ql_path.exists():
        ql = json.loads(ql_path.read_text())
        for comp in ql["comparisons"]:
            if (comp["router"], comp["C"], comp["P"], comp["metric"]) == ("kflat_k16", 32, 4, "rec"):
                n_up = comp["n_up"]
    axes[0].text(
        0.02, 0.98,
        f"C=32, k=16: Holm rejects on {n_up}/12 pairs",
        transform=axes[0].transAxes, va="top", fontsize=7,
    )
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_kcurve.pdf")
    plt.close(fig)


def fig_scan(routing, metric, fname, ylabel):
    fig, axes = plt.subplots(len(DS_ORDER), len(EMB_ORDER), figsize=(7.1, 8.6), sharex=True)
    styles = [
        ("cos", 32, "#222222", "-", "cosine, C=32"),
        ("kflat_k16", 32, "#0072B2", "-", "k-flat k=16, C=32"),
        ("kflat_k32", 32, "#E69F00", "-", "k-flat k=32, C=32"),
        ("grass_k8", 32, "#D55E00", "-", "linear Grassmann, C=32"),
        ("cos", 128, "#222222", "--", "cosine, C=128"),
        ("kflat_k16", 128, "#0072B2", "--", "k-flat k=16, C=128"),
        ("kflat_k32", 128, "#E69F00", "--", "k-flat k=32, C=128"),
    ]
    legend_handles = None
    for i, ds in enumerate(DS_ORDER):
        for j, emb in enumerate(EMB_ORDER):
            ax = axes[i, j]
            cfg = seed_mean_configs(routing[(ds, emb)]["configs"])
            by = {(r["router"], r["C"], r["P"]): r for r in cfg}
            for router, C, color, ls, label in styles:
                xs, ys = [], []
                for P in [1, 2, 4, 8, 16, 32, 64]:
                    if P > C:
                        continue
                    row = by.get((router, C, P))
                    if row is None:
                        continue
                    xs.append(100 * row["cand_frac"])
                    ys.append(100 * row[metric])
                if xs:
                    ax.plot(xs, ys, color=color, ls=ls, marker="o", ms=2.2, lw=1.0, label=label)
            if i == 0 and j == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.set_title(f"{DS_NAME[ds]}, {EMB_NAME[emb]}", fontsize=7.5)
            if i == len(DS_ORDER) - 1:
                ax.set_xlabel("Corpus scanned (%)")
            if j == 0:
                ax.set_ylabel(ylabel)
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / fname, bbox_inches="tight")
    plt.close(fig)


def _family_points(rows, N, m, cost_name, family):
    pts = []
    for r in rows:
        if family == "ivf" and r["router"] != "cos":
            continue
        if family == "kflat" and not str(r["router"]).startswith("kflat"):
            continue
        cost = r["flops_scan"] + (r["flops_route"] if cost_name == "cost_total" else 0.0)
        pts.append((cost / (N * m), r["recall"], r["ndcg"]))
    return pts


def _envelope(pts, budgets, y_index):
    out = []
    for b in budgets:
        ys = [p[y_index] for p in pts if p[0] <= b + 1e-15]
        out.append(max(ys) if ys else np.nan)
    return np.array(out)


def fig_cost(routing):
    ql_path = ROOT / "results" / "query_level.json"
    if ql_path.exists():
        fig_cost_heldout(json.loads(ql_path.read_text()))
        return
    _fig_cost_envelope(routing)


def fig_cost_heldout(ql):
    panels = [
        ("cost_total_recall", "Recall gap (points)", "Routing + scan / brute force"),
        ("cost_scan_recall", "Recall gap (points)", "Scan cost / brute force"),
        ("cost_scan_ndcg", "nDCG gap (points)", "Scan cost / brute force"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (key, ylab, xlab) in zip(axes, panels):
        series = ql["curves"][key]
        budgets = np.array([row["budget"] for row in series])
        values = np.array([row["values"] for row in series], dtype=float).T  # pairs, budgets
        for row in values:
            finite = np.isfinite(row)
            if finite.any():
                ax.plot(budgets[finite], 100 * row[finite], color="#0072B2", alpha=0.12, lw=0.7)
        count = np.sum(np.isfinite(values), axis=0)
        mean = np.full(budgets.shape, np.nan)
        for j in range(len(budgets)):
            col = values[:, j]
            col = col[np.isfinite(col)]
            if len(col):
                mean[j] = col.mean()
        show = count >= 6
        ax.plot(budgets[show], 100 * mean[show], color="#0072B2", lw=1.6, label="k-flat minus cosine")
        ax.axhline(0, color="#444444", lw=0.7)
        ax.set_xscale("log")
        ax.set_xticks([0.02, 0.1, 0.5])
        ax.set_xticklabels(["2%", "10%", "50%"], fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim(0.018, 0.55)
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cost.pdf")
    plt.close(fig)


def _fig_cost_envelope(routing):
    budgets = np.logspace(np.log10(0.02), np.log10(0.5), 28)
    panels = [
        ("cost_total", 1, "Recall@10", "Routing + scan cost / brute force"),
        ("cost_scan", 1, "Recall@10", "Scan cost / brute force"),
        ("cost_scan", 2, "nDCG@10", "Scan cost / brute force"),
    ]
    colors = {"ivf": "#444444", "kflat": "#0072B2"}
    labels = {"ivf": "cosine IVF", "kflat": "k-flat"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (cost_name, y_index, ylab, xlab) in zip(axes, panels):
        store = {fam: [] for fam in colors}
        for (ds, emb), blob in routing.items():
            rows = seed_mean_configs(blob["configs"])
            N, m = blob["N"], blob["m"]
            for fam in colors:
                env = _envelope(_family_points(rows, N, m, cost_name, fam), budgets, y_index)
                store[fam].append(env)
                finite = np.isfinite(env)
                if finite.any():
                    ax.plot(budgets[finite], 100 * env[finite], color=colors[fam], alpha=0.12, lw=0.7)
        for fam, color in colors.items():
            stacked = np.vstack(store[fam])
            mean = np.full(budgets.shape, np.nan)
            count = np.zeros(budgets.shape, dtype=int)
            for j in range(len(budgets)):
                col = stacked[:, j]
                col = col[np.isfinite(col)]
                count[j] = len(col)
                if len(col):
                    mean[j] = col.mean()
            show = count >= 6
            ax.plot(budgets[show], 100 * mean[show], color=color, lw=1.6, label=labels[fam])
        ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim(0.018, 0.55)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_cost.pdf")
    plt.close(fig)


def fig_ann(summary):
    fig, axes = plt.subplots(3, 4, figsize=(7.3, 5.6))
    axes = axes.ravel()
    for ax, (ds, emb) in zip(axes, [(a, b) for a in DS_ORDER for b in EMB_ORDER]):
        rows = summary["baselines"][f"{ds}|{emb}"]
        hnsw = sorted(
            [r for r in rows if r["kind"] == "faiss_hnsw" and r.get("ms") is not None],
            key=lambda r: r["efSearch"],
        )
        ivf = [r for r in rows if r["kind"] == "faiss_ivfflat" and r.get("seed") == 0 and r.get("ms")]
        pq = [r for r in rows if r["kind"] == "faiss_ivfpq" and r.get("ms")]
        ours = [r for r in rows if r["kind"] == "numpy_twolevel" and r.get("ms")]
        flat = [r for r in rows if r["kind"] == "faiss_flat" and r.get("ms")]
        if hnsw:
            ax.plot([r["ms"] for r in hnsw], [100 * r["recall"] for r in hnsw],
                    color="#009E73", marker="o", ms=3, lw=1.1, label="HNSW")
        if ivf:
            ax.scatter([r["ms"] for r in ivf], [100 * r["recall"] for r in ivf],
                       s=10, c="#444444", marker="s", label="IVF-Flat")
        if pq:
            ax.scatter([r["ms"] for r in pq], [100 * r["recall"] for r in pq],
                       s=12, facecolors="none", edgecolors="#D55E00", marker="^", label="IVF-PQ")
        cos = [r for r in ours if r.get("router") == "cos"]
        kf = [r for r in ours if str(r.get("router", "")).startswith("kflat")]
        if cos:
            ax.scatter([r["ms"] for r in cos], [100 * r["recall"] for r in cos],
                       s=12, c="#56B4E9", marker="x", label="NumPy cosine")
        if kf:
            ax.scatter([r["ms"] for r in kf], [100 * r["recall"] for r in kf],
                       s=12, c="#0072B2", marker="o", label="NumPy k-flat")
        if flat:
            ax.scatter([r["ms"] for r in flat], [100 * r["recall"] for r in flat],
                       s=14, c="#000000", marker="*", label="exact")
        ax.set_xscale("log")
        ax.set_title(f"{DS_NAME[ds]}, {EMB_NAME[emb]}", fontsize=7)
        ax.set_xlabel("ms / query")
        ax.set_ylabel("Recall@10")
    handles, labels = axes[0].get_legend_handles_labels()
    # de-duplicate labels
    seen = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig_ann.pdf")
    plt.close(fig)


def fig_aniso(summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
    corr = summary["aniso"]["corr"]
    for ax, key, title, stat in (
        (axes[0], "gain_recall", "Recall@10 gap (points)", corr["spearman_recall"]),
        (axes[1], "gain_ndcg", "nDCG@10 gap (points)", corr["spearman_ndcg"]),
    ):
        for emb, color, marker in (("minilm", "#0072B2", "o"), ("bge-small", "#D55E00", "s")):
            xs, ys = [], []
            for ds in DS_ORDER:
                row = summary["aniso"][f"{ds}|{emb}"]
                xs.append(100 * row["explained_frac_k16"])
                ys.append(100 * row[key])
                ax.annotate(DS_NAME[ds].replace("TREC-COVID", "TREC"), (100 * row["explained_frac_k16"], 100 * row[key]),
                            textcoords="offset points", xytext=(3, 2), fontsize=5.5, color=color)
            ax.scatter(xs, ys, c=color, marker=marker, s=22, label=EMB_NAME[emb], zorder=3)
        ax.axhline(0, color="#888888", lw=0.6)
        ax.set_xlabel("Variance in top 16 directions (%)")
        ax.set_ylabel(title)
        ax.set_title(f"Spearman {stat[0]:.2f} (p={stat[1]:.3f})", fontsize=8)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_aniso.pdf")
    plt.close(fig)


def main():
    summary = load_summary()
    routing = load_routing()
    fig_kcurve(summary, routing)
    fig_scan(routing, "recall", "fig_scan_recall.pdf", "Recall@10 (%)")
    fig_scan(routing, "ndcg", "fig_scan_ndcg.pdf", "nDCG@10 (%)")
    fig_cost(routing)
    fig_ann(summary)
    fig_aniso(summary)
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
