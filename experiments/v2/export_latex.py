"""LaTeX tables and number macros from the released JSON.

Every headline number in the paper is written here, rounded the same way
as the revision brief. The script checks those roundings and stops if the
JSON does not match the brief.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RES = ROOT / "results"
GEN = REPO / "generated"
GEN.mkdir(parents=True, exist_ok=True)

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
EMB_NAME = {"minilm": "MiniLM", "bge-small": "BGE"}
# Full-corpus sizes and query totals recorded with the released subsample.
CORPUS = {
    "scifact": (5183, 300),
    "nfcorpus": (3633, 323),
    "arguana": (8674, 1406),
    "scidocs": (25657, 1000),
    "fiqa": (57600, 648),
    "trec-covid": (171331, 50),
}
ROUTERS = [
    ("l2_k0", "Euclidean centroid ($k{=}0$)"),
    ("grass_k8", "Linear Grassmannian ($k{=}8$)"),
    ("kflat_k4", "$k$-flat, $k{=}4$"),
    ("kflat_k8", "$k$-flat, $k{=}8$"),
    ("kflat_k16", "$k$-flat, $k{=}16$"),
    ("kflat_k32", "$k$-flat, $k{=}32$"),
    ("ppca_k16", "PPCA likelihood ($k{=}16$)"),
]


def load_summary():
    return json.load(open(RES / "summary.json"))


def pct(x, signed=False, digits=1):
    v = 100.0 * float(x)
    if signed:
        return f"{v:+.{digits}f}"
    return f"{v:.{digits}f}"


def ci_pair(triple, digits=1):
    return pct(triple[1], True, digits), pct(triple[2], True, digits)


def sig_counts(summary, router, metric_key, C=32, P=4):
    holm = summary.get("_holm_nd")
    if holm is not None and metric_key == "d_ndcg_vs_cos":
        return holm[(router, C, P)]
    up = down = 0
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            row = summary["main"][f"{ds}|{emb}|{router}|C{C}|P{P}"]
            lo, hi = row[metric_key][1], row[metric_key][2]
            up += int(lo > 0)
            down += int(hi < 0)
    return up, down


def apply_query_level(summary):
    """Replace uncorrected CI counts with Holm rejections, and leaked envelopes with cross-validation."""
    path = ROOT / "results" / "query_level.json"
    if not path.exists():
        raise SystemExit("missing experiments/v2/results/query_level.json")
    ql = json.loads(path.read_text())
    summary["_ql"] = ql
    summary["_holm_nd"] = {}
    summary["_holm_flags"] = {}
    for comp in ql["comparisons"]:
        key = f"{comp['router']}|C{comp['C']}|P{comp['P']}"
        if comp["metric"] == "rec":
            row = summary["kcurve"].get(key)
            if row is None:
                continue
            if abs(row["d_recall"] - comp["mean"]) > 0.002:
                raise SystemExit(f"query-level mean drifted for {key}: {row['d_recall']} vs {comp['mean']}")
            row["n_sig_up"] = comp["n_up"]
            row["n_sig_down"] = comp["n_down"]
        else:
            summary["_holm_nd"][(comp["router"], comp["C"], comp["P"])] = (comp["n_up"], comp["n_down"])
        summary["_holm_flags"][(comp["router"], comp["C"], comp["P"], comp["metric"])] = {
            pair["pair"]: pair["reject"] for pair in comp["pairs"]
        }


def heldout_summary(summary, kind, budget):
    ql = summary.get("_ql")
    if ql is None:
        return None
    rec = nd = None
    for entry in ql["heldout"]:
        if entry["kind"] != kind or abs(float(entry["budget"]) - float(budget)) > 1e-9:
            continue
        if entry["metric"] == "recall":
            rec = entry
        elif entry["metric"] == "ndcg":
            nd = entry
    if rec is None or nd is None or rec["mean"] is None or nd["mean"] is None:
        return None
    return dict(
        n=rec["n_pairs"],
        dr=rec["mean"],
        dn=nd["mean"],
        r_up=rec["n_up"],
        r_dn=rec["n_down"],
        r_tie=rec["n_tie"],
        n_up=nd["n_up"],
        n_dn=nd["n_down"],
        n_tie=nd["n_tie"],
    )


def matched_rows(summary, cost, budget):
    xs = []
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            key = f"{cost}|{budget}|{ds}|{emb}"
            if key in summary["matched"]:
                xs.append(summary["matched"][key])
    return xs


def matched_summary(xs):
    if not xs:
        return None
    dr = np.array([a["kflat"][0] - a["ivf"][0] for a in xs])
    dn = np.array([a["kflat_bestndcg"][1] - a["ivf_bestndcg"][1] for a in xs])
    return {
        "n": len(xs),
        "dr": float(dr.mean()),
        "dn": float(dn.mean()),
        "r_up": int((dr > 0).sum()),
        "r_dn": int((dr < 0).sum()),
        "r_tie": int((dr == 0).sum()),
        "n_up": int((dn > 0).sum()),
        "n_dn": int((dn < 0).sum()),
        "n_tie": int((dn == 0).sum()),
    }


def check_brief(summary):
    expect = {
        "l2_k0": (-1.2, -0.3, 0, 8),
        "grass_k8": (-32.2, -11.1, 0, 12),
        "kflat_k4": (0.0, 0.3, 4, 3),
        "kflat_k8": (0.9, 0.6, 5, 1),
        "kflat_k16": (1.7, 1.3, 9, 0),
        "kflat_k32": (2.2, 1.9, 9, 0),
        "ppca_k16": (-3.7, -0.4, 2, 6),
    }
    for router, (er, en, up, dn) in expect.items():
        row = summary["kcurve"][f"{router}|C32|P4"]
        got = (round(100 * row["d_recall"], 1), round(100 * row["d_ndcg"], 1), row["n_sig_up"], row["n_sig_down"])
        if got != (er, en, up, dn):
            raise SystemExit(f"headline mismatch for {router}: JSON {got} vs brief {(er, en, up, dn)}")
    for key, er, en, up in (
        ("kflat_k16|C64|P4", 2.8, 2.2, 10),
        ("kflat_k16|C256|P16", 1.9, 1.3, 12),
    ):
        row = summary["kcurve"][key]
        got = (round(100 * row["d_recall"], 1), round(100 * row["d_ndcg"], 1), row["n_sig_up"])
        if got != (er, en, up):
            raise SystemExit(f"kcurve mismatch {key}: {got}")
    # nDCG significance is not the recall count. Lock the C=32, P=4 counts.
    ndcg_expect = {
        "kflat_k16": (6, 0),
        "kflat_k32": (8, 0),
        "l2_k0": (0, 6),
        "kflat_k8": (4, 0),
        "kflat_k4": (1, 0),
        "grass_k8": (0, 12),
        "ppca_k16": (2, 2),
    }
    for router, exp in ndcg_expect.items():
        got = sig_counts(summary, router, "d_ndcg_vs_cos")
        if got != exp:
            raise SystemExit(f"nDCG sig mismatch {router}: {got} vs {exp}")
    scan10 = matched_summary(matched_rows(summary, "cost_scan", 0.1))
    if (round(100 * scan10["dr"], 1), scan10["r_up"], scan10["r_dn"], round(100 * scan10["dn"], 2), scan10["n_up"]) != (1.0, 8, 3, 1.58, 12):
        raise SystemExit(f"scan 10% mismatch {scan10}")
    tot20 = matched_summary(matched_rows(summary, "cost_total", 0.2))
    if (round(100 * tot20["dr"], 1), tot20["r_up"], tot20["r_dn"]) != (-4.6, 0, 12):
        raise SystemExit(f"total 20% mismatch {tot20}")
    scan5 = matched_summary(matched_rows(summary, "cost_scan", 0.05))
    if (round(100 * scan5["dn"], 2), scan5["n_up"], scan5["n_dn"]) != (2.36, 12, 0):
        raise SystemExit(f"scan 5% nDCG mismatch {scan5}")


def macro(name, value):
    return f"\\DeclareRobustCommand{{\\{name}}}{{{value}}}"


def write_numbers(summary):
    lines = ["% Auto-generated from experiments/v2/results/summary.json. Do not edit."]
    kc = summary["kcurve"]

    def add_router(prefix, router, C=32, P=4):
        row = kc[f"{router}|C{C}|P{P}"]
        n_up, n_dn = sig_counts(summary, router, "d_ndcg_vs_cos", C, P) if (C, P) == (32, 4) else (None, None)
        lines.append(macro(f"{prefix}R", pct(row["d_recall"], True)))
        lines.append(macro(f"{prefix}N", pct(row["d_ndcg"], True)))
        lines.append(macro(f"{prefix}Up", str(row["n_sig_up"])))
        lines.append(macro(f"{prefix}Dn", str(row["n_sig_down"])))
        if n_up is not None:
            lines.append(macro(f"{prefix}NUp", str(n_up)))
            lines.append(macro(f"{prefix}NDn", str(n_dn)))

    add_router("KZero", "l2_k0")
    add_router("Grass", "grass_k8")
    add_router("KFour", "kflat_k4")
    add_router("KEight", "kflat_k8")
    add_router("KSixteen", "kflat_k16")
    add_router("KThirtyTwo", "kflat_k32")
    add_router("Ppca", "ppca_k16")
    add_router("KSixteenAtCSixtyFour", "kflat_k16", 64, 4)
    add_router("KSixteenAtCTwoFiveSix", "kflat_k16", 256, 16)

    for name, cost, budget in (
        ("ScanTwo", "cost_scan", 0.02),
        ("ScanFive", "cost_scan", 0.05),
        ("ScanTen", "cost_scan", 0.10),
        ("ScanTwenty", "cost_scan", 0.20),
        ("ScanForty", "cost_scan", 0.40),
        ("TotalFive", "cost_total", 0.05),
        ("TotalTen", "cost_total", 0.10),
        ("TotalTwenty", "cost_total", 0.20),
        ("TotalForty", "cost_total", 0.40),
    ):
        sm = heldout_summary(summary, cost, budget)
        if sm is None:
            lines.append(macro(f"{name}NPairs", "0"))
            lines.append(macro(f"{name}R", "n/a"))
            lines.append(macro(f"{name}ND", "n/a"))
            lines.append(macro(f"{name}RUp", "0"))
            lines.append(macro(f"{name}RDn", "0"))
            lines.append(macro(f"{name}RTie", "0"))
            lines.append(macro(f"{name}NUp", "0"))
            lines.append(macro(f"{name}NDn", "0"))
            continue
        lines.append(macro(f"{name}NPairs", str(sm["n"])))
        lines.append(macro(f"{name}R", pct(sm["dr"], True, 1)))
        lines.append(macro(f"{name}ND", pct(sm["dn"], True, 2)))
        lines.append(macro(f"{name}RUp", str(sm["r_up"])))
        lines.append(macro(f"{name}RDn", str(sm["r_dn"])))
        lines.append(macro(f"{name}RTie", str(sm["r_tie"])))
        lines.append(macro(f"{name}NUp", str(sm["n_up"])))
        lines.append(macro(f"{name}NDn", str(sm["n_dn"])))

    efs = [summary["aniso"][f"{ds}|{emb}"]["explained_frac_k16"] for ds in DS_ORDER for emb in EMB_ORDER]
    lines.append(macro("ExplainedLo", pct(min(efs))))
    lines.append(macro("ExplainedHi", pct(max(efs))))
    corr = summary["aniso"]["corr"]
    lines.append(macro("SpearmanR", f"{corr['spearman_recall'][0]:.2f}"))
    lines.append(macro("SpearmanRp", f"{corr['spearman_recall'][1]:.3f}"))
    lines.append(macro("PearsonR", f"{corr['pearson_recall'][0]:.2f}"))
    lines.append(macro("PearsonRp", f"{corr['pearson_recall'][1]:.3f}"))
    lines.append(macro("SpearmanN", f"{corr['spearman_ndcg'][0]:.2f}"))
    lines.append(macro("SpearmanNp", f"{corr['spearman_ndcg'][1]:.3f}"))

    # HNSW ef=16 vs NumPy k-flat k=16, C=32, P=4
    both = rec = lat = 0
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            rows = summary["baselines"][f"{ds}|{emb}"]
            h = next(r for r in rows if r["kind"] == "faiss_hnsw" and r.get("efSearch") == 16)
            k = next(r for r in rows if r["kind"] == "numpy_twolevel" and r.get("router") == "kflat_k16" and r.get("C") == 32 and r.get("P") == 4)
            rb = h["recall"] > k["recall"]
            lb = h["ms"] < k["ms"]
            rec += int(rb)
            lat += int(lb)
            both += int(rb and lb)
    lines.append(macro("HnswRecallWins", str(rec)))
    lines.append(macro("HnswLatWins", str(lat)))
    lines.append(macro("HnswBothWins", str(both)))

    route_us = []
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            rows = summary["baselines"][f"{ds}|{emb}"]
            k = next(r for r in rows if r["kind"] == "numpy_twolevel" and r.get("router") == "kflat_k16" and r.get("C") == 32 and r.get("P") == 4)
            route_us.append(k["route_only_ms"] * 1000)
    lines.append(macro("RouteUsLo", f"{min(route_us):.0f}"))
    lines.append(macro("RouteUsHi", f"{max(route_us):.0f}"))

    ch = summary["cheap"]
    lines.append(macro("CheapCosR", pct(ch["cos"]["recall"])))
    lines.append(macro("CheapKSixteenR", pct(ch["kflat16_fp32"]["recall"])))
    lines.append(macro("CheapIntDR", pct(ch["kflat16_int8"]["d_recall_vs_fp32"], True, 2)))
    lines.append(macro("CheapIntDN", pct(ch["kflat16_int8"]["d_ndcg_vs_fp32"], True, 2)))
    lines.append(macro("CheapFp16DR", pct(ch["kflat16_fp16"]["d_recall_vs_fp32"], True, 2)))
    lines.append(macro("CheapSharedDR", pct(ch["shared_k16"]["d_recall_vs_fp32"], True, 2)))
    lines.append(macro("CheapSharedDN", pct(ch["shared_k16"]["d_ndcg_vs_fp32"], True, 2)))
    lines.append(macro("CheapAdDR", pct(ch["adaptive_tau0.35"]["d_recall_vs_fp32"], True, 2)))
    lines.append(macro("CheapAdK", f"{ch['adaptive_tau0.35']['k_avg']:.1f}"))

    # negative point estimates for k=16 recall, for the prose
    neg = []
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            row = summary["main"][f"{ds}|{emb}|kflat_k16|C32|P4"]
            if row["d_recall_vs_cos"][0] < 0:
                neg.append(f"{ds}/{emb}:{pct(row['d_recall_vs_cos'][0], True)}")
    lines.append("% negative k=16 recall point estimates: " + ", ".join(neg))
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("negative k16 recall:", ", ".join(neg))


def cell_delta(triple, reject=False):
    mean, lo, hi = triple
    body = f"{pct(mean, True)} [{pct(lo, True)}, {pct(hi, True)}]"
    if reject:
        return f"\\textbf{{{body}}}"
    return body


def write_tables(summary):
    # Router means
    rows = []
    for key, label in ROUTERS:
        rec = summary["kcurve"][f"{key}|C32|P4"]
        n_up, n_dn = sig_counts(summary, key, "d_ndcg_vs_cos")
        rows.append(
            f"{label} & {pct(rec['d_recall'], True)} & {rec['n_sig_up']}/{rec['n_sig_down']} "
            f"& {pct(rec['d_ndcg'], True)} & {n_up}/{n_dn} \\\\"
        )
    table_routers = r"""\begin{table}[t]
\centering
\caption{Mean gap versus cosine IVF at $C{=}32$, $P{=}4$, unweighted over 12 dataset/embedder pairs. Counts are Holm-Bonferroni rejections at family-wise 5\% across those 12 pairs (positive, then negative). Recall and nDCG are separate families. The test is a two-sided percentile bootstrap of the seed-averaged per-query difference, 2000 resamples. Gaps are percentage points.}
\label{tab:routers}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\toprule
Router & $\Delta$R@10 & R up/down & $\Delta$nDCG@10 & n up/down \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (GEN / "table_routers.tex").write_text(table_routers)

    # Per-pair k=16
    body = []
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            cos = summary["main"][f"{ds}|{emb}|cos|C32|P4"]
            kf = summary["main"][f"{ds}|{emb}|kflat_k16|C32|P4"]
            blob = json.load(open(RES / f"routing__{ds}__{emb}.json"))
            flags = summary["_holm_flags"]
            pair = f"{ds}|{emb}"
            body.append(
                f"{DS_NAME[ds]} & {EMB_NAME[emb]} & {pct(cos['recall'][0])} & {pct(kf['recall'][0])} "
                f"& {cell_delta(kf['d_recall_vs_cos'], flags[('kflat_k16', 32, 4, 'rec')][pair])} & {pct(cos['ndcg'][0])} & {pct(kf['ndcg'][0])} "
                f"& {cell_delta(kf['d_ndcg_vs_cos'], flags[('kflat_k16', 32, 4, 'ndcg')][pair])} & {pct(blob['exact_ndcg'])} \\\\"
            )
    table_per = r"""\begin{table*}[t]
\centering
\caption{Cosine IVF and the $k{=}16$ k-flat router at $C{=}32$, $P{=}4$. All figures are percentages. Brackets are uncorrected 95\% percentile bootstrap intervals on the difference (k-flat minus cosine). Bold marks a Holm-Bonferroni rejection at family-wise 5\% across the 12 pairs, for that metric. An interval can exclude zero without being bold. Exact is full cosine search on the evaluated document set, not a published full-corpus BEIR number. TREC-COVID uses 50 queries, so its intervals are wide.}
\label{tab:perpair}
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{llrrrrrrr}
\toprule
Data & Model & R cos & R $k$16 & $\Delta$R [CI] & n cos & n $k$16 & $\Delta$n [CI] & Exact n \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (GEN / "table_perpair.tex").write_text(table_per)

    # Appendix: recall deltas for every router
    header_routers = ["l2_k0", "grass_k8", "kflat_k4", "kflat_k8", "kflat_k16", "kflat_k32", "ppca_k16"]
    drows = []
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            cells = [f"{DS_NAME[ds]}", EMB_NAME[emb]]
            for key in header_routers:
                triple = summary["main"][f"{ds}|{emb}|{key}|C32|P4"]["d_recall_vs_cos"]
                reject = summary["_holm_flags"][(key, 32, 4, "rec")][f"{ds}|{emb}"]
                cells.append(cell_delta(triple, reject))
            drows.append(" & ".join(cells) + r" \\")
    table_delta = r"""\begin{table*}[t]
\centering
\caption{Recall@10 gap versus cosine IVF, in points, at $C{=}32$, $P{=}4$. Brackets are uncorrected 95\% percentile bootstrap intervals. Bold marks a Holm-Bonferroni rejection at family-wise 5\% across the 12 pairs for that column.}
\label{tab:deltas}
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{llrrrrrrr}
\toprule
Data & Model & $k{=}0$ & Grass. & $k{=}4$ & $k{=}8$ & $k{=}16$ & $k{=}32$ & PPCA \\
\midrule
""" + "\n".join(drows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (GEN / "table_deltas.tex").write_text(table_delta)

    # Matched budgets
    mrows = []
    for label, cost, budget, macro_prefix in (
        ("2\\% scan", "cost_scan", 0.02, None),
        ("5\\% scan", "cost_scan", 0.05, None),
        ("10\\% scan", "cost_scan", 0.10, None),
        ("20\\% scan", "cost_scan", 0.20, None),
        ("40\\% scan", "cost_scan", 0.40, None),
        ("5\\% total", "cost_total", 0.05, None),
        ("10\\% total", "cost_total", 0.10, None),
        ("20\\% total", "cost_total", 0.20, None),
        ("40\\% total", "cost_total", 0.40, None),
        ("2\\% total", "cost_total", 0.02, None),
    ):
        sm = heldout_summary(summary, cost, budget)
        if sm is None:
            mrows.append(f"{label} & 0 & no feasible config & no feasible config \\\\")
            continue
        mrows.append(
            f"{label} & {sm['n']} & {pct(sm['dr'], True)} ({sm['r_up']}/{sm['r_dn']}) "
            f"& {pct(sm['dn'], True, 2)} ({sm['n_up']}/{sm['n_dn']}) \\\\"
        )
    table_matched = r"""\begin{table*}[t]
\centering
\caption{Held-out gap, k-flat minus cosine IVF, inside a multiply-add budget. For each of 5 query folds the config is chosen on the other folds and scored on the held-out fold. Recall and nDCG are selected separately. Parentheses are Holm-Bonferroni rejections across the pairs that have a feasible config on every fold (positive, then negative). When fewer than 12 pairs are feasible, that smaller set is the family. A pair whose held-out mean is positive need not reject.}
\label{tab:matched}
\small
\begin{tabular}{lrlr}
\toprule
Budget & Pairs & $\Delta$ recall (Holm up/down) & $\Delta$ nDCG (Holm up/down) \\
\midrule
""" + "\n".join(mrows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (GEN / "table_matched.tex").write_text(table_matched)

    # ANN
    arows = []
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            rows = summary["baselines"][f"{ds}|{emb}"]
            h = next(r for r in rows if r["kind"] == "faiss_hnsw" and r.get("efSearch") == 16)
            iv = next(r for r in rows if r["kind"] == "faiss_ivfflat" and r.get("nlist") == 32 and r.get("nprobe") == 4 and r.get("seed") == 0)
            k = next(r for r in rows if r["kind"] == "numpy_twolevel" and r.get("router") == "kflat_k16" and r.get("C") == 32 and r.get("P") == 4)
            arows.append(
                f"{DS_NAME[ds]} & {EMB_NAME[emb]} & {pct(h['recall'])} & {h['ms']:.3f} "
                f"& {pct(iv['recall'])} & {iv['ms']:.3f} & {pct(k['recall'])} & {k['ms']:.3f} \\\\"
            )
    table_ann = r"""\begin{table}[H]
\centering
\setlength{\abovecaptionskip}{4pt}%
\setlength{\belowcaptionskip}{2pt}%
\caption{Seed 0, one thread, batch size 1. Latency is the median of three passes over the queries. HNSW uses $M{=}32$, $\mathrm{efSearch}{=}16$. IVF-Flat uses 32 lists and 4 probes. k-flat is the NumPy router at $k{=}16$, $C{=}32$, $P{=}4$. FAISS is C++ and the k-flat column is Python, so the milliseconds are not an implementation-matched comparison. Recall is.}
\label{tab:ann}
\scriptsize
\renewcommand{\arraystretch}{0.84}%
\setlength{\tabcolsep}{2.5pt}%
\begin{tabular*}{\columnwidth}{@{}ll@{\extracolsep{\fill}}r@{\extracolsep{0pt}\hspace{0.55em}}r@{\extracolsep{\fill}}r@{\extracolsep{0pt}\hspace{0.55em}}r@{\extracolsep{\fill}}r@{\extracolsep{0pt}\hspace{0.55em}}r@{}}
\toprule
& & \multicolumn{2}{c}{HNSW} & \multicolumn{2}{c}{IVF} & \multicolumn{2}{c}{$k$-flat} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Data & Model & R & ms & R & ms & R & ms \\
\midrule
""" + "\n".join(arows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
    (GEN / "table_ann.tex").write_text(table_ann)

    # Anisotropy
    irows = []
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            row = summary["aniso"][f"{ds}|{emb}"]
            irows.append(
                f"{DS_NAME[ds]} & {EMB_NAME[emb]} & {pct(row['explained_frac_k16'])} "
                f"& {pct(row['gain_recall'], True)} & {pct(row['gain_ndcg'], True)} \\\\"
            )
    table_an = r"""\begin{center}
\captionof{table}{Size-weighted fraction of within-cluster variance captured by the top 16 principal directions at $C{=}32$ (mean over five seeds), and the $k{=}16$ gap versus cosine at $C{=}32$, $P{=}4$, in points.}
\label{tab:aniso}
\small
\begin{tabular}{llrrr}
\toprule
Data & Model & Var.\ @16 (\%) & $\Delta$R@10 & $\Delta$nDCG@10 \\
\midrule
""" + "\n".join(irows) + r"""
\bottomrule
\end{tabular}
\end{center}
"""
    (GEN / "table_aniso.tex").write_text(table_an)

    # Cheap variants, memory from the raw rows
    from collections import defaultdict
    agg = defaultdict(list)
    for path in sorted(RES.glob("cheap__*.json")):
        blob = json.load(open(path))
        for row in blob["rows"]:
            if row["C"] == 32 and row["P"] == 4:
                agg[row["variant"]].append(row)
    order = ["cos", "kflat16_fp32", "kflat16_fp16", "kflat16_int8", "shared_k16", "shared_k32", "adaptive_tau0.15", "adaptive_tau0.25", "adaptive_tau0.35"]
    labels = {
        "cos": "Cosine centroid",
        "kflat16_fp32": "$k$-flat $k{=}16$, fp32",
        "kflat16_fp16": "$k$-flat $k{=}16$, fp16",
        "kflat16_int8": "$k$-flat $k{=}16$, int8",
        "shared_k16": "One shared basis, $k{=}16$",
        "shared_k32": "One shared basis, $k{=}32$",
        "adaptive_tau0.15": "Adaptive, $\\tau{=}0.15$",
        "adaptive_tau0.25": "Adaptive, $\\tau{=}0.25$",
        "adaptive_tau0.35": "Adaptive, $\\tau{=}0.35$",
    }
    crows = []
    for key in order:
        xs = agg[key]
        rec = np.mean([r["recall"] for r in xs])
        nd = np.mean([r["ndcg"] for r in xs])
        dr = summary["cheap"][key]["d_recall_vs_fp32"]
        dn = summary["cheap"][key]["d_ndcg_vs_fp32"]
        kavg = np.mean([r["k_avg"] for r in xs])
        mem = np.mean([r["router_mem_bytes"] for r in xs]) / 1e3
        crows.append(
            f"{labels[key]} & {pct(rec)} & {pct(nd)} & {pct(dr, True, 2)} & {pct(dn, True, 2)} & {kavg:.1f} & {mem:.0f} \\\\"
        )
    table_cheap = r"""\begin{center}
\captionof{table}{Cheaper $k$-flat variants at $C{=}32$, $P{=}4$. Means over 12 pairs and five seeds. Deltas are versus fp32 $k{=}16$, in points. Router memory is the mean of the per-pair byte counts, in KB, and does not include the document vectors.}
\label{tab:cheap}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrrr}
\toprule
Variant & R@10 & nDCG & $\Delta$R & $\Delta$n & Mean $k$ & KB \\
\midrule
""" + "\n".join(crows) + r"""
\bottomrule
\end{tabular}
\end{center}
"""
    (GEN / "table_cheap.tex").write_text(table_cheap)

    # Dataset table
    drows = []
    for ds in DS_ORDER:
        full, qtot = CORPUS[ds]
        blob = json.load(open(RES / f"routing__{ds}__minilm.json"))
        bge = json.load(open(RES / f"routing__{ds}__bge-small.json"))
        if blob["N"] != json.load(open(RES / f"routing__{ds}__bge-small.json"))["N"]:
            raise SystemExit(f"doc count mismatch {ds}")
        if int(blob["N"]) > full:
            raise SystemExit(f"{ds} N exceeds full corpus")
        flag = "yes" if blob["N"] < full else "no"
        drows.append(
            f"{DS_NAME[ds]} & {blob['N']:,} & {full:,} & {flag} & {blob['nq']} & {qtot:,} "
            f"& {pct(blob['exact_ndcg'])} & {pct(bge['exact_ndcg'])} \\\\"
        )
    # The comma in 8,674 needs to not break if we use siunitx; we are not. But 8,674 in tabular is fine.
    # Wait: {blob['N']:,} produces 8,674 which is OK in text.
    table_data = r"""\begin{table*}[t]
\centering
\caption{Evaluation slices. Docs and queries are counts after the fixed subsample. Exact nDCG@10 is cosine search on that slice, in percent, and is not a full-corpus BEIR leaderboard number. FiQA and TREC-COVID are subsampled. TREC-COVID keeps 17{,}537 judged documents out of 30{,}000, so the slice is relevance-dense.}
\label{tab:data}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrrrr}
\toprule
Dataset & Docs & Full & Sub. & Q & Q full & Exact MiniLM & Exact BGE \\
\midrule
""" + "\n".join(drows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (GEN / "table_data.tex").write_text(table_data)

    # K-curve appendix, recall gap / nDCG gap and recall up-count
    krows = []
    for C in [16, 32, 64, 128, 256]:
        for P in [2, 4, 8, 16]:
            cells = [str(C), str(P)]
            for key in ["l2_k0", "kflat_k8", "kflat_k16", "kflat_k32", "grass_k8"]:
                id_ = f"{key}|C{C}|P{P}"
                if id_ not in summary["kcurve"]:
                    cells.append("n/a")
                    continue
                row = summary["kcurve"][id_]
                cells.append(f"{pct(row['d_recall'], True)}/{pct(row['d_ndcg'], True)} ({row['n_sig_up']})")
            krows.append(" & ".join(cells) + r" \\")
    table_kc = r"""\begin{table*}[t]
\centering
\caption{Unweighted mean gap versus cosine IVF, as recall/nDCG in points. The parenthesis is the number of Holm-Bonferroni rejections in the positive direction for recall, across the 12 pairs. Grassmannian routing was not run at $C \in \{128, 256\}$. When $P{=}C$ the probed set is the whole corpus and every gap is zero.}
\label{tab:kcurve}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{rrrrrrr}
\toprule
$C$ & $P$ & $k{=}0$ & $k{=}8$ & $k{=}16$ & $k{=}32$ & Grass. \\
\midrule
""" + "\n".join(krows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (GEN / "table_kcurve.tex").write_text(table_kc)
    print("wrote tables to", GEN)


def main():
    summary = load_summary()
    check_brief(summary)
    apply_query_level(summary)
    write_numbers(summary)
    write_tables(summary)


if __name__ == "__main__":
    main()
