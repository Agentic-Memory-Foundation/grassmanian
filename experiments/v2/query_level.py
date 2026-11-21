"""Query-level inference for the paper.

Two products, both from the per-query arrays written by run_routing.py:

1. Holm-adjusted paired bootstrap tests. For one comparison (one router, one
   C, one P, one metric) the family is the 12 dataset/embedder pairs. The test
   is a two-sided percentile bootstrap of the seed-averaged per-query
   difference versus cosine IVF (2000 resamples, Generator seed 0). A pair is
   counted only if Holm-Bonferroni rejects at family-wise alpha 0.05.

2. Budget envelopes chosen by 5-fold query cross-validation. On each fold the
   config is the one with the best selection-split mean inside the budget.
   The reported difference is that config's metric on the held-out fold.
   The same bootstrap test and the same Holm adjustment are used for the counts.

Usage:
    python experiments/v2/query_level.py [routing_dir]

Default routing_dir is experiments/v2/results_repro, then experiments/v2/results
if that directory has no npz files.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DS_ORDER = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid"]
EMB_ORDER = ["minilm", "bge-small"]
N_BOOT = 2000
BOOT_SEED = 0
N_FOLD = 5
FOLD_SEED = 0
ALPHA = 0.05
BUDGETS = [0.02, 0.05, 0.10, 0.20, 0.40]


def bootstrap_stats(x, n=N_BOOT, seed=BOOT_SEED):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), p=1.0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    # Two-sided p for a mean of zero. A zero tail is stored as 1/n, which is
    # an upper bound, so Holm cannot treat an empty tail as a smaller p than
    # the Monte Carlo resolution.
    tail = min(float(np.mean(means <= 0.0)), float(np.mean(means >= 0.0)))
    p = min(1.0, 2.0 * tail)
    if p == 0.0:
        p = 1.0 / n
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(x.mean()), lo=float(lo), hi=float(hi), p=float(p))


def holm_reject(pvals, alpha=ALPHA):
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    order = np.argsort(pvals, kind="mergesort")
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def holm_summary(stats):
    """stats: list of bootstrap_stats dicts, one per pair, in a fixed order."""
    reject = holm_reject([s["p"] for s in stats])
    up = int(sum(reject[i] and stats[i]["mean"] > 0 for i in range(len(stats))))
    down = int(sum(reject[i] and stats[i]["mean"] < 0 for i in range(len(stats))))
    return dict(n=len(stats), n_up=up, n_down=down, reject=[bool(r) for r in reject])


def assign_folds(nq, n_fold=N_FOLD, seed=FOLD_SEED):
    rng = np.random.default_rng(seed)
    folds = np.empty(nq, dtype=np.int32)
    folds[rng.permutation(nq)] = np.arange(nq) % n_fold
    return folds


def family(router):
    if router == "cos":
        return "ivf"
    if router.startswith("kflat_k"):
        return "kflat"
    return None


def k_of(router):
    if router.startswith("kflat_k"):
        return int(router.split("k")[-1])
    return 0


def load_pair(routing_dir, ds, emb):
    blob = json.loads((routing_dir / f"routing__{ds}__{emb}.json").read_text())
    npz = np.load(routing_dir / f"routing__{ds}__{emb}.npz")
    grouped = defaultdict(list)
    for cfg in blob["configs"]:
        grouped[(cfg["router"], cfg["C"], cfg["P"])].append(cfg)
    rows = []
    for (router, C, P), cfgs in grouped.items():
        fam = family(router)
        seeds = sorted(c["seed"] for c in cfgs)
        rec, ndcg, ncand = [], [], []
        for seed in seeds:
            key = f"{router}|C{C}|P{P}|s{seed}"
            rec.append(npz["rec|" + key])
            ndcg.append(npz["ndcg|" + key])
            ncand.append(npz["ncand|" + key])
        route = float(cfgs[0]["flops_route"])
        rows.append(dict(
            router=router, C=int(C), P=int(P), fam=fam, k=k_of(router),
            route=route,
            rec=np.mean(rec, axis=0),
            ndcg=np.mean(ndcg, axis=0),
            ncand=np.mean(ncand, axis=0),
        ))
    return blob, rows


def cost_frac(row, idx, N, m, kind):
    scan = float(row["ncand"][idx].mean()) * m
    total = row["route"] + scan
    return (total if kind == "cost_total" else scan) / (N * m)


def choose(rows, fam, idx, metric, kind, budget, N, m):
    best = None
    for row in rows:
        if row["fam"] != fam:
            continue
        cost = cost_frac(row, idx, N, m, kind)
        if cost > budget:
            continue
        value = float(row[metric][idx].mean())
        cand = (value, -cost, -row["C"], -row["P"], -row["k"], row)
        if best is None or cand[:5] > best[:5]:
            best = cand
    return None if best is None else best[-1]


def crossval_pair(rows, blob, kind, budget, do_boot=True):
    N, m = blob["N"], blob["m"]
    nq = rows[0]["rec"].shape[0]
    folds = assign_folds(nq)
    held = {fam: {metric: np.full(nq, np.nan) for metric in ("rec", "ndcg")} for fam in ("ivf", "kflat")}
    chosen = {fam: {metric: [] for metric in ("rec", "ndcg")} for fam in ("ivf", "kflat")}
    n_ok = {metric: 0 for metric in ("rec", "ndcg")}
    for fold in range(N_FOLD):
        sel = np.flatnonzero(folds != fold)
        ev = np.flatnonzero(folds == fold)
        picks = {}
        for metric in ("rec", "ndcg"):
            ivf = choose(rows, "ivf", sel, metric, kind, budget, N, m)
            kflat = choose(rows, "kflat", sel, metric, kind, budget, N, m)
            picks[metric] = (ivf, kflat)
            if ivf is None or kflat is None:
                continue
            n_ok[metric] += 1
            held["ivf"][metric][ev] = ivf[metric][ev]
            held["kflat"][metric][ev] = kflat[metric][ev]
            chosen["ivf"][metric].append(f"{ivf['router']},C{ivf['C']},P{ivf['P']}")
            chosen["kflat"][metric].append(f"{kflat['router']},C{kflat['C']},P{kflat['P']}")
    out = dict(n_queries=int(nq), folds_ok=n_ok, chosen=chosen)
    for metric, key in (("rec", "recall"), ("ndcg", "ndcg")):
        mask = np.isfinite(held["ivf"][metric]) & np.isfinite(held["kflat"][metric])
        diff = held["kflat"][metric][mask] - held["ivf"][metric][mask]
        out[key] = bootstrap_stats(diff) if do_boot else dict(mean=float(diff.mean()) if mask.any() else float("nan"), lo=None, hi=None, p=None)
        out[key]["n_eval"] = int(mask.sum())
    return out


def fixed_comparison(rows_by_pair, router, C, P, metric):
    stats = []
    for ds, emb in ((d, e) for d in DS_ORDER for e in EMB_ORDER):
        blob, rows = rows_by_pair[(ds, emb)]
        base = next(r for r in rows if r["router"] == "cos" and r["C"] == C and r["P"] == P)
        alt = next(r for r in rows if r["router"] == router and r["C"] == C and r["P"] == P)
        diff = alt[metric] - base[metric]
        st = bootstrap_stats(diff)
        st["pair"] = f"{ds}|{emb}"
        stats.append(st)
    summary = holm_summary(stats)
    summary["mean"] = float(np.mean([s["mean"] for s in stats]))
    summary["pairs"] = [
        dict(pair=s["pair"], mean=s["mean"], lo=s["lo"], hi=s["hi"], p=s["p"], reject=summary["reject"][i])
        for i, s in enumerate(stats)
    ]
    del summary["reject"]
    return summary


def curve(rows_by_pair, kind, metric, budgets):
    """Held-out mean of the selected config, one value per pair per budget."""
    out = []
    for b in budgets:
        vals = []
        for ds, emb in ((d, e) for d in DS_ORDER for e in EMB_ORDER):
            blob, rows = rows_by_pair[(ds, emb)]
            block = crossval_pair(rows, blob, kind, float(b), do_boot=False)
            key = "recall" if metric == "rec" else "ndcg"
            complete = (
                block[key]["n_eval"] == block["n_queries"]
                and block["folds_ok"][metric] == N_FOLD
            )
            vals.append(block[key]["mean"] if complete else float("nan"))
        out.append(dict(budget=float(b), values=vals))
    return out


def main():
    if len(sys.argv) > 1:
        routing_dir = Path(sys.argv[1])
    else:
        repro = ROOT / "results_repro"
        released = ROOT / "results"
        routing_dir = repro if list(repro.glob("routing__*.npz")) else released
    missing = [
        ds + "/" + emb
        for ds in DS_ORDER for emb in EMB_ORDER
        if not (routing_dir / f"routing__{ds}__{emb}.npz").exists()
    ]
    if missing:
        raise SystemExit(f"missing per-query arrays in {routing_dir}: {missing}")

    rows_by_pair = {}
    for ds in DS_ORDER:
        for emb in EMB_ORDER:
            rows_by_pair[(ds, emb)] = load_pair(routing_dir, ds, emb)
            # sanity: seed-averaged recall matches the json mean
            blob, rows = rows_by_pair[(ds, emb)]
            probe = next(r for r in rows if r["router"] == "kflat_k16" and r["C"] == 32 and r["P"] == 4)
            json_mean = np.mean([
                c["recall"] for c in blob["configs"]
                if c["router"] == "kflat_k16" and c["C"] == 32 and c["P"] == 4
            ])
            if abs(float(probe["rec"].mean()) - json_mean) > 1e-4:
                raise SystemExit(f"npz/json mismatch {ds} {emb}: {probe['rec'].mean()} vs {json_mean}")

    comparisons = []
    specs = [
        ("l2_k0", 32, 4), ("grass_k8", 32, 4), ("kflat_k4", 32, 4),
        ("kflat_k8", 32, 4), ("kflat_k16", 32, 4), ("kflat_k32", 32, 4),
        ("ppca_k16", 32, 4), ("kflat_k16", 64, 4), ("kflat_k16", 256, 16),
    ]
    # The k-curve table reports every C, P in the appendix. Cover that grid.
    for C in (16, 32, 64, 128, 256):
        for P in (2, 4, 8, 16):
            for router in ("l2_k0", "kflat_k8", "kflat_k16", "kflat_k32"):
                specs.append((router, C, P))
            if C <= 64:
                for router in ("kflat_k4", "grass_k8", "ppca_k16"):
                    specs.append((router, C, P))
    seen = set()
    for router, C, P in specs:
        key = (router, C, P)
        if key in seen:
            continue
        seen.add(key)
        for metric in ("rec", "ndcg"):
            block = fixed_comparison(rows_by_pair, router, C, P, metric)
            comparisons.append(dict(router=router, C=C, P=P, metric=metric, **block))
            print(
                f"{router:12s} C{C:<3d} P{P:<2d} {metric:5s} "
                f"{100*block['mean']:+6.2f}  holm {block['n_up']}/{block['n_down']}",
                flush=True,
            )

    heldout = []
    for kind in ("cost_total", "cost_scan"):
        for budget in BUDGETS:
            pair_rows = []
            for ds in DS_ORDER:
                for emb in EMB_ORDER:
                    blob, rows = rows_by_pair[(ds, emb)]
                    block = crossval_pair(rows, blob, kind, budget)
                    block["pair"] = f"{ds}|{emb}"
                    pair_rows.append(block)
            for metric in ("recall", "ndcg"):
                usable = [r for r in pair_rows if r[metric]["n_eval"] == r["n_queries"] and r["folds_ok"][{"recall": "rec", "ndcg": "ndcg"}[metric]] == N_FOLD]
                stats = [r[metric] for r in usable]
                holm = holm_summary(stats) if stats else dict(n=0, n_up=0, n_down=0, reject=[])
                means = [s["mean"] for s in stats]
                entry = dict(
                    kind=kind, budget=budget, metric=metric,
                    n_pairs=len(usable),
                    mean=float(np.mean(means)) if means else None,
                    n_up=holm["n_up"], n_down=holm["n_down"],
                    pairs=[
                        dict(
                            pair=r["pair"],
                            mean=r[metric]["mean"], lo=r[metric]["lo"], hi=r[metric]["hi"],
                            p=r[metric]["p"], n_eval=r[metric]["n_eval"],
                            folds_ok=r["folds_ok"][{"recall": "rec", "ndcg": "ndcg"}[metric]],
                            chosen_ivf=r["chosen"]["ivf"][{"recall": "rec", "ndcg": "ndcg"}[metric]],
                            chosen_kflat=r["chosen"]["kflat"][{"recall": "rec", "ndcg": "ndcg"}[metric]],
                            reject=bool(holm["reject"][i]) if stats else False,
                        )
                        for i, r in enumerate(usable)
                    ],
                )
                # ties: mean difference exactly zero
                entry["n_tie"] = int(sum(abs(s["mean"]) < 1e-12 for s in stats))
                heldout.append(entry)
                print(
                    f"heldout {kind:11s} {100*budget:4.0f}% {metric:6s} "
                    f"n={entry['n_pairs']:2d} mean={None if entry['mean'] is None else round(100*entry['mean'], 2)} "
                    f"holm {entry['n_up']}/{entry['n_down']}",
                    flush=True,
                )

    budgets = np.logspace(np.log10(0.02), np.log10(0.5), 28)
    curves = {
        "cost_total_recall": curve(rows_by_pair, "cost_total", "rec", budgets),
        "cost_scan_recall": curve(rows_by_pair, "cost_scan", "rec", budgets),
        "cost_scan_ndcg": curve(rows_by_pair, "cost_scan", "ndcg", budgets),
    }
    out = dict(
        test="two-sided percentile bootstrap of the seed-averaged per-query difference versus cosine, 2000 resamples, Generator seed 0",
        multiple_comparison="Holm-Bonferroni across the 12 dataset/embedder pairs of that comparison, family-wise alpha 0.05. Recall and nDCG are separate families. Rows of a table are not further adjusted.",
        heldout_protocol="5-fold query cross-validation, folds from Generator seed 0. The config maximizes the selection-split mean inside the budget. Ties break toward lower cost, then smaller C, P, and k. The difference is evaluated on the held-out fold.",
        routing_dir=str(routing_dir),
        comparisons=comparisons,
        heldout=heldout,
        curves={k: v for k, v in curves.items()},
    )
    dest = ROOT / "results" / "query_level.json"
    dest.write_text(json.dumps(out))
    print("wrote", dest)


if __name__ == "__main__":
    main()
