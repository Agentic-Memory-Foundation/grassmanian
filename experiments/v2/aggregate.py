"""Aggregate raw results into results/summary.json and results/tables.txt.
CIs: per query, average the metric over the 5 k-means seeds, then bootstrap over queries (2000 resamples,
percentile 95% CI). Paired differences use the same per-query seed-averaged values."""
import json, glob, collections, os, numpy as np
from scipy.stats import spearmanr, pearsonr
from common import BASE, bootstrap_ci
RES = os.environ.get("ROUTING_OUT", f"{BASE}/results")
OUT = []
def p(s=""): OUT.append(s)
pairs = sorted(set(tuple(f.split("routing__")[1][:-5].split("__")) for f in glob.glob(f"{RES}/routing__*.json")))
DS_ORDER = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid"]; EMB_ORDER = ["minilm", "bge-small", "e5-small", "bge-base"]
pairs.sort(key=lambda t: (EMB_ORDER.index(t[1]), DS_ORDER.index(t[0])))
R = {}; A = {}
for ds, emb in pairs:
    R[(ds, emb)] = json.load(open(f"{RES}/routing__{ds}__{emb}.json"))
    A[(ds, emb)] = np.load(f"{RES}/routing__{ds}__{emb}.npz")
def n_seeds(ds, emb):
    a = A[(ds, emb)]
    ss = [int(k.rsplit("|s", 1)[1]) for k in a.files if k.startswith("rec|cos|C32|P4|s")]
    return (max(ss) + 1) if ss else 5
def perq(ds, emb, metric, router, C, P):
    a = A[(ds, emb)]
    ks = [f"{metric}|{router}|C{C}|P{P}|s{s}" for s in range(n_seeds(ds, emb))]
    return np.mean([a[k] for k in ks], 0)
ROUTERS = ["cos", "l2_k0", "grass_k8", "kflat_k4", "kflat_k8", "kflat_k16", "kflat_k32", "ppca_k16"]
summary = {"pairs": [list(x) for x in pairs], "main": {}, "kcurve": {}, "matched": {}, "aniso": {}, "baselines": {}, "cheap": {}}
p("SETUP: two-level index; MiniBatchKMeans partition (n_init=3, batch 1024) shared by all routers; exact cosine re-rank of the probed union;")
p("neighbor recall@10 vs exact cosine top-10; nDCG@10 vs BEIR test qrels (graded gains 2^rel-1). 5 k-means seeds. 95% CI: bootstrap over queries of seed-averaged per-query values.")
p("")
p("DATASETS:")
for ds, emb in pairs:
    if emb == "minilm":
        meta = json.load(open(f"{BASE}/data/{ds}.json"))["meta"]
        p(f"  {ds}: docs used {meta['n_docs']} of {meta['full_corpus']} ({'SUBSAMPLED' if meta['subsampled'] else 'full corpus'}), test queries used {meta['n_queries']} of {meta['n_test_queries_total']}")
p("")
for C, P in [(32, 4), (64, 8), (16, 2)]:
    p(f"TABLE MAIN C={C} P={P}: recall@10 [95% CI] | nDCG@10 [95% CI]; exact-search nDCG@10 shown as 'exact'")
    for ds, emb in pairs:
        ex = R[(ds, emb)]["exact_ndcg"]
        p(f"  {ds} / {emb}  (N={R[(ds,emb)]['N']}, queries={R[(ds,emb)]['nq']}, exact nDCG@10={ex:.3f})")
        base_r = perq(ds, emb, "rec", "cos", C, P); base_n = perq(ds, emb, "ndcg", "cos", C, P)
        for r in ROUTERS:
            rr = perq(ds, emb, "rec", r, C, P); nn = perq(ds, emb, "ndcg", r, C, P)
            cr = bootstrap_ci(rr); cn = bootstrap_ci(nn)
            dr = bootstrap_ci(rr - base_r); dn = bootstrap_ci(nn - base_n)
            cf = np.mean([c["cand_frac"] for c in R[(ds, emb)]["configs"] if c["router"] == r and c["C"] == C and c["P"] == P])
            summary["main"][f"{ds}|{emb}|{r}|C{C}|P{P}"] = dict(recall=cr, ndcg=cn, d_recall_vs_cos=dr, d_ndcg_vs_cos=dn, cand_frac=float(cf))
            p(f"    {r:10s} R@10 {100*cr[0]:5.1f} [{100*cr[1]:5.1f},{100*cr[2]:5.1f}]  dR {100*dr[0]:+5.1f} [{100*dr[1]:+5.1f},{100*dr[2]:+5.1f}] | nDCG {100*cn[0]:5.1f} [{100*cn[1]:5.1f},{100*cn[2]:5.1f}]  dN {100*dn[0]:+5.1f} [{100*dn[1]:+5.1f},{100*dn[2]:+5.1f}] | scanned {100*cf:4.1f}%")
    p("")
# k curve averaged over pairs
p("TABLE K-CURVE: mean over all dataset/embedder pairs of (router - cosine IVF), same C and P; recall then nDCG, in points. Count = pairs where the paired 95% CI is above zero / below zero.")
for C in [16, 32, 64, 128, 256]:
    for P in [2, 4, 8, 16]:
        line = f"  C={C:3d} P={P:2d}:"
        for r in ["l2_k0", "kflat_k4", "kflat_k8", "kflat_k16", "kflat_k32", "ppca_k16", "grass_k8"]:
            if C > 64 and r in ("kflat_k4", "ppca_k16", "grass_k8"): continue
            drs, dns, up, dn_ = [], [], 0, 0
            for ds, emb in pairs:
                a = perq(ds, emb, "rec", r, C, P) - perq(ds, emb, "rec", "cos", C, P)
                b = perq(ds, emb, "ndcg", r, C, P) - perq(ds, emb, "ndcg", "cos", C, P)
                ci = bootstrap_ci(a); drs.append(a.mean()); dns.append(b.mean())
                up += ci[1] > 0; dn_ += ci[2] < 0
            summary["kcurve"][f"{r}|C{C}|P{P}"] = dict(d_recall=float(np.mean(drs)), d_ndcg=float(np.mean(dns)), n_sig_up=int(up), n_sig_down=int(dn_), n_pairs=len(pairs))
            line += f"  {r}: {100*np.mean(drs):+.1f}/{100*np.mean(dns):+.1f} ({up}up,{dn_}dn)"
        p(line)
p("")
# compute-matched
p("TABLE COMPUTE-MATCHED: best seed-mean recall@10 (and the nDCG@10 of that config) reachable within a per-query cost budget, as a fraction of brute-force cost N*m.")
p("  cost_total = routing FLOPs (cos/l2: C*m; kflat_k: C*m*(k+1)) + scan FLOPs (candidates*m). cost_scan = scan only (the large-N regime where routing is negligible).")
p("  Families: IVF-cos = cosine centroid router, C in {16,32,64,128,256}, P in {1,2,4,...,64}; kflat = k-flat router, same C and P grid, k in {8,16,32} (plus k=4 at C<=64).")
p("  Two selections per family: the config with the best recall within budget, and separately the config with the best nDCG within budget.")
for costname in ["cost_total", "cost_scan"]:
    p(f"  [{costname}]")
    for B in [0.02, 0.05, 0.10, 0.20, 0.40]:
        wins = []; winsn = []; line = f"    budget {int(B*100):2d}%:"
        for ds, emb in pairs:
            r = R[(ds, emb)]; Nm = r["N"] * r["m"]
            agg = collections.defaultdict(list)
            for c in r["configs"]:
                agg[(c["router"], c["C"], c["P"])].append(c)
            best = {}; bestn = {}
            for (rt, C, P), cs in agg.items():
                fam = "ivf" if rt == "cos" else ("kflat" if rt.startswith("kflat") else None)
                if fam is None: continue
                cost = np.mean([x["flops_scan"] + (x["flops_route"] if costname == "cost_total" else 0) for x in cs]) / Nm
                if cost > B: continue
                rec = np.mean([x["recall"] for x in cs]); nd = np.mean([x["ndcg"] for x in cs])
                if fam not in best or rec > best[fam][0]: best[fam] = (rec, nd, f"{rt},C{C},P{P}")
                if fam not in bestn or nd > bestn[fam][1]: bestn[fam] = (rec, nd, f"{rt},C{C},P{P}")
            if "ivf" in best and "kflat" in best:
                wins.append(best["kflat"][0] - best["ivf"][0]); winsn.append(bestn["kflat"][1] - bestn["ivf"][1])
                summary["matched"][f"{costname}|{B}|{ds}|{emb}"] = dict(ivf=best["ivf"], kflat=best["kflat"], ivf_bestndcg=bestn["ivf"], kflat_bestndcg=bestn["kflat"])
        if wins:
            line += f" best recall kflat-minus-IVF: mean {100*np.mean(wins):+.1f} pts (kflat better {sum(w>0 for w in wins)}/{len(wins)}, worse {sum(w<0 for w in wins)}); best nDCG kflat-minus-IVF: mean {100*np.mean(winsn):+.2f} pts (better {sum(w>0 for w in winsn)}/{len(winsn)}, worse {sum(w<0 for w in winsn)})"
        p(line)
p("")
# per pair detail for matched (minilm + bge-small) at 10% total and 10% scan
p("TABLE COMPUTE-MATCHED DETAIL (budget 10%): best config per family, recall/nDCG")
for costname in ["cost_total", "cost_scan"]:
    for ds, emb in pairs:
        k = f"{costname}|0.1|{ds}|{emb}"
        if k in summary["matched"]:
            m_ = summary["matched"][k]
            p(f"  [{costname}] {ds}/{emb}: best-recall IVF-cos {100*m_['ivf'][0]:.1f}/{100*m_['ivf'][1]:.1f} ({m_['ivf'][2]}) vs kflat {100*m_['kflat'][0]:.1f}/{100*m_['kflat'][1]:.1f} ({m_['kflat'][2]}); best-nDCG IVF-cos {100*m_['ivf_bestndcg'][1]:.1f} ({m_['ivf_bestndcg'][2]}) vs kflat {100*m_['kflat_bestndcg'][1]:.1f} ({m_['kflat_bestndcg'][2]})")
p("")
# anisotropy
p("TABLE ANISOTROPY: mean fraction of within-cluster variance captured by top-k PCA directions (C=32, k=16, size-weighted, mean over seeds) vs gain of kflat_k16 over cos at C=32,P=4")
xs, ys, yn = [], [], []
for ds, emb in pairs:
    ef = np.mean([s["explained_frac_wmean"] for s in R[(ds, emb)]["cluster_stats"] if s["C"] == 32 and s["k"] == 16])
    g = summary["main"][f"{ds}|{emb}|kflat_k16|C32|P4"]["d_recall_vs_cos"][0]
    gn = summary["main"][f"{ds}|{emb}|kflat_k16|C32|P4"]["d_ndcg_vs_cos"][0]
    xs.append(ef); ys.append(g); yn.append(gn)
    summary["aniso"][f"{ds}|{emb}"] = dict(explained_frac_k16=float(ef), gain_recall=g, gain_ndcg=gn)
    p(f"  {ds:10s} {emb:9s} explained(k=16) {ef:.3f}  gain R@10 {100*g:+.1f}  gain nDCG {100*gn:+.1f}")
if len(xs) >= 4:
    sr = spearmanr(xs, ys); pr = pearsonr(xs, ys); srn = spearmanr(xs, yn)
    summary["aniso"]["corr"] = dict(spearman_recall=[float(sr[0]), float(sr[1])], pearson_recall=[float(pr[0]), float(pr[1])], spearman_ndcg=[float(srn[0]), float(srn[1])], n=len(xs))
    p(f"  Spearman(explained, recall gain) = {sr[0]:.2f} (p={sr[1]:.3f}); Pearson = {pr[0]:.2f} (p={pr[1]:.3f}); Spearman(explained, nDCG gain) = {srn[0]:.2f} (p={srn[1]:.3f}); n={len(xs)} pairs")
p("")
# baselines
bfiles = sorted(glob.glob(f"{RES}/baselines__*.json"))
if bfiles:
    p("TABLE FAISS + LATENCY (seed 0, 1 thread, per-query latency = median of 3 passes over all queries, ms):")
    for f in bfiles:
        b = json.load(open(f)); p(f"  {b['ds']} / {b['emb']} (N={b['N']})")
        sel = []
        for x in b["rows"]:
            k = x["kind"]
            if k == "faiss_flat": sel.append(("FAISS Flat (exact)", x))
            if k == "faiss_ivfflat" and x["seed"] == 0 and (x["nlist"], x["nprobe"]) in [(16, 4), (32, 4), (32, 8), (64, 8), (128, 16)]: sel.append((f"FAISS IVF-Flat nlist={x['nlist']} nprobe={x['nprobe']}", x))
            if k == "faiss_ivfpq" and x["nlist"] == 32 and x["nprobe"] in (4, 8): sel.append((f"FAISS IVF-PQ48 nlist=32 nprobe={x['nprobe']} refine={x['refine']}", x))
            if k == "faiss_hnsw" and x["efSearch"] in (16, 32, 64): sel.append((f"FAISS HNSW M=32 ef={x['efSearch']}", x))
            if k == "numpy_twolevel" and x["C"] == 32 and x["router"] in ("cos", "kflat_k8", "kflat_k16", "kflat_k32"): sel.append((f"ours numpy {x['router']} C=32 P={x['P']}", x))
        for name, x in sel:
            extra = f" route-only {x['route_only_ms']*1000:.0f}us" if "route_only_ms" in x else ""
            p(f"    {name:45s} R@10 {100*x['recall']:5.1f} nDCG {100*x['ndcg']:5.1f}  {x['ms']:.3f} ms{extra}  mem {x['mem_bytes']/1e6:.1f} MB")
        summary["baselines"][f"{b['ds']}|{b['emb']}"] = b["rows"]
    p("")
cfiles = sorted(glob.glob(f"{RES}/cheap__*.json"))
if cfiles:
    p("TABLE CHEAPER VARIANTS (C=32, P=4; mean over seeds and over the listed pairs; recall / nDCG in %, change vs fp32 k=16 in points; router memory in KB)")
    agg = collections.defaultdict(list); names = []
    for f in cfiles:
        c = json.load(open(f)); names.append(f"{c['ds']}/{c['emb']}")
        ref = {}
        for x in c["rows"]:
            if x["C"] == 32 and x["P"] == 4:
                agg[x["variant"]].append((c["ds"], c["emb"], x["seed"], x["recall"], x["ndcg"], x["k_avg"], x["router_mem_bytes"]))
    p("  pairs: " + ", ".join(names))
    refm = {(a, b, s): (r, n) for a, b, s, r, n, _, _ in agg["kflat16_fp32"]}
    for v, xs_ in agg.items():
        rec = np.mean([x[3] for x in xs_]); nd = np.mean([x[4] for x in xs_])
        dr = np.mean([x[3] - refm[(x[0], x[1], x[2])][0] for x in xs_]); dn = np.mean([x[4] - refm[(x[0], x[1], x[2])][1] for x in xs_])
        summary["cheap"][v] = dict(recall=float(rec), ndcg=float(nd), d_recall_vs_fp32=float(dr), d_ndcg_vs_fp32=float(dn), k_avg=float(np.mean([x[5] for x in xs_])))
        p(f"    {v:18s} R@10 {100*rec:5.1f} ({100*dr:+.2f})  nDCG {100*nd:5.1f} ({100*dn:+.2f})  avg k {np.mean([x[5] for x in xs_]):.1f}  router mem (C=32, m from data) {np.mean([x[6] for x in xs_])/1e3:.0f} KB")
json.dump(summary, open(f"{RES}/summary.json", "w"))
open(f"{RES}/tables.txt", "w").write("\n".join(OUT))
print("\n".join(OUT))
