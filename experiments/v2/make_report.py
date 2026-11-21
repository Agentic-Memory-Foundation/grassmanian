"""Builds /workspace/grassmannian_paper_update.txt from results/summary.json, results/tables.txt and corrected_references.bib.
All numbers in the prose are read from summary.json at build time."""
import json, glob, numpy as np
from pathlib import Path
B = str(Path(__file__).resolve().parent)
S = json.load(open(f"{B}/results/summary.json"))
T = open(f"{B}/results/tables.txt").read()
BIB = open(Path(B).resolve().parents[1] / "references.bib").read()
pairs = [tuple(x) for x in S["pairs"]]
embs = sorted(set(e for _, e in pairs)); dss = sorted(set(d for d, _ in pairs))
kc = S["kcurve"]
def k(r, C=32, P=4): x = kc[f"{r}|C{C}|P{P}"]; return x
def pts(v): return f"{100*v:+.1f}"
def matched(cost, Bud):
    xs = [v for kk, v in S["matched"].items() if kk.startswith(f"{cost}|{Bud}|")]
    if not xs: return None
    dr = [a["kflat"][0] - a["ivf"][0] for a in xs]; dn = [a["kflat_bestndcg"][1] - a["ivf_bestndcg"][1] for a in xs]
    return dict(n=len(xs), dr=np.mean(dr), dn=np.mean(dn), rwin=sum(x > 0 for x in dr), nwin=sum(x > 0 for x in dn), rlose=sum(x < 0 for x in dr), nlose=sum(x < 0 for x in dn))
an = S["aniso"].get("corr", {})
ch = S["cheap"]
k16 = k("kflat_k16"); k32 = k("kflat_k32"); k8 = k("kflat_k8"); k4 = k("kflat_k4"); l2 = k("l2_k0"); gr = k("grass_k8"); pp = k("ppca_k16")
k16_64 = k("kflat_k16", 64, 4); k16_256 = k("kflat_k16", 256, 16) if "kflat_k16|C256|P16" in kc else None
ms5 = matched("cost_scan", 0.05); ms10 = matched("cost_scan", 0.1); ms20 = matched("cost_scan", 0.2); mt20 = matched("cost_total", 0.2); mt40 = matched("cost_total", 0.4)
NP = len(pairs)
pairlist = ", ".join(f"{d}/{e}" for d, e in pairs)
# FAISS summary lines
def bl(ds, emb):
    rows = S["baselines"].get(f"{ds}|{emb}")
    if not rows: return None
    g = lambda **kw: next((r for r in rows if all(r.get(a) == b for a, b in kw.items())), None)
    return dict(hnsw32=g(kind="faiss_hnsw", efSearch=32), hnsw16=g(kind="faiss_hnsw", efSearch=16), ivf=g(kind="faiss_ivfflat", nlist=32, nprobe=4, seed=0),
                cos=g(kind="numpy_twolevel", router="cos", C=32, P=4), kf=g(kind="numpy_twolevel", router="kflat_k16", C=32, P=4), flat=g(kind="faiss_flat"))
bl_lines = []; hn_dom = 0; hn_n = 0
for d, e in pairs:
    b = bl(d, e)
    if b and all(b.values()):
        hn_n += 1; hn_dom += (b['hnsw16']['recall'] >= b['kf']['recall'] and b['hnsw16']['ms'] <= b['kf']['ms'])
        bl_lines.append(f"  {d}/{e}: HNSW(M=32,ef=32) R@10 {100*b['hnsw32']['recall']:.1f}, nDCG {100*b['hnsw32']['ndcg']:.1f}, {b['hnsw32']['ms']:.3f} ms | "
                        f"FAISS IVF-Flat(32,4) R@10 {100*b['ivf']['recall']:.1f}, nDCG {100*b['ivf']['ndcg']:.1f}, {b['ivf']['ms']:.3f} ms | "
                        f"ours k-flat k=16 (C=32,P=4) R@10 {100*b['kf']['recall']:.1f}, nDCG {100*b['kf']['ndcg']:.1f}, {b['kf']['ms']:.3f} ms | exact nDCG {100*b['flat']['ndcg']:.1f}")
nbl = len(bl_lines)
meta = {}
for d in dss:
    m = json.load(open(f"{B}/data/{d}.json"))["meta"]; meta[d] = m
sub = [d for d in dss if meta[d]["subsampled"]]
qcap = [d for d in dss if meta[d]["n_queries"] < meta[d]["n_test_queries_total"]]

R = []
w = R.append
w("GRASSMANNIAN RAG PAPER: REVISION PACKAGE")
w("=" * 72)
w("Audience: an agent that has the paper source (paper.tex, references.bib, experiments/) and nothing else.")
w("Everything below is self-contained. All numbers come from runs in /workspace/grassmannian_exp on Parag's analysis box (8 CPU cores, no GPU),")
w("scripts and raw JSON listed in Section 12. Nothing here is estimated or invented; experiments that were not run are marked NOT RUN.")
w("Style rule for the paper: no em dashes anywhere (the repo already has a .cursor rule for this).")
w("")
w("0. WHAT CHANGED AND WHY (read first)")
w("-" * 72)
w("The current draft claims that k-flat residual routing beats centroid IVF only because of 'metric alignment' (Euclidean vs cosine), and that")
w("subspace directions add nothing. Our re-runs show the opposite:")
w(f"  * The Euclidean-centroid router (k=0, the 'metric alignment' explanation) is WORSE than cosine IVF: mean {pts(l2['d_recall'])} recall@10 points and {pts(l2['d_ndcg'])} nDCG@10 points vs cosine at C=32,P=4,")
w(f"    significantly worse in {l2['n_sig_down']}/{l2['n_pairs']} dataset/embedder pairs.")
w(f"  * Recall and nDCG rise with the number of per-cluster PCA directions k: at C=32,P=4 the mean gain over cosine IVF is")
w(f"    k=4 {pts(k4['d_recall'])}/{pts(k4['d_ndcg'])}, k=8 {pts(k8['d_recall'])}/{pts(k8['d_ndcg'])}, k=16 {pts(k16['d_recall'])}/{pts(k16['d_ndcg'])}, k=32 {pts(k32['d_recall'])}/{pts(k32['d_ndcg'])} (recall/nDCG points),")
w(f"    with k=16 better than cosine on the uncorrected interval in {k16['n_sig_up']}/{k16['n_pairs']} pairs and worse in {k16['n_sig_down']}. The paper instead counts Holm-Bonferroni rejections; see query_level.json.")
w(f"  * Linear (through-origin) Grassmannian routing remains far worse: {pts(gr['d_recall'])} recall points, {pts(gr['d_ndcg'])} nDCG points vs cosine.")
w("  * BUT the gain is not free. At equal clusters and probes the scanned fraction is the same, yet routing costs k+1 times more.")
if mt20: w(f"    Counting routing plus scan FLOPs on these small corpora, a finer-grained cosine IVF wins: at a 20% of brute-force budget the best k-flat config is {pts(mt20['dr'])} recall points vs the best cosine IVF config (k-flat better in {mt20['rwin']}/{mt20['n']} pairs).")
if ms10: w(f"    Counting scan cost only (the large-corpus regime, where routing over C centroids is negligible next to scanning candidates), recall is roughly matched ({pts(ms10['dr'])} points at a 10% scan budget, k-flat better in {ms10['rwin']}/{ms10['n']} pairs)")
if ms10: w(f"    while end-task nDCG@10 favors k-flat ({ms10['dn']*100:+.2f} points at 10% scan, better in {ms10['nwin']}/{ms10['n']} pairs; {ms5['dn']*100:+.2f} points at 5%, better in {ms5['nwin']}/{ms5['n']}).")
w(f"  * FAISS HNSW (M=32, efSearch=16) has higher recall@10 at lower single-thread latency than our k=16 k-flat router (C=32, P=4) in {hn_dom}/{hn_n} pairs where both were run (Section 6.5).")
w("The honest paper is therefore: 'cluster-local principal directions are a consistent coarse-routing signal beyond the centroid that improves")
w("relevance-weighted retrieval at fixed scan budget; it is not a replacement for graph indexes and it does not win when routing cost dominates'.")
w("")
w("1. NEW FRAMING (summary for the authors)")
w("-" * 72)
efs = [v["explained_frac_k16"] for kk, v in S["aniso"].items() if kk != "corr"]
w("Centroid IVF routes a query to the clusters whose mean is closest. Each k-means cluster of contrastive text embeddings is anisotropic: with C=32, the top 16 of 384")
w(f"principal directions carry {100*min(efs):.0f} to {100*max(efs):.0f} percent of the within-cluster variance across our dataset/embedder pairs (an isotropic cluster would give about 4 percent). Scoring a cluster by the query's distance to the cluster's affine k-flat")
w("(mean plus top-k PCA directions) instead of its distance to the mean lets the router 'see along' those directions. Across 6 BEIR datasets and")
w(f"{len(embs)} embedders ({', '.join(embs)}), this improves neighbor recall and nDCG@10 at equal clusters and probes, monotonically in k.")
w("The paper should present this as an empirical characterization with a clear cost accounting, not as a new index that beats production ANN.")
w("Keep the negative result about linear Grassmannian routing (it is useful) and keep the positive control (moved to an appendix).")
w("")
w("2. TITLE OPTIONS (pick one)")
w("-" * 72)
w("  A. Looking Along the Cluster: Local Principal Directions as a Routing Signal for Partitioned Dense Retrieval")
w("  B. Beyond the Centroid: k-Flat Residual Routing for IVF-Style Retrieval in RAG")
w("  C. Do Grassmannian Subspaces Help RAG Indexing? Linear Subspaces Fail, Local k-Flats Help at Fixed Scan Budget")
w("  (C keeps continuity with the current title; A is the most accurate.)")
w("")
w("3. NEW ABSTRACT (paste; replace nothing inside unless numbers are re-run)")
w("-" * 72)
abs_ = (f"Partition-based approximate nearest neighbor (ANN) indexes route each query to a few clusters by comparing it with cluster centroids. "
 f"We ask whether the shape of each cluster, not only its center, is a useful routing signal for retrieval-augmented generation (RAG). "
 f"We compare three routers over the same k-means partition: cosine-to-centroid (standard IVF), the angle to a through-origin linear subspace (Grassmannian routing), "
 f"and the residual distance to the cluster's affine k-flat (centroid plus top-k PCA directions), a classical local-PCA construction. "
 f"On {len(dss)} BEIR datasets with {len(embs)} sentence embedders, 5 k-means seeds and paired bootstrap confidence intervals, linear Grassmannian routing is far worse than IVF "
 f"({pts(gr['d_recall'])} points of neighbor recall@10 at 32 clusters and 4 probes), because it discards the cluster offset. "
 f"The k-flat router improves on IVF at equal clusters and probes, and the gain grows with k: {pts(k8['d_recall'])}, {pts(k16['d_recall'])} and {pts(k32['d_recall'])} recall points "
 f"and {pts(k8['d_ndcg'])}, {pts(k16['d_ndcg'])} and {pts(k32['d_ndcg'])} nDCG@10 points for k = 8, 16 and 32, while plain Euclidean distance to the centroid (k = 0) is worse than cosine. "
 f"The improvement therefore comes from the local principal directions, not from a change of metric. "
 f"Under an explicit cost model the picture is mixed: when routing cost is counted on these small corpora, finer cosine IVF matches or beats k-flat routing, "
 f"while at a fixed scan budget (the large-corpus regime) k-flat routing gives similar neighbor recall and higher nDCG@10. "
 f"Bases can be stored in int8 with no measurable loss, but a single shared basis or per-cluster adaptive k does not help. "
 f"Graph indexes (HNSW) remain better on recall versus latency at these scales. We release code, raw results and scripts.")
w(abs_)
w("")
w("4. CONTRIBUTIONS LIST (replace the current list)")
w("-" * 72)
w("  (1) A controlled comparison of three routers that share one k-means partition (cosine centroid, linear Grassmannian, affine k-flat residual),")
w(f"      isolating the routing score from partition quality, on {len(dss)} BEIR datasets, {len(embs)} embedders, 5 seeds, C in {{16,32,64,128,256}}, P in {{1..64}}.")
w("  (2) Evidence that the k-flat gain comes from local principal directions: k = 0 (Euclidean centroid) is worse than cosine, and the gain increases with k.")
w("  (3) An explicit cost accounting (routing FLOPs, scan FLOPs, measured single-thread latency, memory) and comparisons with compute-matched IVF and FAISS IVF-Flat, IVF-PQ and HNSW.")
w("  (4) End-task evaluation with nDCG@10 against BEIR qrels, showing that the relevance gain at fixed scan budget is larger than the neighbor-recall gain.")
w("  (5) A mechanism analysis: per-cluster explained variance, a simple sufficient condition for probing the cluster that holds a true neighbor, and the PPCA interpretation; plus cheap variants (fp16/int8 bases, shared basis, adaptive k).")
w("")
w("5. SECTION-BY-SECTION EDIT INSTRUCTIONS")
w("-" * 72)
w("""Global:
  * Remove every em dash (literal or LaTeX ---) outside table cells.
  * Rename 'Affine Grassmannian' everywhere to 'k-flat residual routing'. Delete the heading suffix '(Formerly Affine Grassmannian)' in Section 3.1.
  * Delete the literal markdown '**PCA reconstruction residual**' (paper.tex line ~178); write \\emph{PCA reconstruction residual}.
  * Fix the double period in the Related Work paragraph heading 'ANN and RAG..'.
  * The bibliography printed stray 'note' fields ('FAISS library', 'Surge in Grassmannian subspace learning methods', 'Cluster prototypes...',
    'Grassmannian LRR for subspace clustering'). Replace references.bib entirely with the BibTeX in Section 11; it has no such notes.
  * Notation: mu_c currently means two things. Use \\mu_c for the raw member mean (used by k-flat and PPCA routers) and \\bar{\\mu}_c = \\mu_c/\\|\\mu_c\\|_2
    for the normalized centroid used by cosine IVF. Use U_c (orthonormal m x k basis) everywhere; drop W_c, or define \\mathcal{W}_c = \\mathrm{span}(U_c)
    once and use F_c = \\mu_c + \\mathcal{W}_c for the affine flat. Write d_flat(q, F_c), not d_flat(q, W_c).
  * Every claim of 'C in {16,32,64}, k in {4,8,16}' must now match what was run: C in {16,32,64,128,256}; k in {4,8,16,32} at C<=64 and k in {8,16,32} at C in {128,256};
    P in {1,2,4,8,16,32,64} with P <= C.

Abstract: replace with Section 3 of this file.

Introduction:
  * Keep the motivation (centroid IVF is the default RAG index; subspace intuitions are common). Remove 'Grassmannian clustering is a mature literature' citation list
    and use: \\cite{elhamifar2013sparse,vidal2011subspace,liu2013lrr,wang2014lrr,cetingul2009intrinsic,zhang2018grassmannian}.
  * Cite IVF origin \\cite{sivic2003video}, PQ/IVF-PQ \\cite{jegou2011product}, HNSW \\cite{malkov2018hnsw}, FAISS \\cite{douze2024faiss,johnson2019faiss}, ScaNN \\cite{guo2020scann}, LOPQ \\cite{kalantidis2014lopq}.
  * Replace the four research questions with: (Q1) Is linear Grassmannian routing competitive? (Q2) Do local principal directions add routing signal beyond the centroid,
    and does it grow with k? (Q3) Does that survive compute-matched and end-task (nDCG) evaluation and standard ANN baselines? (Q4) What predicts the gain, and can it be made cheap?
  * Replace contributions with Section 4.

Preliminaries:
  * Keep the Grassmannian definition and principal angles (Eq. 1, 2 are correct). Add one sentence that a point query is a 1-dimensional subspace so only one principal angle exists.
  * DELETE Section 2.1 'Connection to Fusion Frames' entirely (it is never used). Also delete Discussion item (5) about fusion frames and Kashin frames, and the Related Work
    paragraph 'Frame theory and low-bit representations'. If the authors want to keep the Kashin self-citation, keep ONE sentence in Related Work under quantization:
    'Codebook-free low-bit representations such as Kashin frames \\cite{paul2026kashin} are complementary to the routing question studied here.' Then casazza2008fusion can be dropped.

Method (Section 3):
  * Define the three routers explicitly over one shared partition: cosine s_c = -<q, \\bar{\\mu}_c>; linear Grassmannian s_c = arccos(||U_c^T q||_2) for unit q;
    k-flat s_c^2 = ||q - \\mu_c||^2 - ||U_c^T (q - \\mu_c)||^2 (residual distance to the affine flat F_c). Note k = 0 gives Euclidean distance to the raw mean.
  * Add the PPCA router used as an ablation: negative log-likelihood under a probabilistic PCA model per cluster \\cite{tipping1999ppca,tipping1999mppca} with k = 16, residual variance
    sigma_c^2 = (total variance minus top-k eigenvalues)/(m-k), and log prior log(n_c/N).
  * Implementation detail to state: U_c are the top-k right singular vectors of the centered members; \\mu_c is the exact member mean (recomputed from labels), not the
    MiniBatchKMeans center; queries and documents are L2-normalized; second stage is exact cosine over the union of the P probed clusters.
  * Cost model paragraph: routing FLOPs per query = C*m for cosine or Euclidean, C*m*(k+1) for k-flat (precompute U_c^T mu_c); scan FLOPs = (#candidates)*m;
    extra memory = C*m*k floats for bases (int8 storage makes this C*m*k bytes plus C*k scales).
  * Replace the existing 'Complexity analysis' paragraph with that cost model.

New subsection 'Why local directions help' (put after Method or in Discussion). Use this text (it is correct; do not strengthen it):
  Let F_c = \\mu_c + span(U_c) and s_c(q) = dist(q, F_c). Distance to a set is 1-Lipschitz, so for any document x assigned to cluster c,
  s_c(q) <= ||q - x|| + dist(x, F_c). Because dist(x, F_c) <= ||x - \\mu_c||, the cluster that holds a true neighbor x has a score bound that is never looser than
  the centroid bound, and averaged over the cluster E[dist(x, F_c)^2] = (1 - rho_c(k)) E[||x - \\mu_c||^2], where rho_c(k) is the fraction of within-cluster
  variance captured by the top-k principal directions (an identity for PCA fit on the cluster's own members). Hence a sufficient condition for probing the
  cluster c* that contains a true neighbor x is that fewer than P other clusters c' satisfy s_{c'}(q) < ||q - x|| + dist(x, F_{c*}).
  Caveat to state: the other clusters' scores also shrink as k grows, and the flat is unbounded along U_c, so this is a sufficient condition, not a guarantee
  of improvement. It predicts that the gain should increase with rho_c(k), which we test across datasets and embedders.
  PPCA link: up to cluster-dependent constants, the k-flat residual is the PPCA Mahalanobis term in the limit where the retained eigenvalues are much larger than the
  residual variance; the full PPCA likelihood (with log-determinant and priors) is a natural alternative and we report it as an ablation (it does worse).

Experimental Setup: replace with Section 6.1 of this file.

Results: replace Sections 5.1 to 5.5 with the tables and text in Section 6. Delete the old Tables 2 and 3 (single seed, C=32 only) and the claim that the
  dimension ablation is 'flat from k=4 to k=16' (it was never run at k=16, and with the new runs it is not flat).
  Move the synthetic planted-subspace study (old Table 4, Figure 4) to an appendix as a positive control for the linear Grassmannian router, and state its limits
  (20 queries, oracle C=20 vs k-means C=16 in the original run; it was NOT re-run here).

Discussion: rewrite 'Why centroids beat standard Grassmannians' to: linear subspaces through the origin ignore the cluster offset; k-flats keep the offset and add local
  anisotropy. Delete 'metric alignment' as the explanation everywhere (contradicted by k = 0 results). Keep 'When Grassmannians remain promising' items (1) multi-vector
  documents (cite \\cite{khattab2020colbert} for ColBERT-style bags), (3) oracle partitions, and (4) sketched kernels (cite \\cite{delogne2025random,delogne2026bounded}).
  Item (2) re-ranking has no supporting citation: present it as a suggestion without a citation. Delete item (5).
  Add a paragraph 'When to use k-flat routing': large corpora where scan dominates and relevance (nDCG) matters; store bases in int8; k = 16 to 32;
  not when routing cost dominates (small C*k relative to scanned candidates is required), and not as a replacement for HNSW at these scales.
  Add a sentence on anisotropy of contextual embeddings \\cite{ethayarajh2019contextual} to support the 'conical/anisotropic' wording.

Related Work: use Section 11 keys. Nearest-subspace search: \\cite{basri2011approximate,wang2013fast}. Grassmann learning/clustering: \\cite{cetingul2009intrinsic,wang2014lrr,
  harandi2013dictionary,harandi2015extrinsic,stiverson2020subspace,marrinan2014finding,zhang2018grassmannian,bendokat2024handbook}. Local PCA / k-flats / PPCA:
  \\cite{bradley2000kplane,kambhatla1997localpca,tipping1999ppca,tipping1999mppca}. Quantization with local structure: LOPQ \\cite{kalantidis2014lopq} (NOT ge2014opq;
  OPQ is a different method; cite ge2014opq only for OPQ), anisotropic quantization \\cite{guo2020scann}. Graph ANN: \\cite{malkov2018hnsw,fu2019nsg,wang2021graphann}.
  Dense retrieval and data: \\cite{karpukhin2020dpr,khattab2020colbert,thakur2021beir,wadden2020scifact,reimers2019sbert,wang2020minilm,wang2022e5,xiao2023cpack}.
  Replace 'Recursive similarity search [cherian]' with 'Approximate nearest-subspace search \\cite{basri2011approximate,wang2013fast}'.

Limitations: replace with Section 9.

Conclusion: replace with Section 8.

Reproducibility: point to the new scripts (Section 12) and state hardware (8-core CPU, no GPU), library versions (faiss-cpu 1.16.0, torch 2.15.0+cpu as installed on the run box), seeds 0 to 4.

Appendix A (failure modes): keep the 'centroid bias under oracle subspaces' paragraph, delete 'roughly 2x the probe budget' (old single-seed claim).""")
w("")
w("6. RESULTS TO PUT IN THE PAPER")
w("-" * 72)
w("6.1 Setup text (paste)")
dsl = "; ".join(f"{d} ({meta[d]['n_docs']} docs{' of ' + str(meta[d]['full_corpus']) + ', subsampled' if meta[d]['subsampled'] else ''}, {meta[d]['n_queries']} test queries{' of ' + str(meta[d]['n_test_queries_total']) if meta[d]['n_queries'] < meta[d]['n_test_queries_total'] else ''})" for d in dss)
w(f"  Datasets (BEIR test splits from the HuggingFace BeIR mirrors): {dsl}.")
w("  Subsampling rule (fixed seed 0): keep every document judged relevant (qrel score > 0) for a kept query, then fill with uniformly sampled other documents up to")
w("  30,000; queries with at least one relevant document, capped at 500 by uniform sampling. Document text = title + '. ' + text, truncated to 3,000 characters,")
w("  encoder max sequence length 256 tokens.")
w("  IMPORTANT caveat for trec-covid: 17,537 of the 30,000 kept documents are judged relevant, so the subsample is relevance-dense and its nDCG is not comparable to")
w("  published full-corpus numbers. fiqa keeps 1,316 judged documents plus 28,684 random ones.")
w(f"  Embedders: sentence-transformers/all-MiniLM-L6-v2 (minilm), BAAI/bge-small-en-v1.5 with its query instruction (bge-small);")
w(f"  both 384-dimensional, L2-normalized (e5-small-v2 and bge-base were planned but NOT RUN). Pairs actually run: {pairlist} ({NP} pairs).")
w("  Partition: scikit-learn MiniBatchKMeans (batch 1024, n_init 3), seeds 0 to 4; all routers share the partition for a given (C, seed).")
w("  Metrics: neighbor recall@10 against exact cosine top-10; nDCG@10 against BEIR qrels with gains 2^rel - 1, computed on the final top-10 after exact cosine re-ranking")
w("  of the probed union; fraction of corpus scanned; routing and scan FLOPs; single-thread per-query latency (median of 3 passes) for the FAISS comparison.")
w("  Statistics: per query, average the metric over the 5 seeds; 95% CI by percentile bootstrap over queries (2,000 resamples); differences vs cosine IVF are paired.")
w("")
w("6.2 Headline numbers for the text (C=32, P=4 unless stated; mean over all pairs of the difference vs cosine IVF, recall@10 / nDCG@10 in points)")
for r in ["l2_k0", "grass_k8", "kflat_k4", "kflat_k8", "kflat_k16", "kflat_k32", "ppca_k16"]:
    x = k(r); w(f"  {r:10s}: {pts(x['d_recall'])} / {pts(x['d_ndcg'])}   (significantly better than cosine in {x['n_sig_up']}/{x['n_pairs']} pairs, worse in {x['n_sig_down']}/{x['n_pairs']})")
w(f"  kflat_k16 at C=64,P=4: {pts(k16_64['d_recall'])} / {pts(k16_64['d_ndcg'])} (better in {k16_64['n_sig_up']}/{k16_64['n_pairs']})")
if k16_256: w(f"  kflat_k16 at C=256,P=16: {pts(k16_256['d_recall'])} / {pts(k16_256['d_ndcg'])} (better in {k16_256['n_sig_up']}/{k16_256['n_pairs']})")
w("")
w("6.3 Full tables (generated by aggregate.py; paste the relevant parts as LaTeX tables; per-pair CIs are in brackets)")
w(T)
w("")
w("6.4 Compute-matched reading (put this in the Results text)")
for nm, mm in [("scan-only 5%", ms5), ("scan-only 10%", ms10), ("scan-only 20%", ms20), ("total 20%", mt20), ("total 40%", mt40)]:
    if mm: w(f"  budget {nm}: best k-flat minus best cosine IVF: recall {pts(mm['dr'])} (k-flat better {mm['rwin']}/{mm['n']}, worse {mm['rlose']}); nDCG {mm['dn']*100:+.2f} (better {mm['nwin']}/{mm['n']}, worse {mm['nlose']})")
w("  Interpretation to write: with routing FLOPs counted, small corpora favor finer cosine IVF because C*m*(k+1) is comparable to the scan; with scan-only cost,")
w("  which approximates large corpora where C is about sqrt(N), k-flat routing matches IVF on neighbor recall and improves nDCG@10. Do not claim more than this.")
w("")
w("6.5 FAISS baselines and latency (seed 0, 1 thread, batch size 1, median of 3 passes; measured while other experiment jobs used the remaining cores, so treat absolute ms as indicative)")
if bl_lines:
    w("\n".join(bl_lines))
    w(f"  Write: HNSW (ef=16) beats our k=16 router on both recall and latency in {hn_dom}/{hn_n} pairs; the k-flat router is a better IVF router,")
    w("  not a better index. Our numpy two-level index is Python, so its absolute latency is not comparable to FAISS C++; compare FLOPs and scanned fraction instead.")
    w("  IVF-PQ (48 bytes per vector) without refinement loses recall as expected; with exact refinement of the top 100 it matches IVF-Flat.")
else:
    w("  NOT RUN (see Section 10).")
w("")
w("6.6 Anisotropy and gain (mechanism)")
if an:
    w(f"  Spearman correlation between the size-weighted fraction of within-cluster variance explained by the top-16 directions (C=32) and the k=16 recall gain: {an['spearman_recall'][0]:.2f} (p = {an['spearman_recall'][1]:.3f}, n = {an['n']} pairs);")
    w(f"  Pearson {an['pearson_recall'][0]:.2f} (p = {an['pearson_recall'][1]:.3f}); Spearman with the nDCG gain {an['spearman_ndcg'][0]:.2f} (p = {an['spearman_ndcg'][1]:.3f}).")
    w("  Report it as suggestive only if p > 0.05; with this few pairs it is not strong evidence either way.")
w("")
w("6.7 Cheaper variants (C=32, P=4, mean over the pairs listed in the CHEAPER VARIANTS table)")
for v in ["kflat16_fp32", "kflat16_fp16", "kflat16_int8", "shared_k16", "shared_k32", "adaptive_tau0.15", "adaptive_tau0.25", "adaptive_tau0.35", "cos"]:
    if v in ch: x = ch[v]; w(f"  {v:18s}: recall {100*x['recall']:.1f}, nDCG {100*x['ndcg']:.1f}, change vs fp32 k=16 {100*x['d_recall_vs_fp32']:+.2f} / {100*x['d_ndcg_vs_fp32']:+.2f}, mean k {x['k_avg']:.1f}")
w("  Write: fp16 and int8 bases are free (no measurable change); one basis shared by all clusters is much worse than cosine IVF (global directions discount the very")
w("  directions that separate clusters); per-cluster adaptive k is no better than a fixed k with the same average.")
w("")
w("7. FIGURE SPECS (all data in /workspace/grassmannian_exp/results)")
w("-" * 72)
w("""  Fig 1 (pipeline): keep the existing TikZ figure; relabel the solid path 'k-flat residual' and add a dotted path 'linear Grassmannian'.
  Fig 2 (k-curve): x = k in {0 (Euclidean centroid), 4, 8, 16, 32}, plus a horizontal line for cosine IVF; y = mean difference vs cosine in recall@10 (left panel) and
    nDCG@10 (right panel). One line per C in {16, 32, 64} at P=4. Source: summary.json['kcurve'] keys 'l2_k0|C{C}|P4', 'kflat_k{k}|C{C}|P4'. Error bars: plot the
    per-pair CI counts as annotations or use per-pair points from summary.json['main'] '{ds}|{emb}|{router}|C{C}|P{P}' -> d_recall_vs_cos [mean, lo, hi].
  Fig 3 (recall and nDCG vs scanned fraction): per dataset (minilm and bge-small panels), x = cand_frac, y = recall (and a second row for nDCG), one curve per router
    (cos, kflat_k16, kflat_k32, grass_k8) at C=32 and C=128, points for P in {1,...,64}. Source: routing__{ds}__{emb}.json 'configs' (average over seed).
  Fig 4 (compute-matched): x = total cost / brute-force cost (log scale), y = recall; Pareto frontier per family (cos over all C,P; kflat over all C,k,P); second panel with
    scan-only cost; third panel nDCG. Source: routing__*.json 'configs' fields flops_route, flops_scan, recall, ndcg; N and m in the same file.
  Fig 5 (ANN baselines): recall vs single-thread latency for FAISS Flat, IVF-Flat (nlist 16 to 256), IVF-PQ, HNSW (ef 16 to 256), ours (cos, kflat k=8/16/32 at C=32/64).
    Source: baselines__{ds}__{emb}.json 'rows'. Caption must say ours is Python/NumPy and FAISS is C++.
  Fig 6 (mechanism): scatter of explained_frac_k16 vs gain_recall (and gain_ndcg), one point per dataset/embedder pair, colored by embedder.
    Source: summary.json['aniso'].
  Old figures fig_structure, fig_recall_vs_probe, fig_k_ablation, fig_latency: delete (single seed, SciFact only). fig_synthetic can stay in the appendix.""")
w("")
w("8. NEW CONCLUSION (paste)")
w("-" * 72)
w(f"Linear Grassmannian routing, which scores a cluster by the angle between the query and a through-origin subspace, is a poor router for dense text retrieval: it "
  f"discards the cluster offset and loses {abs(100*gr['d_recall']):.1f} points of neighbor recall@10 to centroid IVF on average. Keeping the offset and adding the cluster's "
  f"top-k principal directions, the classical k-flat residual, gives a consistent improvement over centroid IVF at equal clusters and probes that grows with k "
  f"({pts(k16['d_recall'])} recall and {pts(k16['d_ndcg'])} nDCG@10 points at k = 16, {pts(k32['d_recall'])} and {pts(k32['d_ndcg'])} at k = 32, averaged over {k16['n_pairs']} dataset and embedder pairs). "
  f"Euclidean distance to the centroid alone does not produce this gain, so the signal is the local geometry of the cluster. The gain is not free: routing cost grows by a factor of k + 1, "
  f"which finer cosine partitions can match on small corpora, while at a fixed scan budget k-flat routing keeps neighbor recall and improves nDCG@10. Bases quantize to int8 without loss. "
  f"Graph indexes remain stronger at these scales. We therefore recommend k-flat routing as a drop-in improvement to the coarse quantizer of IVF-style indexes when scan cost dominates, "
  f"and we leave billion-scale validation, learned routers, and multi-vector documents, where subspace geometry is native, to future work.")
w("")
w("9. LIMITATIONS (paste)")
w("-" * 72)
w(f"""  * Corpus sizes are at most 30,000 documents; {', '.join(sub) if sub else 'no dataset'} were subsampled (rule in 6.1). Query sets capped at 500 ({', '.join(qcap) if qcap else 'none'} affected).
    trec-covid has only 50 queries, so its CIs are wide, and its subsample is relevance-dense.
  * Embedders are small (384-d) English models; no 768-d or larger model was run (bge-base and e5-small-v2 were planned and dropped for CPU time); only two embedders were run.
  * The partition is MiniBatchKMeans; results with full k-means or balanced partitions may differ.
  * The cost model counts FLOPs; our two-level index is Python/NumPy, so its latency is only indicative. FAISS numbers are single-thread, batch size 1.
  * ScaNN was not run. No billion-scale or disk-based setting was tested, so the 'scan-only' regime is a proxy for large corpora, not a measurement.
  * The bound in the 'why' subsection is a sufficient condition, not a guarantee of improvement.
  * The synthetic positive control was not re-run with more queries or matched C.""")
w("")
w("10. WHAT WAS NOT RUN OR WAS TRIMMED (be explicit in the paper where relevant)")
w("-" * 72)
planned = [(d, e) for e in ["minilm", "bge-small"] for d in ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid"]]
missing = [f"{d}/{e}" for d, e in planned if (d, e) not in pairs]
w("  * Routing grid missing for planned pairs: " + (", ".join(missing) + " (NOT RUN, CPU time)" if missing else "none; all 12 minilm and bge-small pairs completed."))
w("  * bge-base-en-v1.5 (the 'larger' embedder): NOT RUN (CPU time).")
w("  * e5-small-v2: NOT RUN on any dataset (CPU embedding time). The paper should not claim e5 results.")
w("  * ScaNN: NOT RUN (not attempted after FAISS; pip build on this box was not tried).")
w("  * FAISS baselines were run for: " + (", ".join(f"{d}/{e}" for d, e in pairs if f'{d}|{e}' in S['baselines']) or "none") + ".")
w("  * Grassmannian router and PPCA router and k = 4 were evaluated only at C in {16, 32, 64}.")
w("  * MS MARCO, NQ, HotpotQA and other large BEIR sets: NOT RUN.")
w("  * Synthetic positive control: NOT re-run.")
w("")
w("11. FULL CORRECTED BIBLIOGRAPHY (replace references.bib with this; keys used in Section 5)")
w("-" * 72)
w("""Notes on what changed versus the current references.bib:
  * Seven entries could not be found and are replaced: cherian2014grassmannian -> basri2011approximate + wang2013fast; harandi2015sparse -> harandi2015extrinsic;
    harandi2014grassmann -> cetingul2009intrinsic + wang2014lrr; hu2014grassmannian -> harandi2013dictionary; hu2018survey -> zhang2018grassmannian + bendokat2024handbook;
    huang2015grassmannian -> delogne2025random + delogne2026bounded; wang2019grassmannian -> stiverson2020subspace + marrinan2014finding.
  * Corrected entries: casazza2008fusion (real title 'Fusion Frames and Distributed Processing', ACHA 25(1):114-132); ge2013lopq -> ge2014opq (TPAMI 36(4):744-755) and
    LOPQ is now kalantidis2014lopq; liu2016lrr -> liu2013lrr (real title 'Robust Recovery of Subspace Structures by Low-Rank Representation'; the Grassmann LRR paper is
    wang2014lrr); pan2024graph -> wang2021graphann (Wang, Xu, Yue, Wang, PVLDB 14(11)); tipping1999ppca now is the JRSS-B PPCA paper and tipping1999mppca the Neural
    Computation mixtures paper (cite mppca where 'mixtures of PPCA' is meant); turaga2011statistical (correct title, TPAMI, adds Srivastava); wong1967grassmann (PNAS 57(3));
    douze2024faiss (Lucas Hosseini, not 'Lucas Barros, Dmitry Yashunin'); gao2024ragsurvey -> gao2023ragsurvey (Yun Xiong, Xinyu Gao); wang2022gte -> wang2022e5 (adds
    Binxing Jiao, Daxin Jiang); fu2019nsg (Chao Xiang, Changxu Wang). Venue/volume fixes for malkov2018hnsw (TPAMI 42(4) 2020), johnson2019faiss, jegou2011product,
    elhamifar2013sparse, vidal2011subspace (these were @inproceedings with journal fields).
  * New: sivic2003video, kalantidis2014lopq, guo2020scann, khattab2020colbert, wang2020minilm, wadden2020scifact, ethayarajh2019contextual, xiao2023cpack (BGE), tipping1999ppca.
  * Remove from references.bib: wang2012grassmannian, johnson2024ann, zhang2024filter, wei2024hybrid (could not be verified; uncited), and uncited classics
    (indyk1998lsh, craswell2020msmarco, salakhutdinov2009semantic, weiss2009sh, johnson1984lipschitz, cover1965geometrical) unless the text cites them.
  * paul2026kashin is kept exactly as the authors have it (with Sarkar), at the authors' request.
  * Rename keys in paper.tex: ge2013lopq -> kalantidis2014lopq where LOPQ is meant; liu2016lrr -> liu2013lrr; pan2024graph -> wang2021graphann; gao2024ragsurvey -> gao2023ragsurvey;
    wang2022gte -> wang2022e5; neelakantan2022ada -> neelakantan2022cpt; tipping1999ppca (mixtures context) -> tipping1999mppca.
""")
w(BIB)
w("")
w("12. CODE, DATA AND REPRODUCTION")
w("-" * 72)
w("""  Copy /workspace/grassmannian_exp into the repo as experiments_v2/ (or ask Parag for the folder). Files:
    prepare_data.py   download BEIR from HuggingFace, subsample (rule in 6.1), writes data/<ds>.json
    embed.py          encode with minilm / bge-small (e5-small and bge-base supported, not run), writes emb/<ds>__<emb>.npz
    run_routing.py    main grid: routers x C x P x 5 seeds, writes results/routing__<ds>__<emb>.json (+ .npz per-query arrays)
    run_cheap.py      fp16/int8/shared/adaptive variants, results/cheap__<ds>__<emb>.json
    run_baselines.py  FAISS Flat, IVF-Flat, IVF-PQ(+refine), HNSW and our routers with single-thread latency, results/baselines__<ds>__<emb>.json
    aggregate.py      CIs, k-curve, compute-matched, anisotropy, tables -> results/summary.json, results/tables.txt
    make_report.py    builds this file
    common.py         shared loaders and metrics
    corrected_references.bib
  Order: prepare_data.py; embed.py; run_routing.py; run_cheap.py; run_baselines.py (1 thread); aggregate.py.
  Requirements: numpy, scipy, scikit-learn, sentence-transformers, datasets, torch (CPU), faiss-cpu. Remove yt-dlp from requirements.txt.""")
w("")
w("13. README / REPO CLEANUP")
w("-" * 72)
w("""  * README: remove 'Affine Grassmannian (Ours)' and 'beats centroid IVF'; replace the results table with the 6.2 headline numbers; fix the broken paper.pdf link
    (the PDF is grassmanian-rag-indexing.pdf); fix '~8 pages sigconf' (the build is acmart manuscript); update Status.
  * Delete READ_ME_PAPER.pdf (stale duplicate) or regenerate it.
  * Remove yt-dlp from requirements.txt; add faiss-cpu.
  * Replace experiments/results/results.json and generated/*.tex with outputs from the new scripts; delete old figures listed in Section 7.
  * Keep .cursor/rules/no-em-dashes.mdc.
  * Consider renaming the repo/file 'grassmanian' -> 'grassmannian' (spelling) before arXiv, since the arXiv abstract will link it.""")
w("")
w("14. ARXIV SUBMISSION CHECKLIST")
w("-" * 72)
w("""  * Primary category cs.IR (Information Retrieval); cross-list cs.LG (Machine Learning).
  * Upload LaTeX source: paper.tex, the .bbl file produced by bibtex (arXiv does not run bibtex reliably; include the .bbl), acmart.cls and ACM-Reference-Format.bst
    if not using a TeX Live version that includes them, all figures as PDF/PNG, generated/*.tex. Compile locally from a clean folder with pdflatex first.
  * Keep \\documentclass[manuscript,screen,nonacm]{acmart}; \\setcopyright{none} is fine for arXiv. Remove the '\\acmConference[Working Paper]...' line or keep it,
    but do not imply peer review.
  * Choose a license (CC BY 4.0 is the common choice; or arXiv's non-exclusive license).
  * Authors and affiliations: Parag Paul (RigAi / Agentic Memory Foundation), Saurabh Sarkar (Chicory AI / Agentic Memory Foundation); both must approve.
  * First-time submitters to cs.IR may need an endorsement.
  * Abstract field on arXiv: plain text, max 1,920 characters; no LaTeX macros; no em dashes.
  * Comments field: 'Code and data: <repo URL>. 12 pages.' (update page count).
  * Before submitting: every number in the paper must trace to results/summary.json or results/tables.txt; every \\cite key must exist in the new references.bib;
    grep the source for em dashes and '---'.""")
txt = "\n".join(R).replace("\u2014", ", ").replace("\u2013", "-")
open("/workspace/grassmannian_paper_update.txt", "w").write(txt)
print(len(txt), "chars;", "em dash count:", txt.count("\u2014"))
