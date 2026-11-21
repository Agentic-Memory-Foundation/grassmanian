"""Cheaper k-flat variants. Usage: python run_cheap.py <dataset> <embedder>
Variants of the k-flat residual router (k=16 unless noted): fp32 bases, fp16 bases, int8 bases
(symmetric per-basis-vector scale), one basis shared by all clusters (global PCA of within-cluster
residuals, k=16 and 32), and per-cluster adaptive k (smallest k whose top-k PCA directions explain
>= tau of the cluster's variance, capped at 32)."""
import sys, json, numpy as np
from sklearn.cluster import MiniBatchKMeans
from common import BASE, load, gains_matrix, ndcg_at10, topk_masked
ds, emb = sys.argv[1], sys.argv[2]
d, X, Q = load(ds, emb); N, m = X.shape; nq = len(Q)
gains = gains_matrix(d); S = Q @ X.T
truth = np.argpartition(-S, 9, axis=1)[:, :10]
rows = []
def evaluate(sc, lab, C, P):
    order = np.argsort(sc, 1); probed = np.zeros((nq, C), bool)
    np.put_along_axis(probed, order[:, :P], True, 1); mask = probed[:, lab]
    return np.take_along_axis(mask, truth, 1).mean(1), ndcg_at10(topk_masked(S, mask), gains)
def kflat(U, mu, kvec=None):
    D = Q[:, None, :] - mu[None]; l2 = (D ** 2).sum(-1)
    Z = np.einsum("qcm,cmk->qck", D, U)
    if kvec is not None:
        Z = Z * (np.arange(U.shape[2])[None, None, :] < kvec[None, :, None])
    return l2 - (Z ** 2).sum(-1)
for C in [32, 64]:
    for seed in range(5):
        km = MiniBatchKMeans(n_clusters=C, random_state=seed, batch_size=1024, n_init=3).fit(X)
        lab = km.labels_
        mu = np.stack([X[lab == c].mean(0) for c in range(C)])
        U = np.zeros((C, m, 32)); fr = np.zeros((C, 32))
        for c in range(C):
            Xc = X[lab == c] - mu[c]; _, s, vt = np.linalg.svd(Xc, full_matrices=False)
            kk = min(32, len(s)); U[c, :, :kk] = vt[:kk].T
            fr[c, :kk] = np.cumsum(s[:kk] ** 2) / (s ** 2).sum()
        R = X - mu[lab]; _, _, gv = np.linalg.svd(R, full_matrices=False)
        U16 = U[:, :, :16]
        sc_ = np.max(np.abs(U16), axis=1, keepdims=True) / 127.0
        variants = {
            "cos": (-(Q @ (mu / np.linalg.norm(mu, axis=1, keepdims=True)).T), 0, C * m * 4),
            "kflat16_fp32": (kflat(U16, mu), 16, C * m * 4 * 17),
            "kflat16_fp16": (kflat(U16.astype(np.float16).astype(np.float64), mu.astype(np.float16).astype(np.float64)), 16, C * m * 2 * 17),
            "kflat16_int8": (kflat(np.round(U16 / sc_) * sc_, mu), 16, C * m * 4 + C * m * 16 + C * 16 * 4),
            "shared_k16": (kflat(np.repeat(gv[:16].T[None], C, 0), mu), 16, C * m * 4 + m * 16 * 4),
            "shared_k32": (kflat(np.repeat(gv[:32].T[None], C, 0), mu), 32, C * m * 4 + m * 32 * 4),
        }
        for tau in [0.15, 0.25, 0.35]:
            kv = np.array([min(32, int(np.searchsorted(fr[c], tau) + 1)) for c in range(C)])
            variants[f"adaptive_tau{tau}"] = (kflat(U, mu, kv), float(kv.mean()), C * m * 4 * (1 + float(kv.mean())))
        for name, (sc, kavg, mem) in variants.items():
            for P in [2, 4, 8]:
                r, nd = evaluate(sc, lab, C, P)
                rows.append(dict(variant=name, C=C, seed=seed, P=P, k_avg=kavg, router_mem_bytes=mem,
                                 recall=float(r.mean()), ndcg=float(nd.mean())))
        print(ds, emb, C, seed, flush=True)
json.dump(dict(ds=ds, emb=emb, rows=rows), open(f"{BASE}/results/cheap__{ds}__{emb}.json", "w"))
print("done")
