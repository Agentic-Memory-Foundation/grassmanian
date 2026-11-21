"""Main routing experiment. Usage: python run_routing.py <dataset> <embedder>
Two-level index: MiniBatch k-means partition (same partition shared by all routers for a given C, seed),
then route by one of several scores, probe P clusters, exact cosine re-rank inside the probed union.
Writes results/routing__<ds>__<emb>.npz (per-query arrays) and .json (config list, cluster stats)."""
import os, sys, json, time, numpy as np
from sklearn.cluster import MiniBatchKMeans
from common import BASE, load, gains_matrix, ndcg_at10, topk_masked
ds, emb = sys.argv[1], sys.argv[2]
d, X, Q = load(ds, emb)
N, m = X.shape; nq = Q.shape[0]
gains = gains_matrix(d)
S = Q @ X.T
truth = np.argpartition(-S, 9, axis=1)[:, :10]
exact_top = topk_masked(S, np.ones_like(S, dtype=bool))
exact_ndcg = ndcg_at10(exact_top, gains)
KS = [4, 8, 16, 32]; PS = [1, 2, 4, 8, 16, 32, 64]
SEEDS = range(int(os.environ.get("N_SEEDS", "5")))
OUTDIR = os.environ.get("ROUTING_OUT", f"{BASE}/results")
os.makedirs(OUTDIR, exist_ok=True)
CS_SUB = [16, 32, 64, 128, 256]; CS_IVF = [16, 32, 64, 128, 256]
FULL = [16, 32, 64]  # grass_k8, ppca_k16, kflat_k4 only at these C (cost); kflat k in {8,16,32} at all C
arrays = {"exact_ndcg": exact_ndcg.astype(np.float32)}
configs = []; cluster_stats = []
for C in CS_IVF:
    for seed in SEEDS:
        km = MiniBatchKMeans(n_clusters=C, random_state=seed, batch_size=1024, n_init=3).fit(X)
        lab = km.labels_
        # exact member means (centering for PCA); empty clusters get the km center
        mu = np.stack([X[lab == c].mean(0) if np.any(lab == c) else km.cluster_centers_[c] for c in range(C)])
        mun = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12)
        sizes = np.bincount(lab, minlength=C)
        routers = {"cos": -(Q @ mun.T), "l2_k0": ((Q[:, None, :] - mu[None]) ** 2).sum(-1)}
        flops = {"cos": C * m, "l2_k0": C * m}
        mem = {"cos": C * m, "l2_k0": C * m}
        if C in CS_SUB:
            U = np.zeros((C, m, 32)); lam = np.zeros((C, 32)); tot = np.zeros(C)
            for c in range(C):
                Xc = X[lab == c] - mu[c]
                if len(Xc) < 2: continue
                _, s, vt = np.linalg.svd(Xc, full_matrices=False)
                kk = min(32, len(s))
                U[c, :, :kk] = vt[:kk].T
                ev = s ** 2 / (len(Xc) - 1)
                lam[c, :kk] = ev[:kk]; tot[c] = ev.sum()
            for k in KS:
                cluster_stats.append(dict(C=C, seed=seed, k=k,
                    explained_frac_mean=float(np.mean(lam[:, :k].sum(1) / np.maximum(tot, 1e-12))),
                    explained_frac_wmean=float(np.sum(sizes * lam[:, :k].sum(1) / np.maximum(tot, 1e-12)) / sizes.sum())))
            Dq = (Q[:, None, :] - mu[None]).astype(np.float32)   # nq,C,m
            l2 = (Dq ** 2).sum(-1)
            Z = np.einsum("qcm,cmk->qck", Dq, U.astype(np.float32))                 # nq,C,32
            for k in (KS if C in FULL else [8, 16, 32]):
                routers[f"kflat_k{k}"] = l2 - (Z[:, :, :k] ** 2).sum(-1)
                flops[f"kflat_k{k}"] = C * m * (k + 1); mem[f"kflat_k{k}"] = C * m * (k + 1)
            if C in FULL:
              # linear Grassmannian (paper baseline): angle between q and span(U_c), k=8
              proj = np.einsum("qm,cmk->qck", Q, U[:, :, :8])
              routers["grass_k8"] = np.arccos(np.clip(np.linalg.norm(proj, axis=-1), 0, 1))
              flops["grass_k8"] = C * m * 8; mem["grass_k8"] = C * m * 8
              # PPCA (Tipping-Bishop) negative log-likelihood, k=16, with log prior n_c/N
              k = 16
              l = np.maximum(lam[:, :k], 1e-8)
              sig2 = np.maximum((tot - l.sum(1)) / (m - k), 1e-8)
              z2 = Z[:, :, :k] ** 2
              nll = 0.5 * (np.log(l).sum(1) + (m - k) * np.log(sig2))[None] \
                  + 0.5 * ((z2 / l[None]).sum(-1) + (l2 - z2.sum(-1)) / sig2[None]) \
                  - np.log(np.maximum(sizes, 1) / N)[None]
              routers["ppca_k16"] = nll
              flops["ppca_k16"] = C * m * 17; mem["ppca_k16"] = C * m * 17
        for name, sc in routers.items():
            order = np.argsort(sc, axis=1)
            for P in PS:
                if P > C: continue
                probed = np.zeros((nq, C), bool)
                np.put_along_axis(probed, order[:, :P], True, 1)
                mask = probed[:, lab]
                rec = np.take_along_axis(mask, truth, 1).mean(1)
                ncand = mask.sum(1)
                top = topk_masked(S, mask)
                nd = ndcg_at10(top, gains)
                key = f"{name}|C{C}|P{P}|s{seed}"
                arrays["rec|" + key] = rec.astype(np.float32)
                arrays["ndcg|" + key] = nd.astype(np.float32)
                configs.append(dict(router=name, C=C, P=P, seed=seed, recall=float(rec.mean()),
                    ndcg=float(nd.mean()), cand_frac=float(ncand.mean() / N),
                    flops_route=flops[name], flops_scan=float(ncand.mean() * m),
                    mem_floats=mem[name]))
        print(ds, emb, C, seed, flush=True)
np.savez_compressed(f"{OUTDIR}/routing__{ds}__{emb}.npz", **arrays)
json.dump(dict(ds=ds, emb=emb, N=N, m=m, nq=nq, exact_ndcg=float(exact_ndcg.mean()),
               n_seeds=len(list(SEEDS)),
               configs=configs, cluster_stats=cluster_stats),
          open(f"{OUTDIR}/routing__{ds}__{emb}.json", "w"))
print("done", ds, emb)
