"""FAISS baselines (IVF-Flat, IVF-PQ, HNSW) and single-thread per-query latency of our numpy routers.
Usage: python run_baselines.py <dataset> <embedder>
All latency numbers: 1 thread, per-query (batch size 1), median over 3 repeats of the full query set."""
import sys, json, time, numpy as np, faiss
from sklearn.cluster import MiniBatchKMeans
from common import BASE, load, gains_matrix, ndcg_at10
faiss.omp_set_num_threads(1)
ds, emb = sys.argv[1], sys.argv[2]
d, X, Q = load(ds, emb); N, m = X.shape; nq = len(Q)
gains = gains_matrix(d)
S = Q @ X.T
truth = np.argpartition(-S, 9, axis=1)[:, :10]
def rec_of(I): return float(np.mean([len(set(I[i].tolist()) & set(truth[i].tolist())) / 10 for i in range(nq)]))
def timed(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for i in range(nq): fn(i)
        ts.append((time.perf_counter() - t) / nq * 1e3)
    return float(np.median(ts))
rows = []
def add(kind, params, I, ms, mem_bytes):
    nd = ndcg_at10(I, gains)
    rows.append(dict(kind=kind, **params, recall=rec_of(I), ndcg=float(nd.mean()), ms=ms, mem_bytes=mem_bytes))
fl = faiss.IndexFlatIP(m); fl.add(X)
_, I = fl.search(Q, 10)
add("faiss_flat", {}, I, timed(lambda i: fl.search(Q[i:i+1], 10)), X.nbytes)
for nlist in [16, 32, 64, 128, 256]:
    for seed in range(3):
        qz = faiss.IndexFlatIP(m)
        ix = faiss.IndexIVFFlat(qz, m, nlist, faiss.METRIC_INNER_PRODUCT)
        ix.cp.seed = seed; ix.train(X); ix.add(X)
        for nprobe in [1, 2, 4, 8, 16]:
            if nprobe > nlist: continue
            ix.nprobe = nprobe
            _, I = ix.search(Q, 10)
            ms = timed(lambda i: ix.search(Q[i:i+1], 10)) if seed == 0 else None
            add("faiss_ivfflat", dict(nlist=nlist, nprobe=nprobe, seed=seed), I, ms, X.nbytes + nlist * m * 4)
for nlist in [32, 64]:
    qz = faiss.IndexFlatIP(m)
    ix = faiss.IndexIVFPQ(qz, m, nlist, 48, 8, faiss.METRIC_INNER_PRODUCT)
    ix.cp.seed = 0; ix.train(X); ix.add(X)
    rr = faiss.IndexRefineFlat(ix, faiss.swig_ptr(X)); rr.k_factor = 10
    for nprobe in [1, 2, 4, 8, 16]:
        ix.nprobe = nprobe
        _, I = ix.search(Q, 10)
        add("faiss_ivfpq", dict(nlist=nlist, nprobe=nprobe, refine=False), I,
            timed(lambda i: ix.search(Q[i:i+1], 10)), N * 48 + nlist * m * 4)
        _, I = rr.search(Q, 10)
        add("faiss_ivfpq", dict(nlist=nlist, nprobe=nprobe, refine=True), I,
            timed(lambda i: rr.search(Q[i:i+1], 10)), N * 48 + X.nbytes)
hn = faiss.IndexHNSWFlat(m, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 200; hn.add(X)
for ef in [16, 32, 64, 128, 256]:
    hn.hnsw.efSearch = ef
    _, I = hn.search(Q, 10)
    add("faiss_hnsw", dict(M=32, efSearch=ef), I, timed(lambda i: hn.search(Q[i:i+1], 10)), X.nbytes + N * 64 * 4)
for C in [32, 64]:
    km = MiniBatchKMeans(n_clusters=C, random_state=0, batch_size=1024, n_init=3).fit(X)
    lab = km.labels_
    order = np.argsort(lab, kind="stable"); Xs = X[order]
    bounds = np.searchsorted(lab[order], np.arange(C + 1))
    mu = np.stack([X[lab == c].mean(0) for c in range(C)]).astype(np.float32)
    mun = mu / np.linalg.norm(mu, axis=1, keepdims=True)
    mu2 = (mu ** 2).sum(1)
    U = np.zeros((C, m, 32), np.float32)
    for c in range(C):
        _, _, vt = np.linalg.svd(X[lab == c] - mu[c], full_matrices=False); kk = min(32, len(vt)); U[c, :, :kk] = vt[:kk].T
    def search(i, scores_fn, P):
        q = Q[i]; s = scores_fn(q)
        pr = np.argpartition(s, P - 1)[:P]
        idx = np.concatenate([order[bounds[c]:bounds[c + 1]] for c in pr])
        sims = np.concatenate([Xs[bounds[c]:bounds[c + 1]] @ q for c in pr])
        t = np.argpartition(-sims, 9)[:10]; return idx[t[np.argsort(-sims[t])]]
    fns = {"cos": (lambda q: -(mun @ q), 0)}
    for k in [4, 8, 16, 32]:
        Uk = np.ascontiguousarray(U[:, :, :k].transpose(0, 2, 1).reshape(C * k, m))
        Bk = np.einsum("ckm,cm->ck", Uk.reshape(C, k, m), mu)
        def f(q, Uk=Uk, Bk=Bk, k=k):
            l2 = 1.0 - 2.0 * (mu @ q) + mu2
            z = (Uk @ q).reshape(C, k) - Bk
            return l2 - (z * z).sum(1)
        fns[f"kflat_k{k}"] = (f, k)
    for name, (fn, k) in fns.items():
        for P in [2, 4, 8]:
            I = np.stack([search(i, fn, P) for i in range(nq)])
            add("numpy_twolevel", dict(router=name, C=C, P=P), I, timed(lambda i: search(i, fn, P)),
                X.nbytes + C * m * 4 * (1 + k))
            rows[-1]["route_only_ms"] = timed(lambda i: fn(Q[i]))
json.dump(dict(ds=ds, emb=emb, N=N, nq=nq, rows=rows), open(f"{BASE}/results/baselines__{ds}__{emb}.json", "w"))
print("done", ds, emb)
