import json, numpy as np
from pathlib import Path
BASE = str(Path(__file__).resolve().parent)
def load(ds, emb):
    d = json.load(open(f"{BASE}/data/{ds}.json"))
    z = np.load(f"{BASE}/emb/{ds}__{emb}.npz")
    return d, z["X"], z["Q"]
def gains_matrix(d):
    """Return dict qidx -> {docidx: gain}."""
    idx = {x: i for i, x in enumerate(d["doc_ids"])}
    return [{idx[k]: v for k, v in d["qrels"][q].items()} for q in d["query_ids"]]
def ndcg_at10(top, gains):
    """top: (nq,10) doc indices ranked; gains: list of dict."""
    out = np.zeros(len(gains))
    disc = 1.0 / np.log2(np.arange(2, 12))
    for i, g in enumerate(gains):
        dcg = sum((2 ** g.get(int(t), 0) - 1) * disc[r] for r, t in enumerate(top[i]) if t >= 0)
        ideal = sorted(g.values(), reverse=True)[:10]
        idcg = sum((2 ** v - 1) * disc[r] for r, v in enumerate(ideal))
        out[i] = dcg / idcg if idcg > 0 else 0.0
    return out
def topk_masked(S, mask, k=10):
    Sm = np.where(mask, S, -np.inf)
    part = np.argpartition(-Sm, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(Sm, part, 1)
    o = np.argsort(-vals, axis=1)
    top = np.take_along_axis(part, o, 1)
    top[np.take_along_axis(vals, o, 1) == -np.inf] = -1
    return top
def bootstrap_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
