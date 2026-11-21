"""Embed prepared datasets. Usage: python embed.py <model_key> <dataset> [<dataset> ...]
Outputs emb/<dataset>__<model_key>.npz with L2-normalized float32 X (docs), Q (queries)."""
import json, sys, time, os, numpy as np, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
torch.set_num_threads(min(4, os.cpu_count() or 1))
ROOT = Path(__file__).resolve().parent
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-small": ("intfloat/e5-small-v2", "query: ", "passage: "),
    "bge-base": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
}
(ROOT / "emb").mkdir(parents=True, exist_ok=True)
key = sys.argv[1]; name, qp, dp = MODELS[key]
m = SentenceTransformer(name, device="cpu"); m.max_seq_length = 256
for ds in sys.argv[2:]:
    out = str(ROOT / "emb" / f"{ds}__{key}.npz")
    if os.path.exists(out): print("skip", out); continue
    d = json.load(open(ROOT / "data" / f"{ds}.json"))
    t = time.time()
    # sort by length for speed, then unsort
    docs = [dp + x for x in d["docs"]]
    order = np.argsort([len(x) for x in docs])
    Xs = m.encode([docs[i] for i in order], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    X = np.empty_like(Xs); X[order] = Xs
    Q = m.encode([qp + x for x in d["queries"]], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    np.savez(out, X=X.astype(np.float32), Q=Q.astype(np.float32))
    print(ds, key, X.shape, Q.shape, f"{time.time()-t:.0f}s", flush=True)
