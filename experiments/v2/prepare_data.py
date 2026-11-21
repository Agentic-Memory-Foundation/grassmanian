"""Download BEIR datasets from HuggingFace, subsample deterministically, save text + qrels.
Rules: keep every doc judged relevant (score>0) for kept test queries; fill with random
non-judged docs up to MAX_DOCS (seed 0). Queries: test queries with >=1 relevant doc, capped at MAX_Q (seed 0)."""
import json, os, random, sys
from pathlib import Path
from datasets import load_dataset
MAX_DOCS = 30000
MAX_Q = 500
OUT = str(Path(__file__).resolve().parent / "data")
os.makedirs(OUT, exist_ok=True)
def prep(name):
    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qr = load_dataset(f"BeIR/{name}-qrels", split="test")
    qrels = {}
    for r in qr:
        s = int(r["score"])
        if s > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = s
    qtext = {str(r["_id"]): (r["text"] or "").strip() for r in queries}
    qids = sorted(q for q in qrels if qtext.get(q))
    rng = random.Random(0)
    if len(qids) > MAX_Q:
        qids = sorted(rng.sample(qids, MAX_Q))
    needed = set(d for q in qids for d in qrels[q])
    docs = {}
    for r in corpus:
        body = ((r["title"] or "") + ". " + (r["text"] or "")).strip(". ").strip()
        if body:
            docs[str(r["_id"])] = body[:3000]
    full_n = len(docs)
    needed &= set(docs)
    others = sorted(set(docs) - needed)
    fill = max(0, MAX_DOCS - len(needed))
    keep = sorted(needed) + (sorted(rng.sample(others, fill)) if fill < len(others) else others)
    keep_set = set(keep)
    qrels_k = {q: {d: s for d, s in qrels[q].items() if d in keep_set} for q in qids}
    qids = [q for q in qids if qrels_k[q]]
    meta = dict(name=name, full_corpus=full_n, n_docs=len(keep), n_queries=len(qids),
                n_test_queries_total=len(qrels), subsampled=len(keep) < full_n)
    json.dump(dict(meta=meta, doc_ids=keep, docs=[docs[d] for d in keep], query_ids=qids,
                   queries=[qtext[q] for q in qids], qrels={q: qrels_k[q] for q in qids}),
              open(f"{OUT}/{name}.json", "w"))
    print(meta, flush=True)
for n in sys.argv[1:]:
    try: prep(n)
    except Exception as e: print("FAILED", n, repr(e), flush=True)
