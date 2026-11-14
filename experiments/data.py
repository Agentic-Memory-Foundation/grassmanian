"""Load small BEIR-style corpora for Grassmannian indexing experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / n).astype(np.float32)


def embed_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    return _normalize(np.asarray(emb, dtype=np.float32))


def load_scifact(max_docs: int = 6000, max_queries: int = 400):
    """Load SciFact corpus/queries/qrels via HuggingFace datasets.

    Preferentially keeps qrels-covered documents so relevance metrics are defined.
    """
    from datasets import load_dataset

    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/scifact-qrels", split="test")

    qrels_raw: dict[str, set[str]] = {}
    needed_docs: set[str] = set()
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        if int(row.get("score", 1)) <= 0:
            continue
        qrels_raw.setdefault(qid, set()).add(did)
        needed_docs.add(did)

    # Map all corpus rows; put qrels docs first so truncation cannot drop them.
    rows = []
    for row in corpus_ds:
        title = row.get("title") or ""
        text = row.get("text") or ""
        body = (title + ". " + text).strip()
        if not body:
            continue
        rows.append((str(row["_id"]), body[:2000]))
    rows.sort(key=lambda t: (0 if t[0] in needed_docs else 1, t[0]))
    rows = rows[:max_docs]

    doc_ids = [r[0] for r in rows]
    docs = [r[1] for r in rows]
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    kept_docs = set(doc_ids)

    qtext = {str(row["_id"]): (row.get("text") or "").strip() for row in queries_ds}
    queries = []
    query_ids = []
    qrels: dict[str, set[str]] = {}
    for qid, rels in qrels_raw.items():
        rels_kept = {d for d in rels if d in kept_docs}
        if not rels_kept:
            continue
        text = qtext.get(qid, "").strip()
        if not text:
            continue
        query_ids.append(qid)
        queries.append(text)
        qrels[qid] = rels_kept
        if len(queries) >= max_queries:
            break

    if not queries:
        raise RuntimeError("SciFact produced zero usable queries after qrels filtering")

    return {
        "name": "scifact",
        "docs": docs,
        "doc_ids": doc_ids,
        "queries": queries,
        "query_ids": query_ids,
        "qrels": qrels,
        "id_to_idx": id_to_idx,
    }


def load_or_build_embeddings(bundle: dict, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    key = f"{bundle['name']}_{len(bundle['docs'])}_{len(bundle['queries'])}_minilm"
    path = CACHE / f"{key}.npz"
    meta_path = CACHE / f"{key}.json"
    if path.exists():
        data = np.load(path)
        return data["doc_emb"], data["query_emb"]

    print(f"Embedding {len(bundle['docs'])} docs + {len(bundle['queries'])} queries…")
    doc_emb = embed_texts(bundle["docs"], model_name=model_name)
    query_emb = embed_texts(bundle["queries"], model_name=model_name)
    np.savez_compressed(path, doc_emb=doc_emb, query_emb=query_emb)
    meta_path.write_text(json.dumps({"model": model_name, "n_docs": len(bundle["docs"]), "n_queries": len(bundle["queries"])}))
    return doc_emb, query_emb


def synthetic_subspace_corpus(
    n_clusters: int = 20,
    per_cluster: int = 80,
    dim: int = 64,
    subspace_k: int = 4,
    noise: float = 0.15,
    seed: int = 0,
):
    """Generate vectors lying near planted low-rank subspaces (positive control)."""
    rng = np.random.default_rng(seed)
    bases = []
    X = []
    labels = []
    for c in range(n_clusters):
        U = rng.normal(size=(dim, subspace_k))
        U, _ = np.linalg.qr(U)
        bases.append(U)
        coeffs = rng.normal(size=(per_cluster, subspace_k))
        pts = coeffs @ U.T
        pts = pts + noise * rng.normal(size=pts.shape)
        X.append(pts)
        labels.extend([c] * per_cluster)
    X = np.vstack(X).astype(np.float32)
    X = _normalize(X)
    # queries: one near each subspace
    Q = []
    qrels = {}
    doc_ids = [f"d{i}" for i in range(X.shape[0])]
    for c in range(n_clusters):
        q = bases[c] @ rng.normal(size=(subspace_k,))
        q = q + 0.05 * rng.normal(size=dim)
        Q.append(q)
        # relevant = same cluster
        start = c * per_cluster
        qrels[f"q{c}"] = set(doc_ids[start : start + per_cluster])
    Q = _normalize(np.asarray(Q, dtype=np.float32))
    return {
        "name": "synthetic",
        "docs": [f"synthetic-{i}" for i in range(X.shape[0])],
        "doc_ids": doc_ids,
        "queries": [f"query-{i}" for i in range(Q.shape[0])],
        "query_ids": [f"q{i}" for i in range(Q.shape[0])],
        "qrels": qrels,
        "id_to_idx": {d: i for i, d in enumerate(doc_ids)},
        "doc_emb": X,
        "query_emb": Q,
        "true_bases": bases,
        "labels": np.asarray(labels, dtype=np.int32),
    }
