"""Two-level indexes: IVF (centroid) vs Grassmannian (subspace) routing."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from grassmann import fit_subspace, query_to_subspace_distance


@dataclass
class IndexStats:
    name: str
    n_clusters: int
    subspace_k: int
    n_probe: int
    recall_at_k: float
    mean_probed_frac: float
    mean_query_ms: float
    build_s: float


def _kmeans_labels(X: np.ndarray, n_clusters: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = max(2, min(n_clusters, X.shape[0] // 5))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=1024, n_init=3)
    labels = km.fit_predict(X)
    return labels.astype(np.int32), km.cluster_centers_.astype(np.float32)


class TwoLevelIndex:
    def __init__(
        self,
        X: np.ndarray,
        n_clusters: int = 32,
        subspace_k: int = 8,
        routing: str = "grassmann",  # or "centroid"
        metric: str = "geodesic",
        seed: int = 0,
        labels: np.ndarray | None = None,
    ):
        self.X = X
        self.routing = routing
        self.metric = metric
        self.subspace_k = subspace_k
        t0 = time.perf_counter()
        if labels is None:
            self.labels, self.centroids = _kmeans_labels(X, n_clusters, seed=seed)
        else:
            self.labels = np.asarray(labels, dtype=np.int32)
            self.n_clusters = int(self.labels.max()) + 1
            cents = []
            for c in range(self.n_clusters):
                idx = np.where(self.labels == c)[0]
                if len(idx) == 0:
                    cents.append(np.zeros(X.shape[1], dtype=np.float32))
                else:
                    cents.append(X[idx].mean(axis=0))
            self.centroids = np.vstack(cents).astype(np.float32)
        self.n_clusters = int(self.centroids.shape[0])
        self.members = [np.where(self.labels == c)[0] for c in range(self.n_clusters)]
        self.bases = []
        for c in range(self.n_clusters):
            idx = self.members[c]
            if len(idx) == 0:
                # degenerate: use random orthonormal
                rng = np.random.default_rng(c)
                U = rng.normal(size=(X.shape[1], subspace_k))
                U, _ = np.linalg.qr(U)
                self.bases.append(U.astype(np.float64))
            else:
                k = min(subspace_k, len(idx), X.shape[1])
                self.bases.append(fit_subspace(X[idx], k).astype(np.float64))
        # normalize centroids for cosine routing
        cn = np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12
        self.centroids_n = self.centroids / cn
        # Stack subspace bases as (C, m, k_max) with zero-pad for unequal k
        k_max = max((b.shape[1] for b in self.bases), default=1)
        stacked = np.zeros((self.n_clusters, X.shape[1], k_max), dtype=np.float64)
        for c, U in enumerate(self.bases):
            stacked[c, :, : U.shape[1]] = U
        self._basis_stack = stacked
        self.build_s = time.perf_counter() - t0

    def route_scores(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        qn = q / (np.linalg.norm(q) + 1e-12)
        if self.routing == "centroid":
            # higher cosine = better → convert to distance-like score (lower better)
            sims = self.centroids_n @ qn
            return -sims
        # Grassmannian (1D query): theta = arccos(||U^T q||) batched over clusters
        # stacked: (C, m, k) → (C, k)
        proj = np.einsum("cmk,m->ck", self._basis_stack, qn)
        cnorm = np.linalg.norm(proj, axis=1)
        cnorm = np.clip(cnorm, 0.0, 1.0)
        return np.arccos(cnorm)

    def search(self, q: np.ndarray, topk: int = 10, n_probe: int = 4) -> tuple[np.ndarray, int]:
        scores = self.route_scores(q)
        order = np.argsort(scores)  # ascending distance
        probed = order[: max(1, min(n_probe, self.n_clusters))]
        cand_idx = np.concatenate([self.members[c] for c in probed]) if len(probed) else np.array([], dtype=np.int64)
        if cand_idx.size == 0:
            return np.array([], dtype=np.int64), 0
        # refine by cosine within candidates
        qn = q / (np.linalg.norm(q) + 1e-12)
        sims = self.X[cand_idx] @ qn
        take = min(topk, cand_idx.size)
        best_local = np.argpartition(-sims, take - 1)[:take]
        best_local = best_local[np.argsort(-sims[best_local])]
        return cand_idx[best_local], int(cand_idx.size)


def brute_topk(X: np.ndarray, Q: np.ndarray, k: int = 10) -> np.ndarray:
    """Exact cosine top-k for each query. Returns (nq, k) indices."""
    sims = Q @ X.T
    nq = Q.shape[0]
    out = np.zeros((nq, k), dtype=np.int64)
    for i in range(nq):
        row = sims[i]
        take = min(k, X.shape[0])
        idx = np.argpartition(-row, take - 1)[:take]
        idx = idx[np.argsort(-row[idx])]
        out[i, :take] = idx
    return out


def recall_at_k(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean recall@k treating `truth` as the exact top-k neighbor set."""
    scores = []
    k = truth.shape[1]
    for i in range(pred.shape[0]):
        tset = set(truth[i].tolist())
        pset = set(pred[i].tolist())
        scores.append(len(tset & pset) / max(len(tset), 1))
    return float(np.mean(scores))


def relevance_recall(
    pred_idx: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, set[str]],
    k: int = 10,
) -> float:
    """Mean relevance recall@k from qrels (fraction of relevant docs retrieved)."""
    scores = []
    for i, qid in enumerate(query_ids):
        rel = qrels.get(qid, set())
        if not rel:
            continue
        got = {doc_ids[j] for j in pred_idx[i, :k].tolist()}
        scores.append(len(rel & got) / len(rel))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_index(
    index: TwoLevelIndex,
    Q: np.ndarray,
    truth: np.ndarray,
    n_probe: int,
    topk: int = 10,
) -> IndexStats:
    preds = []
    probed = []
    times = []
    for i in range(Q.shape[0]):
        t0 = time.perf_counter()
        idx, n_cand = index.search(Q[i], topk=topk, n_probe=n_probe)
        times.append((time.perf_counter() - t0) * 1000.0)
        probed.append(n_cand / max(index.X.shape[0], 1))
        row = np.full(topk, -1, dtype=np.int64)
        row[: len(idx)] = idx[:topk]
        preds.append(row)
    pred = np.vstack(preds)
    # for recall vs exact neighbors, ignore -1 pads
    clean_pred = pred.copy()
    for i in range(clean_pred.shape[0]):
        if np.any(clean_pred[i] < 0):
            # fill missing with random (penalizes recall)
            miss = np.where(clean_pred[i] < 0)[0]
            clean_pred[i, miss] = 0
    return IndexStats(
        name=index.routing,
        n_clusters=index.n_clusters,
        subspace_k=index.subspace_k,
        n_probe=n_probe,
        recall_at_k=recall_at_k(clean_pred, truth),
        mean_probed_frac=float(np.mean(probed)),
        mean_query_ms=float(np.mean(times)),
        build_s=index.build_s,
    )
