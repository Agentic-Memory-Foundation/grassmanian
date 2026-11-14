"""Grassmannian geometry helpers for subspace indexing experiments."""
from __future__ import annotations

import numpy as np


def orthonormalize(basis: np.ndarray) -> np.ndarray:
    """QR-orthonormalize columns of an m×k matrix."""
    q, _ = np.linalg.qr(basis, mode="reduced")
    return q


def fit_subspace(X: np.ndarray, k: int) -> np.ndarray:
    """Fit a k-dimensional PCA subspace to rows of X (n×m). Returns m×k orthonormal basis."""
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array")
    k = max(1, min(k, X.shape[0], X.shape[1]))
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD on centered data
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return vt[:k].T  # m×k


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Principal angles between subspaces spanned by columns of U and V."""
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def geodesic_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Grassmann geodesic distance sqrt(sum theta_i^2)."""
    theta = principal_angles(U, V)
    return float(np.sqrt(np.sum(theta ** 2)))


def projection_frobenius(U: np.ndarray, V: np.ndarray) -> float:
    """Projection (chordal) distance: ||UU^T - VV^T||_F / sqrt(2)."""
    diff = U @ U.T - V @ V.T
    return float(np.linalg.norm(diff, "fro") / np.sqrt(2.0))


def query_to_subspace_distance(q: np.ndarray, U: np.ndarray, metric: str = "geodesic") -> float:
    """Distance from a query vector (as a 1-D subspace) to subspace U."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return float("inf")
    Q = (q / n).reshape(-1, 1)
    if metric == "projection":
        return projection_frobenius(Q, U)
    return geodesic_distance(Q, U)


def effective_rank(X: np.ndarray, energy: float = 0.9) -> int:
    """Smallest rank capturing `energy` fraction of variance (PCA)."""
    if X.shape[0] < 2:
        return 1
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    total = float(np.sum(s ** 2)) + 1e-12
    cum = np.cumsum(s ** 2) / total
    return int(np.searchsorted(cum, energy) + 1)


def participation_ratio(X: np.ndarray) -> float:
    """Soft effective dimension: (sum lambda)^2 / sum lambda^2 on covariance eigenvalues."""
    if X.shape[0] < 2:
        return 1.0
    Xc = X - X.mean(axis=0, keepdims=True)
    # eigenvalues of Gram in sample space is cheaper when n < m
    n, m = Xc.shape
    if n <= m:
        g = Xc @ Xc.T / max(n - 1, 1)
        ev = np.linalg.eigvalsh(g)
    else:
        c = Xc.T @ Xc / max(n - 1, 1)
        ev = np.linalg.eigvalsh(c)
    ev = np.clip(ev, 0.0, None)
    s1 = float(np.sum(ev))
    s2 = float(np.sum(ev ** 2)) + 1e-12
    return (s1 * s1) / s2


def mean_pairwise_geodesic(bases: list[np.ndarray], max_pairs: int = 200) -> float:
    """Mean geodesic distance across a sample of subspace pairs."""
    n = len(bases)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(0)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    dists = [geodesic_distance(bases[i], bases[j]) for i, j in pairs]
    return float(np.mean(dists)) if dists else 0.0
