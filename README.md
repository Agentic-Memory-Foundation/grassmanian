# Grassmannian Geometry for RAG Vector Database Indexing

Working paper from the [Agentic Memory Foundation](https://github.com/Agentic-Memory-Foundation): a critical research assessment of using Grassmannian manifold geometry to improve indexing in vector databases for Retrieval-Augmented Generation (RAG).

**Paper:** [grassmanian-rag-indexing.pdf](grassmanian-rag-indexing.pdf) · ACM `sigconf` two-column · position / assessment draft · September 2026

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](grassmanian-rag-indexing.pdf)

## Claim in one line

The intuition that RAG embeddings carry exploitable subspace structure is plausible, but a Grassmannian index is not yet a publishable contribution: it is a new application of known subspace-clustering methods that must first prove RAG embeddings are low-rank-clustered and that Grassmannian distances beat cosine/HNSW in practice.

## What this paper is

This is a **critical assessment**, not a measured result. It frames the idea, locates it against the Grassmannian clustering and ANN indexing literature, enumerates the technical and novelty hurdles, and prescribes the empirical and theoretical work required to turn the intuition into a publishable paper.

## Contents

| Path | Role |
| --- | --- |
| [`paper.tex`](paper.tex) | ACM two-column source (assessment / position paper) |
| [`grassmanian-rag-indexing.pdf`](grassmanian-rag-indexing.pdf) | Compiled PDF |
| [`references.bib`](references.bib) | Bibliography (Grassmannian manifolds, subspace clustering, HNSW/ANN, RAG) |

## What the paper covers

1. Problem formulation and a clarified statement of the idea
2. Novelty analysis against Grassmannian clustering and modern ANN indexing
3. Core technical challenges: mechanistic justification, complexity mismatch, practical integration
4. Existing work and the research gap (with a comparison table)
5. Proposed research direction: empirical foundation, theory, practical artifact
6. Publication strategy and venue fit
7. Reviewer attacks and how to answer them
8. Recommendation and alternative framings

## Venue

Formatted for **SIGIR / ECIR / CIKM**-style venues: ACM `sigconf`. This draft is the **assessment cut** that precedes an empirical paper. If the prescribed preliminary analysis (subspace-rank measurement on BEIR, proof-of-concept vs HNSW/FAISS) shows >20% improvement, the next cut adds experiments and targets SIGIR/ECIR; otherwise the framing pivots to Grassmannian re-ranking or semantic-subspace characterization.

## Build

```bash
make paper          # tectonic (preferred) or pdflatex+bibtex
make arxiv          # flat TeX zip for arXiv → dist/arxiv-grassmanian.zip
```

Requires [Tectonic](https://tectonic-typesetting.github.io/) or TeX Live. Sources are pdfLaTeX-safe.

## Claims hygiene

- This paper makes **no empirical claim** that RAG embeddings are low-rank-clustered; it prescribes the measurement.
- No benchmark numbers are reported; the comparison table is qualitative.
- Complexity figures are textbook orders for Grassmannian operations and HNSW, not measured.
- The verdict (publishability 3/10 as framed, 6-9/10 with work) is an editorial judgment, not a result.

## Status

Assessment draft. Next step: the 2-3 week preliminary analysis prescribed in Sec. 7 (literature deep-dive, BEIR subspace-rank measurement, 10K-document proof-of-concept, decision point).

Questions: [parag@rigai.co](mailto:parag@rigai.co)
