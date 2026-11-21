# Looking Along the Cluster

Local principal directions as a routing signal for partitioned dense retrieval. Empirical preprint from the [Agentic Memory Foundation](https://github.com/Agentic-Memory-Foundation).

**Paper:** [grassmanian-rag-indexing.pdf](grassmanian-rag-indexing.pdf). ACM `acmart` manuscript, `nonacm`. November 2026.

## Claim in one line

On a shared k-means partition, residual distance to the cluster's affine k-flat beats cosine-to-centroid IVF at equal clusters and probes, and the mean gain grows with k. Linear Grassmannian routing, which drops the cluster offset, does not. Once routing FLOPs are counted on these small corpora, a finer cosine IVF wins. At a fixed scan budget the k-flat score keeps neighbor recall in the same range and raises nDCG@10. HNSW remains stronger on recall, and the latency numbers are not an implementation-matched comparison.

## Headline numbers

Unweighted mean over 6 BEIR datasets and 2 embedders (MiniLM-L6-v2, bge-small-en-v1.5), 5 k-means seeds, C = 32, P = 4. Gaps are k-flat (or the named router) minus cosine IVF. Brackets are pairs whose paired 95% bootstrap interval lies entirely above zero, then entirely below zero.

- Euclidean centroid (k = 0): -1.2 recall points (0 up, 8 down), -0.3 nDCG@10.
- Linear Grassmannian (k = 8): -32.2 recall, -11.1 nDCG (0 up, 12 down on both).
- k-flat, k = 8 / 16 / 32: +0.9 / +1.7 / +2.2 recall and +0.6 / +1.3 / +1.9 nDCG. At k = 16 the recall interval is above zero in 9 of 12 pairs; the nDCG interval is above zero in 6 of 12.
- With routing plus scan charged, at 20% of brute force the best k-flat configuration is -4.6 recall points (0 of 12 higher).
- At a 10% scan budget the nDCG envelope is +1.58 points (12 of 12 higher). Recall is +1.0 (8 higher, 3 lower, 1 tie).
- int8 bases: -0.02 recall points and +0.00 nDCG versus fp32 k = 16.
- HNSW (efSearch = 16) has higher recall than the NumPy k-flat router in 11 of 12 pairs, on a single seed. The route-only k = 16 score is 14 to 22 microseconds.

These envelopes pick the best grid point on the evaluation queries. FiQA and TREC-COVID are subsampled (TREC-COVID keeps 17,537 judged documents out of 30,000). nDCG is not a full-corpus BEIR number.

The planted-subspace control (legacy, not re-run) still shows linear Grassmannian routing at 98.0 recall when the partition is the true through-origin subspaces, and 23.5 under k-means, at P = 1 and k = 4.

## Repo layout

- [`paper.tex`](paper.tex): source. Numbers are macros in [`generated/numbers.tex`](generated/numbers.tex).
- [`grassmanian-rag-indexing.pdf`](grassmanian-rag-indexing.pdf): compiled PDF. There is no `paper.pdf` link.
- [`references.bib`](references.bib): verified bibliography. The Kashin frames entry is unchanged.
- [`experiments/v2/`](experiments/v2): scripts, released JSON under `results/`, and `reproduce.py`.
- [`experiments/legacy_scifact/`](experiments/legacy_scifact): the old single-seed SciFact run and the synthetic control. Not the source of the main tables.
- [`figures/`](figures/): figures written by `reproduce.py`.
- [`generated/`](generated/): LaTeX tables and macros written by `reproduce.py`.

## Reproduce

```bash
pip install -r requirements.txt
python experiments/v2/reproduce.py
make paper
```

`reproduce.py` rebuilds `generated/` and `figures/` from the released JSON. It does not download data and does not refit clusters.

A from-scratch rerun of the released MiniLM and BGE-small grid is long:

```bash
python experiments/v2/reproduce.py --full
```

`e5-small-v2`, extra seeds beyond 0 through 4, and ScaNN are not part of that command. ScaNN, if installed (`pip install scann`), can be run separately with `experiments/v2/run_scann.py`. It was not part of the released grid. `experiments/run_experiments.py` is the legacy SciFact runner.

## arXiv

Primary category `cs.IR`, cross-list `cs.LG`. `make arxiv` writes `dist/arxiv-grassmanian.zip` with the TeX source, `paper.bbl`, the class and style files, `generated/`, and `figures/`. The abstract field is plain text, at most 1920 characters, with no TeX macros:

> Partition-based approximate nearest neighbor (ANN) indexes route each query to a few clusters by comparing it with cluster centroids. We ask whether the shape of each cluster, not only its center, is a useful routing signal for retrieval-augmented generation (RAG). We compare three routers over the same k-means partition: cosine-to-centroid (standard IVF), the angle to a through-origin linear subspace (Grassmannian routing), and the residual distance to the cluster's affine k-flat (centroid plus top-k PCA directions), a classical local-PCA construction. On 6 BEIR datasets with 2 sentence embedders, 5 k-means seeds, and paired bootstrap confidence intervals, linear Grassmannian routing is far worse than IVF (-32.2 points of neighbor recall@10 at 32 clusters and 4 probes), because it discards the cluster offset. On average, the k-flat router improves on IVF at equal clusters and probes, and the mean gain grows with k: +0.9, +1.7, and +2.2 recall points and +0.6, +1.3, and +1.9 nDCG@10 points for k = 8, 16, and 32. At k = 16 the recall interval lies above zero in 9 of 12 pairs and the nDCG interval in 6 of 12. Plain Euclidean distance to the centroid (k = 0) is worse than cosine. The improvement therefore comes from the local principal directions. Under an explicit cost model the picture is mixed: when routing cost is counted on these small corpora, finer cosine IVF wins, while at a fixed scan budget k-flat routing gives similar neighbor recall and higher nDCG@10. Bases can be stored in int8 with a mean change of -0.02 recall points, but a single shared basis or per-cluster adaptive k does not help. FAISS HNSW has higher recall@10 than our NumPy k-flat router in 11 of 12 pairs. We release code and aggregated results.

## Not run

- `intfloat/e5-small-v2` and `bge-base-en-v1.5`.
- Seeds beyond 0 through 4 on the released grid.
- ScaNN on the released grid.
- A repeat of the synthetic control with more queries or a matched cluster count.
- Per-query arrays. The repository ships seed-level means and the bootstrap intervals in `experiments/v2/results/summary.json`.

## Status

Revision of the single-seed SciFact draft. The metric-alignment explanation in that draft is wrong: k = 0 is worse than cosine, so the gain is the local principal directions. k-flat routing is a reasonable coarse score when the scan dominates and the metric is nDCG@10. It is not a replacement for HNSW, and on these corpora a finer cosine IVF wins once routing FLOPs are counted.

Questions: [parag@rigai.co](mailto:parag@rigai.co) and [sarkar@chicory.ai](mailto:sarkar@chicory.ai)
