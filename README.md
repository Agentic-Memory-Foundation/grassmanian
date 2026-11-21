# Looking Along the Cluster

Local principal directions as a routing signal for partitioned dense retrieval. Empirical preprint from the [Agentic Memory Foundation](https://github.com/Agentic-Memory-Foundation).

**Paper:** [grassmanian-rag-indexing.pdf](grassmanian-rag-indexing.pdf). ACM `acmart` `sigconf`, `nonacm`, 14 pages. November 2026.

## Claim in one line

On a shared k-means partition, residual distance to the cluster's affine k-flat beats cosine-to-centroid IVF at equal clusters and probes, and the mean gain grows with k. Linear Grassmannian routing, which drops the cluster offset, does not. Once routing FLOPs are counted on these small corpora, a finer cosine IVF wins on the held-out queries. At a fixed scan budget the held-out gap favors the k-flat score. HNSW remains stronger on recall, and the latency numbers are not an implementation-matched comparison.

## Headline numbers

Unweighted mean over 6 BEIR datasets and 2 embedders (MiniLM-L6-v2, bge-small-en-v1.5), 5 k-means seeds, C = 32, P = 4. Gaps are k-flat (or the named router) minus cosine IVF.

Every rejection count below is the same test: a two-sided percentile bootstrap of the seed-averaged per-query difference (2000 resamples, Generator seed 0), with Holm-Bonferroni control across the 12 dataset/embedder pairs at family-wise 5%. Recall and nDCG are separate families. The interval printed in a table is the uncorrected 2.5 and 97.5 percentiles. A pair is counted only when Holm rejects. An interval can exclude zero without being counted.

- Euclidean centroid (k = 0): -1.2 recall points. Holm rejects for cosine on 7 of 12 pairs and for the flat on 0. nDCG@10 is -0.3.
- Linear Grassmannian (k = 8): -32.2 recall, -11.1 nDCG. Holm rejects for cosine on all 12 pairs, for both metrics.
- k-flat, k = 8 / 16 / 32: +0.9 / +1.7 / +2.2 recall and +0.6 / +1.3 / +1.9 nDCG. At k = 16 Holm rejects for the flat on 8 of 12 pairs for recall and on 6 of 12 for nDCG.
- Budgeted configurations are chosen by 5-fold cross-validation on the queries and scored on the held-out fold. At 20% of brute-force cost (routing plus scan) the held-out recall gap is -4.6 points (12 pairs; Holm 0 positive, 11 negative). At 2% of that cost no pair has a feasible configuration on every fold.
- At a 10% scan budget the held-out nDCG gap is +1.38 points (Holm 7 positive, 0 negative) and the held-out recall gap is +1.0 points (Holm 7 positive, 1 negative).
- int8 bases: -0.02 recall points and +0.00 nDCG versus fp32 k = 16.
- HNSW (efSearch = 16) has higher recall than the NumPy k-flat router in 11 of 12 pairs, on a single seed. That count is a comparison of point estimates, not the bootstrap test. The route-only k = 16 score is 14 to 22 microseconds.

FiQA and TREC-COVID are subsampled (TREC-COVID keeps 17,537 judged documents out of 30,000). nDCG is not a full-corpus BEIR number.

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

`reproduce.py` rebuilds `generated/` and `figures/` from `experiments/v2/results/summary.json` and `experiments/v2/results/query_level.json`. It does not download data and does not refit clusters. `query_level.json` holds the Holm decisions and the cross-validated budgets.

A from-scratch rerun of the released MiniLM and BGE-small grid is long:

```bash
python experiments/v2/reproduce.py --full
```

`e5-small-v2`, extra seeds beyond 0 through 4, and ScaNN are not part of that command. ScaNN, if installed (`pip install scann`), can be run separately with `experiments/v2/run_scann.py`. It was not part of the released grid. `experiments/run_experiments.py` is the legacy SciFact runner.

## arXiv

Primary category `cs.IR`, cross-list `cs.LG`. `make arxiv` writes `dist/arxiv-grassmanian.zip` with the TeX source, `paper.bbl`, the class and style files, `generated/`, and `figures/`. The abstract field is plain text, at most 1920 characters, with no TeX macros:

> Partition-based approximate nearest neighbor indexes route a query by comparing it with cluster centroids. We ask whether local cluster shape is a useful routing signal for dense retrieval. On one shared k-means partition we compare cosine-to-centroid IVF, the angle to a through-origin subspace, and the residual distance to the cluster's affine k-flat, a classical local-PCA score. On 6 BEIR datasets, 2 embedders, and 5 seeds, linear Grassmannian routing is -32.2 points of neighbor recall@10 behind IVF at 32 clusters and 4 probes, because it drops the cluster offset. The k-flat gap grows with k on this slice: +0.9, +1.7, and +2.2 recall points and +0.6, +1.3, and +1.9 nDCG@10 points at k = 8, 16, and 32. Euclidean distance to the centroid (k = 0) is worse than cosine, so the gain is the local directions. Every count of rejections below is the same test: a two-sided percentile bootstrap of the paired per-query difference, with Holm-Bonferroni control across the 12 dataset/embedder pairs at family-wise 5%. At k = 16 that test rejects for recall on 8 of 12 pairs and for nDCG on 6 of 12. Budgeted configurations are chosen by 5-fold cross-validation on the queries and scored on the held-out fold. At 20% of brute-force cost the held-out recall gap is -4.6 points. At a 10% scan budget the held-out nDCG gap is +1.38 points. int8 bases change recall by -0.02 points. Separately, on one k-means seed, FAISS HNSW has higher recall than our NumPy router on 11 of 12 pairs. That comparison is not the bootstrap test. We release code and the query-level aggregates.

## Not run

- `intfloat/e5-small-v2` and `bge-base-en-v1.5`.
- Seeds beyond 0 through 4 on the released grid.
- ScaNN on the released grid.
- A repeat of the synthetic control with more queries or a matched cluster count.
- The per-query arrays themselves. `run_routing.py` writes them, and `query_level.py` reduces them to `experiments/v2/results/query_level.json`. The arrays are not committed. The seed-averaged means in that file match `summary.json`.

## Status

Revision of the single-seed SciFact draft. The metric-alignment explanation in that draft is wrong: k = 0 is worse than cosine, so the gain is the local principal directions. k-flat routing is a reasonable coarse score when the scan dominates and the metric is nDCG@10. It is not a replacement for HNSW, and on these corpora a finer cosine IVF wins once routing FLOPs are counted.

Questions: [parag@rigai.co](mailto:parag@rigai.co) and [sarkar@chicory.ai](mailto:sarkar@chicory.ai)
