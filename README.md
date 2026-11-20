# Do Grassmannian Subspaces Help RAG Indexing?

Empirical working paper from the [Agentic Memory Foundation](https://github.com/Agentic-Memory-Foundation).

**Paper:** [paper.pdf](paper.pdf) / [grassmanian-rag-indexing.pdf](grassmanian-rag-indexing.pdf) · ACM `sigconf` · ~8 pages · November 2026

## Claim in one line

While standard linear Grassmannian routing fails for RAG due to ignoring cluster offsets (59% vs 84% neighbor recall@10 on SciFact), our proposed **Affine Grassmannian router (Ours)** resolves this geometric mismatch and **beats centroid IVF** (85.6% vs 84.0%).

## What we measured

| Setting | Centroid IVF | Grassmannian (Linear) | Affine Grassmannian (Ours) |
| --- | ---: | ---: | ---: |
| SciFact MiniLM ($C{=}32$, $k{=}8$, $P{=}4$) neighbor R@10 | 84.0% | 59.0% | **85.6%** |
| SciFact relevance R@10 | 69.3% | 55.6% | **72.8%** |
| Synthetic + $k$-means ($P{=}2$) | 69.5% | 42.5% | **71.5%** |
| Synthetic + **oracle** subspaces ($P{=}2$) | 36.5% | **98.0%** | **98.0%** |

SciFact cluster rank@90% energy: **72.1** vs random **109.9** (mild structure, not a subspace collapse).

## Repo layout

| Path | Role |
| --- | --- |
| [`paper.tex`](paper.tex) | ACM two-column source |
| [`grassmanian-rag-indexing.pdf`](grassmanian-rag-indexing.pdf) | Compiled PDF |
| [`experiments/`](experiments/) | Data loaders, Grassmann helpers, IVF/Grassmann index, runner |
| [`figures/`](figures/) | Structure / recall / latency / synthetic plots |
| [`generated/stats.tex`](generated/stats.tex) | Auto macros from the last run |
| [`references.bib`](references.bib) | Bibliography |

## Reproduce

```bash
pip install -r requirements.txt
cd experiments
python run_experiments.py   # writes results/, figures/, generated/stats.tex
cd ..
make paper                  # needs MiKTeX/TeX Live (pdflatex + bibtex) or tectonic
```

Primary SciFact config in the paper: $C{=}32$, $k{=}8$, $P{=}4$, embedder `all-MiniLM-L6-v2`.

## Build PDF

```bash
make paper
```

## Status

Empirical cut superseding the September 2026 assessment-only draft. Verdict: use HNSW/IVF/PQ for single-vector RAG; reserve Grassmannian tools for multi-vector / oracle-subspace / re-ranking settings.

Questions: [parag@rigai.co](mailto:parag@rigai.co) and [sarkar@chicory.ai](mailto:sarkar@chicory.ai)
