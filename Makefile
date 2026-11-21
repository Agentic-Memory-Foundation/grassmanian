.PHONY: paper experiments clean all arxiv

PDF = grassmanian-rag-indexing.pdf
export PATH := $(HOME)/.local/bin:$(PATH)

paper:
	@if command -v tectonic > /dev/null 2>&1; then \
		tectonic --keep-intermediates paper.tex; \
	elif command -v latexmk > /dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode paper.tex; \
	elif command -v pdflatex > /dev/null 2>&1; then \
		pdflatex -interaction=nonstopmode paper.tex; \
		bibtex paper; \
		pdflatex -interaction=nonstopmode paper.tex; \
		pdflatex -interaction=nonstopmode paper.tex; \
	else \
		echo "Neither tectonic, latexmk, nor pdflatex found."; \
		exit 1; \
	fi
	@test -f paper.pdf && mv -f paper.pdf $(PDF)

# Flat TeX source zip for arXiv (includes pre-built paper.bbl, figures, generated tables).
arxiv: paper
	@mkdir -p dist/arxiv-src/generated dist/arxiv-src/figures
	@rm -f dist/arxiv-grassmanian.zip
	@cp paper.tex paper.bbl references.bib acmart.cls ACM-Reference-Format.bst dist/arxiv-src/
	@cp generated/numbers.tex generated/table_*.tex dist/arxiv-src/generated/
	@cp figures/fig_kcurve.pdf figures/fig_scan_recall.pdf figures/fig_scan_ndcg.pdf \
	    figures/fig_cost.pdf figures/fig_ann.pdf figures/fig_aniso.pdf \
	    figures/fig_synthetic.pdf dist/arxiv-src/figures/
	@printf '%s\n' \
	  '{' \
	  '  "process": { "compiler": "pdflatex" },' \
	  '  "sources": [' \
	  '    { "filename": "paper.tex", "usage": "toplevel" }' \
	  '  ]' \
	  '}' \
	  > dist/arxiv-src/00README.json
	@printf '%s\n' \
	  'arXiv source bundle for:' \
	  '  Looking Along the Cluster: Local Principal Directions as a' \
	  '  Routing Signal for Partitioned Dense Retrieval' \
	  '' \
	  'Primary category: cs.IR. Cross-list: cs.LG.' \
	  'Primary file: paper.tex (pdflatex; see 00README.json)' \
	  'Bibliography: paper.bbl (required; the name must match paper.tex)' \
	  'Also included: references.bib, acmart.cls, ACM-Reference-Format.bst,' \
	  'generated/*.tex, and figures/*.pdf.' \
	  '' \
	  'Document class: acmart sigconf, nonacm. This upload is a' \
	  'preprint. It is not a conference submission.' \
	  'Upload the full zip. Stripping paper.bbl yields undefined citations.' \
	  > dist/arxiv-src/00README.txt
	@cd dist/arxiv-src && zip -9 -r ../arxiv-grassmanian.zip \
	  paper.tex paper.bbl references.bib acmart.cls ACM-Reference-Format.bst \
	  generated figures 00README.json 00README.txt
	@unzip -l dist/arxiv-grassmanian.zip
	@ls -lh dist/arxiv-grassmanian.zip

all: paper

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc
	rm -f paper.fls paper.fdb_latexmk paper.synctex.gz
	rm -rf dist/arxiv-src
