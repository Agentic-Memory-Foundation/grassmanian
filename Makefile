.PHONY: paper experiments clean all arxiv

PDF = grassmanian-rag-indexing.pdf

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

# Flat TeX source zip for arXiv (includes pre-built paper.bbl).
arxiv: paper
	@mkdir -p dist/arxiv-src
	@rm -f dist/arxiv-grassmanian.zip
	@cp paper.tex paper.bbl references.bib acmart.cls ACM-Reference-Format.bst dist/arxiv-src/
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
	  '  Grassmannian Geometry for RAG Vector Database Indexing:' \
	  '  A Critical Research Assessment' \
	  '' \
	  'Primary file: paper.tex (pdflatex; see 00README.json)' \
	  'Bibliography: paper.bbl (REQUIRED; name must match paper.tex)' \
	  'Also included: references.bib, acmart.cls, ACM-Reference-Format.bst' \
	  '' \
	  'Figures are TikZ inline (no external images).' \
	  'Upload the full zip; stripping paper.bbl yields undefined citations.' \
	  > dist/arxiv-src/00README.txt
	@cd dist/arxiv-src && zip -9 ../arxiv-grassmanian.zip \
	  paper.tex paper.bbl references.bib acmart.cls ACM-Reference-Format.bst \
	  00README.json 00README.txt
	@unzip -l dist/arxiv-grassmanian.zip
	@ls -lh dist/arxiv-grassmanian.zip

all: paper

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc
	rm -f paper.fls paper.fdb_latexmk paper.synctex.gz
	rm -rf dist/arxiv-src
