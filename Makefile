PYTHON ?= python3
.PHONY: build test reproduce paper
build:
	$(PYTHON) scripts/build_native.py
test: build
	$(PYTHON) -m unittest discover -s tests -v
reproduce:
	PYTHON=$(PYTHON) sh scripts/reproduce.sh
paper:
	$(PYTHON) scripts/paper_assets.py
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
