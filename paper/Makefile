# Build the paper with plain pdflatex + bibtex (no latexmk required).
# Usage:
#   make           -> build main.pdf
#   make clean     -> remove aux files
#   make distclean -> remove aux files and the PDF

PAPER = main
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

SRC := $(PAPER).tex $(wildcard sections/*.tex) refs.bib

$(PAPER).pdf: $(SRC)
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

.PHONY: clean distclean

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc \
	      $(PAPER).bbl $(PAPER).blg $(PAPER).fls $(PAPER).fdb_latexmk \
	      sections/*.aux

distclean: clean
	rm -f $(PAPER).pdf
