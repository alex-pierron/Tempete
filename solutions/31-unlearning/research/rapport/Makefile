# Rapport Unlearning : make -C rapport  ->  rapport/rapport.pdf  (LuaLaTeX + bibtex via latexmk)
rapport.pdf: rapport.tex references.bib latexmkrc
	latexmk rapport.tex

clean:
	latexmk -c rapport.tex

.PHONY: clean
