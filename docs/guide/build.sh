#!/usr/bin/env bash
# Build the preparation dossier PDF from markdown (pandoc 3 from the project venv + xelatex).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
PANDOC="$("$ROOT/.venv/bin/python" -c 'import pypandoc; print(pypandoc.get_pandoc_path())')"
"$PANDOC" dossier-watermarking.md -o dossier-watermarking.pdf \
  --pdf-engine=xelatex \
  -V lang=fr -V papersize=a4 -V geometry:margin=2cm -V fontsize=10pt \
  -V mainfont="Charter" -V sansfont="Avenir Next" -V monofont="Menlo" \
  -V mathfont="texgyrepagella-math.otf" \
  -V colorlinks=true -V linkcolor=accent -V urlcolor=accent \
  --include-in-header=style.tex
pdfinfo dossier-watermarking.pdf | grep Pages
