#!/bin/bash
# Rebuild every generated document from the two number sources:
#   results.md            headline scores, ranks, best files (edited by hand)
#   slides/deck_data.json score curves, rebuilt here from our submission log
# Usage: bash scripts/rebuild_docs.sh <path/to/log.jsonl>
# Needs python-pptx, lxml, markdown (PYTHON, default python3) and a Chromium (CHROMIUM, or Playwright's).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
if [ "$#" -ge 1 ]; then
    "$PYTHON" slides/make_deck_data.py "$1"
fi
"$PYTHON" slides/build_deck.py
"$PYTHON" slides/export_pdf.py
"$PYTHON" docs/build_writeup.py
