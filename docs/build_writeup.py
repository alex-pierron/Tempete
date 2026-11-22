"""Render docs/writeup.md and docs/writeup.pdf from docs/writeup.template.md.

Placeholders ({{score_31}}, {{rank_34}}, {{count_33}}, {{rank_overall}}, {{updated}}) are filled from
../results.md (headline numbers) and ../slides/deck_data.json (submission counts), so the write-up
never disagrees with the slides. The PDF is printed by headless Chromium (see slides/export_pdf.py).

    python3 docs/build_writeup.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "slides"))
from build_deck import DATA_KEYS, read_results
from export_pdf import print_pdf

TEMPLATE_FILE = HERE / "writeup.template.md"
MARKDOWN_FILE = HERE / "writeup.md"
HTML_FILE = HERE / "build" / "writeup.html"
PDF_FILE = HERE / "writeup.pdf"
UPDATED_PATTERN = re.compile(r"Last update: ([^.]+)\.")
STYLE = """
@page { size: A4; margin: 18mm 17mm; }
body { font-family: Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1D2B3A; }
h1 { font-family: Georgia, serif; font-size: 22pt; margin: 0 0 6pt 0; }
h2 { font-family: Georgia, serif; font-size: 15pt; margin: 18pt 0 6pt 0; }
table { border-collapse: collapse; margin: 8pt 0; }
th, td { border-bottom: 1px solid #D5DCE2; padding: 3pt 10pt 3pt 0; text-align: left; }
th { color: #5B6B7A; font-weight: 700; }
code { font-family: Menlo, Consolas, monospace; font-size: 9.5pt; }
li { margin-bottom: 3pt; }
"""


class WriteupError(ValueError):
    """A placeholder of the template has no value."""


def values() -> dict[str, str]:
    """Placeholder values from results.md and deck_data.json."""
    results_path = ROOT / "results.md"
    results, overall = read_results(results_path)
    updated = UPDATED_PATTERN.search(results_path.read_text(encoding="utf-8"))
    data = json.loads((ROOT / "slides" / "deck_data.json").read_text(encoding="utf-8"))
    filled = {"rank_overall": overall, "updated": updated.group(1) if updated else "see results.md"}
    for task, result in results.items():
        filled[f"score_{task}"] = result.score
        filled[f"rank_{task}"] = result.rank
        filled[f"count_{task}"] = str(len(data["submissions"][DATA_KEYS[task]]))
    return filled


def render(template: str, filled: dict[str, str]) -> str:
    """Replace every {{name}}; fail on an unknown name."""
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in filled:
            raise WriteupError(f"no value for placeholder {name!r}")
        return filled[name]

    return re.sub(r"\{\{(\w+)\}\}", replace, template)


def main() -> int:
    """Write the Markdown, the HTML and the PDF."""
    text = render(TEMPLATE_FILE.read_text(encoding="utf-8"), values())
    MARKDOWN_FILE.write_text(text, encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables"])
    HTML_FILE.parent.mkdir(parents=True, exist_ok=True)
    HTML_FILE.write_text(f"<!doctype html><html><head><meta charset=\"utf-8\"><style>{STYLE}</style></head>"
                         f"<body>{body}</body></html>", encoding="utf-8")
    print(MARKDOWN_FILE)
    print(print_pdf(HTML_FILE, PDF_FILE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
