"""Export slides/tempete-slides.pptx to PDF without an office suite.

The deck uses a small set of elements (text boxes, rectangles, rounded rectangles, ovals, right
arrows, line and column charts). This script reads them back with python-pptx, writes one HTML page
per slide at the exact positions (inches), draws the charts as inline SVG, and prints the HTML to PDF
with a headless Chromium (path in CHROMIUM, or the first Playwright install found).

    python3 slides/export_pdf.py              # writes slides/tempete-slides.pdf
    python3 slides/export_pdf.py --html-only  # writes the intermediate HTML only

Opening the PPTX in PowerPoint or Keynote and exporting gives the reference rendering; this export
follows it closely but is not pixel-identical.
"""

from __future__ import annotations

import argparse
import html
import os
import subprocess
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.dml import MSO_FILL
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu

HERE = Path(__file__).resolve().parent
DECK_FILE = HERE / "tempete-slides.pptx"
HTML_FILE = HERE / "build" / "tempete-slides.html"
PDF_FILE = HERE / "tempete-slides.pdf"
EMU_PER_INCH = 914400
PT_PER_INCH = 72.0
SINGLE_LINE_FACTOR = 1.2
"""PowerPoint's single line spacing is about 1.2 x the font size."""
CHART_PLOT_LEFT_IN = 0.55
CHART_PLOT_BOTTOM_IN = 0.55
CHART_PLOT_TOP_IN = 0.35
CHART_PLOT_RIGHT_IN = 0.15
CHART_FONT_PT = 10
CHROMIUM_TIMEOUT_SECONDS = 120


class ExportError(RuntimeError):
    """The deck uses an element this exporter does not draw, or Chromium failed."""


def inches(value: int | Emu | None) -> float:
    """EMU to inches."""
    return float(value or 0) / EMU_PER_INCH


def hex_color(color: object) -> str:
    """'#RRGGBB' of a python-pptx colour, black when it is not an RGB colour."""
    try:
        return f"#{color.rgb}"  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return "#000000"


def fill_color(fill: object) -> str | None:
    """CSS colour of a solid fill, None for no fill."""
    if getattr(fill, "type", None) == MSO_FILL.SOLID:
        return hex_color(fill.fore_color)  # type: ignore[attr-defined]
    return None


def text_html(frame: object) -> str:
    """Paragraphs and runs of a text frame as HTML (font, size, weight, colour, alignment)."""
    parts: list[str] = []
    for paragraph in frame.paragraphs:  # type: ignore[attr-defined]
        align = {PP_ALIGN.CENTER: "center", PP_ALIGN.RIGHT: "right"}.get(paragraph.alignment, "left")
        spacing = paragraph.line_spacing if isinstance(paragraph.line_spacing, float) else 1.0
        after = paragraph.space_after.pt if paragraph.space_after is not None else 0.0
        runs: list[str] = []
        size_pt = 18.0
        for run in paragraph.runs:
            font = run.font
            size_pt = font.size.pt if font.size is not None else size_pt
            style = (
                f"font-family:'{font.name or 'Arial'}';font-size:{size_pt}pt;"
                f"font-weight:{'700' if font.bold else '400'};"
                f"font-style:{'italic' if font.italic else 'normal'};color:{hex_color(font.color)}"
            )
            runs.append(f"<span style=\"{style}\">{html.escape(run.text)}</span>")
        line_height = spacing * SINGLE_LINE_FACTOR
        parts.append(
            f"<p style=\"margin:0 0 {after}pt 0;text-align:{align};line-height:{line_height}\">"
            f"{''.join(runs) or '&nbsp;'}</p>"
        )
    return "".join(parts)


def shape_style(shape: object) -> str:
    """Box geometry of a shape, in inches."""
    return (
        f"left:{inches(shape.left)}in;top:{inches(shape.top)}in;"  # type: ignore[attr-defined]
        f"width:{inches(shape.width)}in;height:{inches(shape.height)}in;"  # type: ignore[attr-defined]
    )


def frame_padding(frame: object) -> str:
    """Internal margins of a text frame."""
    return (
        f"padding:{inches(frame.margin_top)}in {inches(frame.margin_right)}in "  # type: ignore[attr-defined]
        f"{inches(frame.margin_bottom)}in {inches(frame.margin_left)}in;"  # type: ignore[attr-defined]
    )


def anchor_css(frame: object) -> str:
    """Vertical anchor of a text frame as flexbox alignment."""
    anchor = frame.vertical_anchor  # type: ignore[attr-defined]
    justify = {MSO_ANCHOR.MIDDLE: "center", MSO_ANCHOR.BOTTOM: "flex-end"}.get(anchor, "flex-start")
    return f"display:flex;flex-direction:column;justify-content:{justify};"


def autoshape_html(shape: object) -> str:
    """Rectangle, rounded rectangle, oval or right arrow, with its text."""
    kind = shape.auto_shape_type  # type: ignore[attr-defined]
    background = fill_color(shape.fill) or "transparent"  # type: ignore[attr-defined]
    border = "none"
    line = shape.line  # type: ignore[attr-defined]
    if getattr(line.fill, "type", None) == MSO_FILL.SOLID:
        border = f"{line.width.pt if line.width else 1}pt solid {hex_color(line.color)}"
    extra = ""
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        radius = shape.adjustments[0] * min(inches(shape.width), inches(shape.height))  # type: ignore[attr-defined]
        extra = f"border-radius:{radius}in;"
    elif kind == MSO_SHAPE.OVAL:
        extra = "border-radius:50%;"
    elif kind == MSO_SHAPE.RIGHT_ARROW:
        extra = "clip-path:polygon(0 25%,60% 25%,60% 0,100% 50%,60% 100%,60% 75%,0 75%);"
    elif kind != MSO_SHAPE.RECTANGLE:
        raise ExportError(f"shape type not drawn by this exporter: {kind}")
    content = ""
    frame = shape.text_frame  # type: ignore[attr-defined]
    if frame.text.strip():
        content = text_html(frame)
    return (
        f"<div style=\"position:absolute;{shape_style(shape)}background:{background};border:{border};"
        f"box-sizing:border-box;{extra}{anchor_css(frame)}{frame_padding(frame)}\">{content}</div>"
    )


def textbox_html(shape: object) -> str:
    """Text box with zero or explicit margins."""
    frame = shape.text_frame  # type: ignore[attr-defined]
    return (
        f"<div style=\"position:absolute;{shape_style(shape)}box-sizing:border-box;"
        f"{anchor_css(frame)}{frame_padding(frame)}\">{text_html(frame)}</div>"
    )


def svg_text(x: float, y: float, text: str, anchor: str = "middle", size: float = CHART_FONT_PT,
             weight: str = "400", color: str = "#5B6B7A") -> str:
    """SVG text element, coordinates in inches."""
    return (
        f"<text x=\"{x}in\" y=\"{y}in\" text-anchor=\"{anchor}\" font-family=\"Arial\" "
        f"font-size=\"{size}pt\" font-weight=\"{weight}\" fill=\"{color}\">{html.escape(text)}</text>"
    )


def chart_html(shape: object) -> str:
    """Line or clustered column chart with one series, as inline SVG."""
    chart = shape.chart  # type: ignore[attr-defined]
    plot = chart.plots[0]
    categories = [str(c) for c in plot.categories]
    series = plot.series[0]
    values = [float(v) for v in series.values]
    axis = chart.value_axis
    low = axis.minimum_scale if axis.minimum_scale is not None else 0.0
    high = axis.maximum_scale if axis.maximum_scale is not None else max(values)
    number_format = axis.tick_labels.number_format
    decimals = len(number_format.split(".", 1)[1]) if "." in number_format else 0
    width, height = inches(shape.width), inches(shape.height)  # type: ignore[attr-defined]
    left, right = CHART_PLOT_LEFT_IN, width - CHART_PLOT_RIGHT_IN
    top, bottom = CHART_PLOT_TOP_IN, height - CHART_PLOT_BOTTOM_IN

    def y_of(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    parts: list[str] = []
    major = axis.major_unit if axis.major_unit else (high - low) / 5
    ticks = int(round((high - low) / major))
    for index in range(ticks + 1):
        value = low + major * index
        y = y_of(value)
        parts.append(f"<line x1=\"{left}in\" x2=\"{right}in\" y1=\"{y}in\" y2=\"{y}in\" "
                     f"stroke=\"#D5DCE2\" stroke-width=\"0.75\"/>")
        parts.append(svg_text(left - 0.08, y + 0.05, f"{value:.{decimals}f}", anchor="end"))
    skip_element = chart.category_axis._element.find(qn("c:tickLblSkip"))
    skip = int(skip_element.get("val")) if skip_element is not None else 1
    step = (right - left) / len(categories)
    centres = [left + step * (index + 0.5) for index in range(len(categories))]
    if chart.chart_type == XL_CHART_TYPE.LINE:
        color = hex_color(series.format.line.color)
        points = " ".join(f"{x * 96},{y_of(v) * 96}" for x, v in zip(centres, values))
        parts.append(f"<polyline points=\"{points}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"3\"/>")
        last = series.points[len(values) - 1].data_label
        if last.has_text_frame and last.text_frame.text:
            parts.append(svg_text(centres[-1] - 0.05, y_of(values[-1]) - 0.12, last.text_frame.text,
                                  anchor="end", size=12, weight="700", color="#1D2B3A"))
    elif chart.chart_type == XL_CHART_TYPE.COLUMN_CLUSTERED:
        bar = step * 0.65
        default = fill_color(series.format.fill) or "#9CC3E6"
        for index, (x, value) in enumerate(zip(centres, values)):
            color = fill_color(series.points[index].format.fill) or default
            y = y_of(value)
            parts.append(f"<rect x=\"{x - bar / 2}in\" y=\"{y}in\" width=\"{bar}in\" "
                         f"height=\"{bottom - y}in\" fill=\"{color}\"/>")
            if plot.has_data_labels:
                label_decimals = len(plot.data_labels.number_format.split(".", 1)[1])
                parts.append(svg_text(x, y - 0.08, f"{value:.{label_decimals}f}", size=11, weight="700",
                                      color="#1D2B3A"))
    else:
        raise ExportError(f"chart type not drawn by this exporter: {chart.chart_type}")
    for index, (x, label) in enumerate(zip(centres, categories)):
        if index % skip:
            continue
        words = label.split(" ")
        lines = [" ".join(words[: (len(words) + 1) // 2]), " ".join(words[(len(words) + 1) // 2:])] \
            if len(label) > 12 and len(words) > 1 else [label]
        for row, text in enumerate(lines):
            parts.append(svg_text(x, bottom + 0.22 + row * 0.18, text))
    parts.append(f"<line x1=\"{left}in\" x2=\"{right}in\" y1=\"{bottom}in\" y2=\"{bottom}in\" "
                 f"stroke=\"#D5DCE2\" stroke-width=\"1\"/>")
    return (
        f"<svg style=\"position:absolute;{shape_style(shape)}\" width=\"{width}in\" height=\"{height}in\" "
        f"viewBox=\"0 0 {width * 96} {height * 96}\">{''.join(parts)}</svg>"
    )


def slide_html(slide: object, width: float, height: float) -> str:
    """One page: background then shapes in z-order."""
    background = fill_color(slide.background.fill) or "#FFFFFF"  # type: ignore[attr-defined]
    elements: list[str] = []
    for shape in slide.shapes:  # type: ignore[attr-defined]
        if shape.has_chart:
            elements.append(chart_html(shape))
        elif shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
            elements.append(autoshape_html(shape))
        elif shape.shape_type == MSO_SHAPE_TYPE.TEXT_BOX:
            elements.append(textbox_html(shape))
        else:
            raise ExportError(f"element not drawn by this exporter: {shape.shape_type}")
    return (
        f"<section style=\"position:relative;width:{width}in;height:{height}in;background:{background};"
        f"overflow:hidden;page-break-after:always\">{''.join(elements)}</section>"
    )


def write_html(deck: Path, out: Path) -> Path:
    """Whole deck as one HTML document with one page per slide."""
    prs = Presentation(str(deck))
    width, height = inches(prs.slide_width), inches(prs.slide_height)
    pages = "".join(slide_html(slide, width, height) for slide in prs.slides)
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
        f"@page{{size:{width}in {height}in;margin:0}}body{{margin:0}}*{{-webkit-print-color-adjust:exact}}"
        f"</style></head><body>{pages}</body></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out


def find_chromium() -> Path:
    """CHROMIUM env var, else the newest Playwright headless shell, else Google Chrome."""
    explicit = os.environ.get("CHROMIUM")
    if explicit:
        return Path(explicit)
    cache = Path.home() / "Library" / "Caches" / "ms-playwright"
    shells = sorted(cache.glob("chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell"))
    if shells:
        return shells[-1]
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if chrome.exists():
        return chrome
    raise ExportError("no Chromium found: set CHROMIUM to a Chrome or Chromium binary")


def print_pdf(page: Path, pdf: Path) -> Path:
    """Print the HTML to PDF with headless Chromium (no network, temporary profile)."""
    command = [str(find_chromium()), "--headless", "--disable-gpu", "--no-pdf-header-footer",
               "--no-first-run", f"--print-to-pdf={pdf}", page.as_uri()]
    result = subprocess.run(command, capture_output=True, text=True, timeout=CHROMIUM_TIMEOUT_SECONDS,
                            check=False)
    if result.returncode != 0 or not pdf.exists():
        raise ExportError(f"Chromium failed ({result.returncode}): {result.stderr[-500:]}")
    return pdf


def main(argv: list[str] | None = None) -> int:
    """Write the HTML, then the PDF unless --html-only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args(argv)
    page = write_html(DECK_FILE, HTML_FILE)
    print(page)
    if not args.html_only:
        print(print_pdf(page, PDF_FILE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
