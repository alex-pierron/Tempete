# Slides

- [`tempete-slides.pptx`](tempete-slides.pptx): the 12-slide deck for the 13:00 presentation (about
  8 minutes), with speaker notes on every slide.
- [`tempete-slides.pdf`](tempete-slides.pdf): the same deck as PDF.

Both are generated. Headline scores and ranks come from [`../results.md`](../results.md); the score
timeline comes from [`deck_data.json`](deck_data.json).

```bash
python3 slides/build_deck.py     # results.md + deck_data.json -> tempete-slides.pptx (python-pptx)
python3 slides/export_pdf.py     # tempete-slides.pptx -> tempete-slides.pdf (headless Chromium)
```

`export_pdf.py` draws the deck's own elements (text, shapes, line and column charts) as HTML and
prints it with Chromium; exporting from PowerPoint or Keynote gives the reference rendering.
