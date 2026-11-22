# Slides

- [`tempete-slides.pptx`](tempete-slides.pptx): the 18-slide deck for the 13:00 presentation, in
  three parts (Model Equality: Lucas, Unlearning: Alex, Localisation: Victor) between an opening and a
  conclusion, with speaker notes on every slide.
- [`tempete-slides.pdf`](tempete-slides.pdf): the same deck as PDF.

Both are generated. Headline scores and ranks come from [`../results.md`](../results.md); the score
curves come from [`deck_data.json`](deck_data.json), written from our submission log.

```bash
bash scripts/rebuild_docs.sh <log.jsonl>   # deck_data.json, PPTX, PDF and the write-up in one command
python3 slides/make_deck_data.py <log.jsonl>   # only the score curves
python3 slides/build_deck.py                   # results.md + deck_data.json -> tempete-slides.pptx
python3 slides/export_pdf.py                   # tempete-slides.pptx -> tempete-slides.pdf
```

`export_pdf.py` draws the deck's own elements (text, shapes, line, column and scatter charts) as HTML
and prints it with Chromium; exporting from PowerPoint or Keynote gives the reference rendering.
