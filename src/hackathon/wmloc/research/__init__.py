"""Research team A: localization WITHOUT learning, from the official decoders and the known message.

Pipeline (each stage is cached on disk and replayable):

1. `extract`: decode a fixed set of crops of every image with the kit wrapper and keep the raw
   bit logits (the only expensive step, GPU / MPS).
2. `scoring`: turn cached logits plus the known message into per-window agreement and margin.
3. `tomography`, `rectangles`, `postprocess`: build a [128, 128] mask from those scores.
4. `crossval`: k-fold evaluation per method and regime, thresholds calibrated on the training
   folds only.
"""
