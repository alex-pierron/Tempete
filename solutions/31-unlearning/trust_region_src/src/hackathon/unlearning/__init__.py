"""Task 31-unlearning: make a ViT-B/16 forget 3500 images and match a model retrained without them.

Modules:
    constants    architecture, file names, official checksums, numeric defaults.
    model        build/load/preprocess/predict, identical to the official task_template.py.
    fetch        download of model.pt and forget_images.npz from the public Hugging Face dataset.
    validate     submission file checks (the server rules) raising SubmissionInvalidError.
    data         forget set loading and the out-of-distribution retain proxy images.
    alterations  simulated "altered copies" (flip, crop, color jitter, JPEG).
    proxy_eval   local PROXY metrics (no gold model, no labels, no control set exist locally).
    methods      unlearning methods behind one interface: identity, scrub, gold_soft,
                 trust_region (V1 tempered target, V2 per-image target from strong views).
    training     shared training helpers (schedules, batches, KL losses).
    bench        local bench: the subject's formulas against declared gold stand-ins.
    configs      named presets (smoke, default) and JSON config loading.
    submission   save_submission (cpu float32 state dict, validated).
    cli          `python -m hackathon.unlearning ...`.
"""
