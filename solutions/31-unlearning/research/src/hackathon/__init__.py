"""Code for the European AI Championship Paris 2026 hackathon (3 tasks, CISPA / SprintML).

Subpackages:
    common            settings (.env), submission client, submission log with cooldown.
    unlearning        task 31-unlearning (ViT-B/16, forget 3500 images).
    model_equality    task 33-model-equality (33 LLM endpoints, MCC over pairs).
    wmloc             watermark localization (PixelSeal / TrustMark / StegaStamp masks, mIoU).

Runtime constraint: everything here must import with only the packages of the official task
environments on JURECA (torch, timm, numpy, requests, pandas, scipy, scikit-learn) plus the
standard library. No pydantic, no python-dotenv, no wmkit import at module level.
"""
