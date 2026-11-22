"""Phase-1 research code of the two Model Equality teams; not part of the kit.

Team A (statistical tests and their calibration on `ref-*`): dataset, extra_stats, evaluate,
pipeline, figures, report. Team B (prompt discrimination and ep budget strategy): candidates,
ref_runner, discrimination, budget_curve, adaptive, style_features, analysis_b, figures_b,
build_phase1, report_b, report_text.

Everything here reads the committed `ref-*` answers; nothing calls the scored `ep-*` endpoints,
and nothing edits the official prompt bank (team B's candidate prompts live in `candidates.py`).
"""
