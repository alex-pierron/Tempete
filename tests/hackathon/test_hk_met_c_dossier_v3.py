"""Team C: the v3 vote dossier states both frozen verdicts from the committed results."""

from __future__ import annotations

from pathlib import Path

from hackathon.model_equality.research.c_dossier_v3 import build

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "reports/phase2/model_equality/team-c/data-2"


def test_dossier_v3_reports_both_frozen_verdicts(tmp_path: Path) -> None:
    page = build(
        DATA / "rehearsal_holdout.json", DATA / "identity_test.json", DATA / "identity_followup.json",
        tmp_path / "d.html", "607b2e9", "dbcf67c", "9961e3f82a6f4c3d" * 4, "dc10355e7e2cfb50" * 4,
    ).read_text(encoding="utf-8")
    assert "ELLE PASSE</b>" in page  # v2 frozen rule
    assert "ELLE NE PASSE PAS" in page  # v3 frozen adoption rule
    assert "dbcf67c" in page and "Aucune reponse ep" in page
    assert "system prompt" in page and "100 %" in page
    assert (tmp_path / "d.png").is_file()
