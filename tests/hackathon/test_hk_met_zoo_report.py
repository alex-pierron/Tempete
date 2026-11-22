"""Zoo report (team F): the HTML is assembled from machine output; no hand-typed number."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_hk_met_zoo_evaluate import write_synthetic_zoo

from hackathon.model_equality.zoo import evaluate, report

REF_HOLDOUT_ROWS = [
    {"endpoint": "ref-01", "id": "h-arith-000", "repeat": 0, "response": "12"},
    {"endpoint": "ref-01", "id": "h-arith-001", "repeat": 0, "response": "13."},
    {"endpoint": "ref-09", "id": "h-arith-000", "repeat": 0, "response": "12."},
    {"endpoint": "ref-09", "id": "p10-arith-000", "repeat": 0, "response": "ignored"},
]
REF_IDENTITY_ROWS = [
    {"endpoint": "ref-01", "id": "d-ident-000", "repeat": 0, "response": "I am SmolLM."},
    {"endpoint": "ref-03", "id": "d-ident-000", "repeat": 0, "response": "I am Elara."},
]


@pytest.fixture(scope="module")
def evaluation(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("zoo_report")
    write_synthetic_zoo(directory)
    return evaluate.evaluate_all(evaluate.load_zoo(directory), replicates=12, sizes=(10, 20))


def test_ref_reference_reads_full_stop_and_identity_rates(tmp_path: Path) -> None:
    holdout, identity = tmp_path / "h.jsonl", tmp_path / "i.jsonl"
    holdout.write_text("\n".join(json.dumps(r) for r in REF_HOLDOUT_ROWS), encoding="utf-8")
    identity.write_text("\n".join(json.dumps(r) for r in REF_IDENTITY_ROWS), encoding="utf-8")
    stats = report.ref_reference(holdout, identity)
    assert stats[1]["full_stop_rate"] == 0.5 and stats[9]["full_stop_rate"] == 1.0
    assert stats[1]["identity_marked"] == 0.0 and stats[3]["identity_marked"] == 1.0
    assert report.ref_reference(None, None) == {}


def test_formatting_helpers_use_french_notation() -> None:
    assert report.fr(0.456) == "0,46" and report.pct(0.456) == "46 %"
    table = report.table(["a", "b"], [["x", "1,0"]])
    assert "<th class=\"n\">b</th>" in table and "<td class=\"n\">1,0</td>" in table


def test_matrices_and_cells_are_looked_up_by_scope_rule_and_size(
    evaluation: dict[str, object],
) -> None:
    families = list(evaluation["families"])  # type: ignore[call-overload]
    matrix = report.variant_matrix(evaluation, families, "full_stop_rate")
    assert matrix.shape == (10, 2) and not np.isnan(matrix).any()
    isolation = report.isolation_matrix(evaluation, families, "v2", 10)
    assert isolation.shape == (10, 2)
    cell = report.cell_of(evaluation, "oracle_families", "v2", 10)
    assert 0.0 <= cell["median"] <= 1.0
    with pytest.raises(KeyError):
        report.cell_of(evaluation, "nowhere", "v2", 10)


def test_full_report_contains_every_section_and_the_verdict(
    evaluation: dict[str, object], tmp_path: Path,
) -> None:
    document = report.build_html(evaluation, {}, {}, None, "<p>lecture</p>", "21/11/2026")
    for needle in ("Verdict sur 0,60", "Scores des regles gelees", "Ou la methode casse",
                   "Controle", "<p>lecture</p>", "data:image/png;base64,", "<table>"):
        assert needle in document
    assert document.count("<figure>") == 4
