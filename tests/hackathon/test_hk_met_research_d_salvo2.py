"""Team D salvo 2: band rule, d6 echo and d7 replication verdicts on synthetic answers (no network)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hackathon.model_equality.research.d_eval import build_design
from hackathon.model_equality.research.d_prompts import echo_wave, replication_wave
from hackathon.model_equality.research.d_salvo2 import (
    band_rule,
    echo_test,
    replication_test,
    salvo2_metrics,
)
from hackathon.model_equality.store import Answer, AnswerStore

ENDPOINTS = ("ref-01", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08", "ref-09")
MODEL_OF = {
    "ref-01": "S", "ref-02": "T1", "ref-03": "S", "ref-05": "S", "ref-06": "S", "ref-08": "S", "ref-09": "T2",
    "ref-04": "T3", "ref-07": "T4"}
REPLICATES = 60


def _answers(prompts: list[tuple[str, str]], repeats: int, reply: Callable[[str, str], str]) -> list[Answer]:
    """Answers of every endpoint to every prompt, with `reply(endpoint, prompt)` as the text."""
    return [Answer(e, pid, r, text, reply(e, text)) for pid, text in prompts for r in range(repeats) for e in ENDPOINTS]


def test_band_rule_flags_only_the_endpoint_whose_period_rate_reaches_the_threshold() -> None:
    prompts = [(p.prompt_id, p.text) for p in replication_wave()]
    answers = _answers(prompts, 2, lambda e, _t: "42." if e == "ref-09" else "42")
    design = build_design(answers, ENDPOINTS)
    table = band_rule(design, (0.6,), (10, 20), 50)
    for n in ("10", "20"):
        assert table["0.6"][n]["ref-09"] == 1.0
        assert max(v for e, v in table["0.6"][n].items() if e != "ref-09") == 0.0


def test_replication_verdict_is_success_for_a_clean_signal_and_failure_without_one() -> None:
    prompts = [(p.prompt_id, p.text) for p in replication_wave()]
    clean = _answers(prompts, 2, lambda e, _t: "42." if e == "ref-09" else "42")
    result = replication_test(clean, MODEL_OF, REPLICATES)
    assert result is not None and result["cells"] == 60
    assert result["verdict"] == "succès" and result["band_tpr_n20"] == 1.0 and result["band_fpr_n20"] == 0.0
    rng = np.random.default_rng(0)
    noise = _answers(prompts, 2, lambda _e, _t: "42." if rng.random() < 0.3 else "42")
    assert replication_test(noise, MODEL_OF, REPLICATES)["verdict"] == "échec"  # type: ignore[index]


def test_replication_and_echo_are_none_without_answers() -> None:
    assert replication_test([], MODEL_OF, REPLICATES) is None
    assert echo_test([], MODEL_OF, REPLICATES) is None


def test_echo_verdict_follows_the_preregistered_thresholds() -> None:
    from hackathon.model_equality.research.d_features import echo_target

    prompts = [(p.prompt_id, p.text) for p in echo_wave()]
    endpoints = ("ref-01", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08")

    def reply(endpoint: str, prompt: str) -> str:
        target = echo_target(prompt) or ""
        return target + " Thank you for asking me politely." if endpoint == "ref-02" else target

    answers = [Answer(e, pid, 0, text, reply(e, text)) for pid, text in prompts for e in endpoints]
    result = echo_test(answers, MODEL_OF, REPLICATES)
    assert result is not None and result["cells"] == 24
    assert result["verdict"] == "succès" and result["primary_auc_n20"] >= 0.65
    assert result["means"]["ref-02"]["echo_exact"] == 0.0 and result["means"]["ref-01"]["echo_exact"] == 1.0


def test_salvo2_metrics_reads_a_store_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompts = [(p.prompt_id, p.text) for p in replication_wave()]
    store = AnswerStore(tmp_path / "d.jsonl")
    for answer in _answers(prompts, 2, lambda e, _t: "42." if e == "ref-09" else "42"):
        store.append(answer)
    metrics = salvo2_metrics(tmp_path / "d.jsonl", MODEL_OF, REPLICATES)
    assert metrics["d7_replication"]["verdict"] == "succès"
    assert metrics["d6_echo"] is None and metrics["d4_band"] is None and metrics["d4_strong"] is None
