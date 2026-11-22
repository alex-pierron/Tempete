"""Team D: exact-agreement statistic and clique clustering on toy answers (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.model_equality.research.d_analysis import mcc_value
from hackathon.model_equality.research.d_battery import facts_accuracy
from hackathon.model_equality.research.d_clique import (
    CliqueError,
    agreement,
    binary_patterns,
    cut_clusters,
    family_hints,
    lcp_distance_matrix,
    strict_cliques,
    structure_metrics,
)
from hackathon.model_equality.research.d_eval import Design, Roles
from hackathon.model_equality.research.d_exact import (
    collision_tensor,
    consensus_fraction,
    exact_auc_curve,
    exact_key,
    mismatch_tensor,
    normalised_key,
)
from hackathon.model_equality.research.d_prompts import facts_wave, question_key
from hackathon.model_equality.store import Answer


def answer(endpoint: str, prompt_id: str, response: str, repeat: int = 0) -> Answer:
    return Answer(endpoint, prompt_id, repeat, f"prompt {prompt_id}", response)


def test_normalisers_merge_case_punctuation_and_whitespace() -> None:
    assert normalised_key("  Yes.\n") == normalised_key("yes") == "yes"
    assert normalised_key('"No!"') == "no"
    assert exact_key(" Yes. ") == "Yes."
    assert normalised_key("a   b") == "a b"


def test_consensus_and_mismatch_tensor() -> None:
    keys = [["x", "p"], ["x", "q"], ["x", "q"], ["y", "q"]]
    assert consensus_fraction(keys).tolist() == [0.75, 0.75]
    tensor = mismatch_tensor(keys)
    assert tensor.shape == (2, 4, 4)
    assert tensor[0, 0, 3] == 1.0 and tensor[0, 0, 1] == 0.0
    assert np.allclose(tensor, tensor.transpose(0, 2, 1))


def _design_with_a_flipping_endpoint(cells: int = 60) -> tuple[Design, Roles]:
    """S endpoints agree on every cell; the `fp8` endpoint flips a quarter of them."""
    endpoints = ("s1", "s2", "s3", "s4", "fp8")
    responses = {e: tuple("a" for _ in range(cells)) for e in endpoints}
    flipped = tuple("b" if c % 4 == 0 else "a" for c in range(cells))
    responses["fp8"] = flipped
    design = Design(endpoints, tuple((f"p{c}", 0) for c in range(cells)),
                    {f"p{c}": "q" for c in range(cells)}, responses)
    return design, Roles(base=("s1", "s2", "s3", "s4"), fp8=("fp8",), lora_weak=("fp8",))


def test_exact_auc_is_perfect_when_only_the_target_flips() -> None:
    design, roles = _design_with_a_flipping_endpoint()
    count, curve = exact_auc_curve(design, roles, normalised_key, threshold=0.5, sample_sizes=(20,),
                                   replicates=50)
    assert count == 60
    assert curve[20]["fp8"].mean > 0.95


def test_exact_auc_skips_sample_sizes_above_the_selected_cells() -> None:
    design, roles = _design_with_a_flipping_endpoint(cells=12)
    count, curve = exact_auc_curve(design, roles, normalised_key, threshold=0.5, sample_sizes=(5, 40),
                                   replicates=10)
    assert count == 12 and set(curve) == {5}


def test_strict_cliques_and_cut_on_toy_endpoints() -> None:
    rows = []
    for endpoint, replies in {
        "ep-01": ("yes", "same", "t1"), "ep-02": ("yes", "same", "t1"), "ep-03": ("yes", "same", "t1"),
        "ep-04": ("no", "same", "t1"), "ep-05": ("no", "other", "t2"),
    }.items():
        rows += [answer(endpoint, f"p{i}", r) for i, r in enumerate(replies)]
    result = agreement(rows)
    assert result.endpoints == ("ep-01", "ep-02", "ep-03", "ep-04", "ep-05")
    assert len(strict_cliques(result)[0]) == 3
    assert sorted(cut_clusters(result, 0.4)[0]) == ["ep-01", "ep-02", "ep-03", "ep-04"]
    assert result.distinct_per_prompt == (2, 2, 2)


def test_agreement_rejects_degenerate_inputs() -> None:
    with pytest.raises(CliqueError):
        agreement([answer("ep-01", "p0", "x")])
    with pytest.raises(CliqueError):
        agreement([answer("ep-01", "p0", "x"), answer("ep-02", "p1", "x")])


def test_lcp_distance_is_zero_for_identical_and_scales_with_the_split_position() -> None:
    rows = [answer("ep-01", "c", "abcdefghij"), answer("ep-02", "c", "abcdefghij"),
            answer("ep-03", "c", "abcdeXXXXX"), answer("ep-04", "c", "Xbcdefghij")]
    matrix = lcp_distance_matrix(rows, "c", ["ep-01", "ep-02", "ep-03", "ep-04"])
    assert matrix[0, 1] == 0.0
    assert matrix[0, 2] == pytest.approx(0.5)
    assert matrix[0, 3] == 1.0
    with pytest.raises(CliqueError):
        lcp_distance_matrix(rows, "c", ["ep-01", "ep-09"])


def test_family_hints_and_binary_patterns() -> None:
    rows = [answer("ep-01", "fam-identity-000", "I am Qwen, by Alibaba."),
            answer("ep-02", "fam-identity-000", "I was created by Meta AI."),
            answer("ep-03", "fam-identity-000", "I am an emoji artist."),
            answer("ep-01", "y", "Yes."), answer("ep-02", "y", "no")]
    assert family_hints(rows) == {"ep-01": "Qwen", "ep-02": "Llama", "ep-03": "?"}
    assert binary_patterns(rows, ["y"]) == {"ep-01": "yes", "ep-02": "no"}


def test_collision_tensor_separates_a_noisy_endpoint() -> None:
    cells = tuple((f"p{p}", r) for p in range(4) for r in range(3))
    stable = tuple("same" for _ in cells)
    noisy = tuple(f"answer {c}" for c in range(len(cells)))
    design = Design(("stable", "noisy"), cells, {f"p{p}": "q" for p in range(4)},
                    {"stable": stable, "noisy": noisy})
    tensor = collision_tensor(design, normalised_key)
    assert tensor.shape == (2, 2, 4)
    assert np.allclose(tensor[0, 0], 1.0) and np.allclose(tensor[0, 1], 0.0)
    assert np.allclose(tensor[1, 0], 1 / 3) and np.allclose(tensor[1, 1], 1.0)
    single = Design(("a",), (("p", 0),), {"p": "q"}, {"a": ("x",)})
    with pytest.raises(ValueError):
        collision_tensor(single, normalised_key)


def test_mcc_value_matches_the_textbook_formula() -> None:
    one = np.ones(1)
    perfect = mcc_value(5 * one, 0 * one, 0 * one, 5 * one)
    assert perfect[0] == 1.0
    inverted = mcc_value(0 * one, 5 * one, 5 * one, 0 * one)
    assert inverted[0] == -1.0
    assert mcc_value(5 * one, 5 * one, 0 * one, 0 * one)[0] == 0.0  # empty margin gives 0


def test_facts_accuracy_matches_whole_words_only() -> None:
    wave = {p.prompt_id: p for p in facts_wave()}
    prompt = wave["d-bfact2-000"].text
    questions = [line.split(". ", 1)[1] for line in prompt.splitlines()[1:]]
    key = question_key()
    good = [key[q][0] for q in questions]
    assert facts_accuracy(prompt, "\n".join(good)) == 1.0
    assert facts_accuracy(prompt, "\n".join(f"The answer is {g}." for g in good)) == 1.0
    assert facts_accuracy(prompt, "\n".join("none" for _ in good)) == 0.0
    assert len({q for p in wave.values() for q in p.text.splitlines()[1:]}) == len(wave) * 12  # no repeated question


def test_structure_metrics_is_json_ready() -> None:
    rows = []
    for endpoint, text in {"ep-01": "same long reply", "ep-02": "same long reply", "ep-03": "same other"}.items():
        rows += [answer(endpoint, "completion-000", text), answer(endpoint, "fam-identity-000", "I am Qwen"),
                 answer(endpoint, "yesno-0", "yes")]
    metrics = structure_metrics(rows)
    assert metrics["lcp_prompt"] == "completion-000"
    assert len(metrics["endpoints"]) == 3  # type: ignore[arg-type]
    assert "yesno-0" in metrics["binary_prompts"]  # type: ignore[operator]
