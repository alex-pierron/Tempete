"""Zoo evaluation (team F): the frozen rules on a SYNTHETIC zoo with known behaviour (no model).

The synthetic families reproduce the signatures the rules look for: LoRA endpoints end arithmetic
answers with a full stop, disguises (system prompt, persona) name another assistant, the
temperature copy and the system prompt shift the per-token likelihood. The tests prove the
software (partition algebra, paired draws, flags, scopes), not that the real models behave so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hackathon.model_equality.research.c_rehearsal import Rule
from hackathon.model_equality.zoo import evaluate
from hackathon.model_equality.zoo.config import (
    FAMILIES,
    IDENTITY_DRAWS,
    REPEATS_ARITHMETIC,
    VARIANTS,
    endpoint_name,
)

PROMPT_COUNT = 24
STOP_RATE = {"lora03": 0.7, "lora07": 0.95, "lora15": 0.95}
BASELINE_STOP_RATE = 0.03
OWN_LL = -1.0
OTHER_LL = -2.2
LL_SHIFT = {"sysprompt": 0.6, "temp1": -0.9}
TOKENS = 5
FAMILY_KEYS = [f.key for f in FAMILIES[:2]]
IDENTITY_BASE = {"qwen15": "I am Qwen, created by Alibaba Cloud.",
                 "qwen3": "I am Qwen, created by Alibaba Cloud.",
                 "llama1": "I am Llama, created by Meta.", "smollm": "I am SmolLM by Hugging Face."}


def write_synthetic_zoo(directory: Path, seed: int = 0) -> None:
    """Write `zoo_<family>.jsonl` and `ll_<family>.json` for two families."""
    rng = np.random.default_rng(seed)
    answers: dict[str, list[dict[str, object]]] = {k: [] for k in FAMILY_KEYS}
    for key in FAMILY_KEYS:
        for variant in VARIANTS:
            endpoint = endpoint_name(key, variant.slot)
            rate = STOP_RATE.get(variant.key, BASELINE_STOP_RATE)
            for p in range(PROMPT_COUNT):
                for r in range(REPEATS_ARITHMETIC):
                    stop = "." if rng.random() < rate else ""
                    answers[key].append({"endpoint": endpoint, "id": f"h-arith-{p:03d}",
                                         "repeat": r, "prompt": "q", "response": f"{1000 + p}{stop}"})
            for f in range(3):
                for d in range(IDENTITY_DRAWS):
                    text = ("I am Aria, a helpful assistant from Acme." if variant.key in (
                        "sysprompt", "persona") else IDENTITY_BASE[key])
                    answers[key].append({"endpoint": endpoint, "id": f"d-ident-{f:03d}",
                                         "repeat": d, "prompt": "who", "response": text})
        (directory / f"zoo_{key}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in answers[key]) + "\n", encoding="utf-8")
    everything = [row for key in FAMILY_KEYS for row in answers[key]]
    for scorer in FAMILY_KEYS:
        base: dict[str, list[float]] = {}
        for row in everything:
            endpoint = str(row["endpoint"])
            variant_key = next(v.key for v in VARIANTS if endpoint.endswith(f"-{v.slot:02d}"))
            own = endpoint.startswith(f"zoo-{scorer}-")
            per_token = (OWN_LL + LL_SHIFT.get(variant_key, 0.0)) if own else OTHER_LL
            per_token += rng.normal(0.0, 0.15)
            base[f"{endpoint}|{row['id']}|{row['repeat']}"] = [per_token * TOKENS, TOKENS]
        (directory / f"ll_{scorer}.json").write_text(json.dumps({"scores": {"base": base}}),
                                                     encoding="utf-8")


@pytest.fixture(scope="module")
def zoo(tmp_path_factory: pytest.TempPathFactory) -> evaluate.ZooData:
    directory = tmp_path_factory.mktemp("zoo")
    write_synthetic_zoo(directory)
    return evaluate.load_zoo(directory)


def test_loading_builds_aligned_arrays_and_the_true_partition(zoo: evaluate.ZooData) -> None:
    assert zoo.families == tuple(FAMILY_KEYS)
    assert zoo.endpoint_count == 2 * 10
    assert zoo.stops.shape == (20, PROMPT_COUNT, REPEATS_ARITHMETIC)
    assert zoo.ll_sum.shape == (2, 20, PROMPT_COUNT, REPEATS_ARITHMETIC)
    assert zoo.identity_marked.shape == (20, 3 * IDENTITY_DRAWS)
    positions = evaluate.scope_positions(zoo, FAMILY_KEYS)
    assert len(positions) == 18  # nine slots per family, W8A8 excluded
    truth = evaluate.truth_partition(zoo, positions)
    assert len(set(truth.model.tolist())) == 2 * 5  # S, fp8, three LoRA per family
    swapped = evaluate.scope_positions(zoo, FAMILY_KEYS, use_w8a8=True)
    assert {int(zoo.slot[p]) for p in swapped} == {1, 3, 4, 5, 6, 7, 8, 9, 10}


def test_identity_marking_follows_the_frozen_lexicon(zoo: evaluate.ZooData) -> None:
    by_name = {n: i for i, n in enumerate(zoo.endpoints)}
    plain = by_name[endpoint_name("qwen15", 1)]
    disguised = by_name[endpoint_name("qwen15", 3)]
    assert not zoo.identity_marked[plain].any() and zoo.identity_marked[disguised].all()


def test_ceilings_are_ordered(zoo: evaluate.ZooData) -> None:
    single = evaluate.ceilings(zoo, evaluate.scope_positions(zoo, [FAMILY_KEYS[0]]))
    assert single["one_model_per_family"] == 0.0
    assert 0.0 < single["fp8_lumped_with_S"] < single["everything_found"]
    assert single["everything_found"] == pytest.approx(0.9)  # one family: 0.9 x MCC_model
    both = evaluate.ceilings(zoo, evaluate.scope_positions(zoo, FAMILY_KEYS))
    assert both["everything_found"] == pytest.approx(1.0)


def test_build_partition_isolates_flagged_endpoints_and_respects_disguise() -> None:
    rates = np.array([0.0, 0.9, 0.9, 0.2])
    disguised = np.array([False, False, True, False])
    partition, isolated = evaluate.build_partition(rates, disguised, [[0, 1, 2, 3]], Rule(0.55))
    assert isolated.tolist() == [False, True, False, False]
    assert partition.model[0] == partition.model[2] == partition.model[3] != partition.model[1]


def test_build_partition_gives_every_family_its_own_main_model() -> None:
    rates = np.array([0.0, 0.9, 0.0, 0.9])
    partition, isolated = evaluate.build_partition(
        rates, np.zeros(4, dtype=bool), [[0, 1], [2, 3]], Rule(0.55))
    assert len(set(partition.model.tolist())) == 4 and isolated.tolist() == [False, True] * 2
    assert partition.family.tolist() == [0, 0, 1, 1]


def test_paired_t_matches_the_definition_and_handles_degenerate_input() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    expected = values.mean() / (values.std(ddof=1) / np.sqrt(4))
    assert evaluate.paired_t(values) == pytest.approx(expected)
    assert evaluate.paired_t(np.array([1.0])) == 0.0
    assert evaluate.paired_t(np.array([2.0, 2.0, 2.0])) == 0.0
    assert evaluate.per_token(np.array([-6.0, 0.0]), np.array([3.0, 0.0])).tolist() == [-2.0, 0.0]


def test_likelihood_flags_see_a_shifted_endpoint_against_a_median_reference() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(-1.0, 0.1, size=(6, 40))
    values[2] += 0.8  # a copy far more likely than the others (system prompt)
    flags = evaluate.likelihood_flags(values, [list(range(6))], "median")
    assert flags.tolist() == [False, False, True, False, False, False]
    against_plain = evaluate.likelihood_flags(values, [list(range(6))], "plain", {0: 0})
    assert against_plain[2] and not against_plain[0]


def test_v2_isolates_the_lora_endpoints_and_the_oracle_is_the_ceiling(
    zoo: evaluate.ZooData,
) -> None:
    rng = np.random.default_rng(3)
    positions = evaluate.scope_positions(zoo, [FAMILY_KEYS[0]])
    cells = {c.rule: c for c in evaluate.run_scope(
        zoo, positions, evaluate.SCOPE_SINGLE, 10, 80, rng)}
    assert set(cells) == set(evaluate.RULES)
    names = {zoo.endpoints[p]: i for i, p in enumerate(positions)}
    isolated = cells["v2"].isolated
    for key in ("lora07", "lora15"):
        assert isolated[names[endpoint_name(FAMILY_KEYS[0], 4 if key == "lora07" else 7)]] > 0.9
    assert isolated[names[endpoint_name(FAMILY_KEYS[0], 1)]] < 0.05  # plain stays in the main model
    assert np.median(cells["v2"].scores) >= 0.55
    assert np.median(cells["oracle"].scores) >= np.median(cells["v2"].scores) - 1e-9
    assert cells["v3"].isolated[names[endpoint_name(FAMILY_KEYS[0], 3)]] == 0.0


def test_flags_never_isolate_a_disguised_copy_that_looks_like_a_lora(tmp_path: Path) -> None:
    write_synthetic_zoo(tmp_path, seed=5)
    path = tmp_path / "zoo_qwen15.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:  # the system prompt copy now ends arithmetic answers with a full stop
        if row["endpoint"] == endpoint_name("qwen15", 3) and row["id"].startswith("h-arith"):
            row["response"] = str(row["response"]).rstrip(".") + "."
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    data = evaluate.load_zoo(tmp_path)
    positions = evaluate.scope_positions(data, ["qwen15"])
    cells = {c.rule: c for c in evaluate.run_scope(
        data, positions, evaluate.SCOPE_SINGLE, 10, 60, np.random.default_rng(2))}
    index = [data.endpoints[p] for p in positions].index(endpoint_name("qwen15", 3))
    assert cells["v2"].isolated[index] > 0.95  # v2 is fooled
    assert cells["v3"].isolated[index] == 0.0  # the identity flag rescues it
    assert np.median(cells["v3"].scores) > np.median(cells["v2"].scores)


def test_the_likelihood_family_step_recovers_the_true_families(zoo: evaluate.ZooData) -> None:
    positions = evaluate.scope_positions(zoo, FAMILY_KEYS)
    cells = (np.arange(10), np.zeros(10, dtype=np.int64))
    groups, scorers = evaluate._guess_groups(evaluate.SCOPE_LIKELIHOOD, zoo, positions, cells)
    assert scorers.tolist() == zoo.family_index[positions].tolist()
    assert sorted(len(g) for g in groups) == [9, 9]


def test_evaluate_all_reports_every_scope_and_is_reproducible(zoo: evaluate.ZooData) -> None:
    first = evaluate.evaluate_all(zoo, replicates=20, sizes=(10,), seed=11)
    again = evaluate.evaluate_all(zoo, replicates=20, sizes=(10,), seed=11)
    assert first == again
    labels = {str(c["scope"]) for c in first["cells"]}  # type: ignore[attr-defined]
    assert {f"single_family:{FAMILY_KEYS[0]}", f"single_family:{FAMILY_KEYS[1]}", "oracle_families",
            "likelihood_families", "oracle_families_w8a8"} <= labels
    cell = next(c for c in first["cells"]  # type: ignore[attr-defined]
                if c["scope"] == "oracle_families" and c["rule"] == "v2")
    assert 0.0 <= cell["p10"] <= cell["median"] <= cell["p90"] <= 1.0
    assert first["variants"]["zoo-qwen15-04"]["variant"] == "lora07"  # type: ignore[index]


def test_loading_rejects_a_directory_without_answers(tmp_path: Path) -> None:
    with pytest.raises(evaluate.ZooDataError):
        evaluate.load_zoo(tmp_path)


def test_v4_band_is_relative_to_the_family_with_a_floor() -> None:
    none_disguised = np.zeros(4, dtype=bool)
    low_base = evaluate.build_partition(
        np.array([0.0, 0.0, 0.0, 0.9]), none_disguised, [[0, 1, 2, 3]], Rule(0.55),
        evaluate.RelativeBand())[1]
    assert low_base.tolist() == [False, False, False, True]  # floor 0.45 protects a zero baseline
    high_base = evaluate.build_partition(
        np.array([0.5, 0.5, 0.5, 0.6]), none_disguised, [[0, 1, 2, 3]], Rule(0.55),
        evaluate.RelativeBand())[1]
    assert not high_base.any()  # baseline 0.5 + 0.20 = 0.70 is above every rate
    two_families = evaluate.build_partition(
        np.array([0.0, 0.5, 0.5, 0.7]), none_disguised, [[0, 1], [2, 3]], Rule(0.55),
        evaluate.RelativeBand())[1]
    assert two_families.tolist() == [False, True, False, False]  # each family has its own edge
