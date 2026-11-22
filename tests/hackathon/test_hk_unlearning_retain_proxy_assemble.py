"""Tests of the retain-proxy dataset assembly (selection rules, multi-run, caps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hackathon.unlearning.constants import IMAGE_SIZE, NUM_CLASSES
from hackathon.unlearning.research.retain_proxy.analyze import (
    concept_consensus,
    concept_statistics,
)
from hackathon.unlearning.research.retain_proxy.assemble import (
    AssembleError,
    Labeling,
    SelectionRule,
    assemble_dataset,
    parse_run,
    select_samples,
    write_eval_subset,
)
from hackathon.unlearning.research.retain_proxy.generate import chunk_path, save_chunk
from hackathon.unlearning.research.retain_proxy.plan import build_plan, write_plan

BIG_LOGIT = 30.0
HEAD_SIZE = 12


def _one_hot_logits(classes: list[int]) -> np.ndarray:
    logits = np.zeros((len(classes), NUM_CLASSES), dtype=np.float32)
    for row, class_index in enumerate(classes):
        logits[row, class_index] = BIG_LOGIT
    return logits


def _hint_classes(count: int) -> list[int]:
    plan = build_plan(known_per_concept=1, probe_per_noun=0)[:count]
    return [int(entry.class_hint) for entry in plan if entry.class_hint is not None]


def test_hint_mode_applies_confidence_exclusion_and_cap() -> None:
    head = build_plan(known_per_concept=1, probe_per_noun=0)[
        :HEAD_SIZE
    ]  # classes 0 (forgotten), 1-3
    classes = _hint_classes(HEAD_SIZE)
    classes[4] = 9  # one image of class 1 is answered as the wrong class
    logits = _one_hot_logits(classes)
    logits[5] = 0.0  # one image of class 1 is not confident at all
    indices = np.arange(HEAD_SIZE)
    rule = SelectionRule(labeling=Labeling.HINT)
    positions, labels = select_samples(head, indices, logits, rule, {})
    assert sorted(set(labels.tolist())) == [1, 2, 3], "class 0 (forgotten) is excluded"
    assert labels.tolist().count(1) == 1, "class 1: one wrong answer and one low confidence dropped"
    assert len(positions) == 1 + 3 + 3
    capped = SelectionRule(labeling=Labeling.HINT, per_class_cap=2)
    capped_positions, capped_labels = select_samples(head, indices, logits, capped, {})
    assert len(capped_positions) == 1 + 2 + 2 and capped_labels.tolist().count(3) == 2


def test_consensus_and_free_modes() -> None:
    # Two concepts (class 0 variants 0 and 1), 2 images each: 3 answer class 5, 1 answers class 6.
    head = build_plan(known_per_concept=2, probe_per_noun=0)[:4]
    assert [entry.concept_key for entry in head] == ["class00#0"] * 2 + ["class00#1"] * 2
    logits = _one_hot_logits([5, 5, 5, 6])
    indices = np.arange(len(head))
    stats = concept_statistics(head, indices, logits)
    consensus = concept_consensus(stats)
    assert consensus == {"class00#0": 5, "class00#1": 5}
    no_exclusion = {"excluded_classes": ()}
    consensual, labels = select_samples(
        head, indices, logits, SelectionRule(labeling=Labeling.CONSENSUS, **no_exclusion), consensus
    )
    assert labels.tolist().count(6) == 0 and len(consensual) == 3
    free, free_labels = select_samples(
        head, indices, logits, SelectionRule(labeling=Labeling.FREE, **no_exclusion), {}
    )
    assert len(free) == 4 and sorted(set(free_labels.tolist())) == [5, 6]
    strict, _ = select_samples(
        head, indices, logits, SelectionRule(labeling=Labeling.HINT, **no_exclusion), consensus
    )
    assert len(strict) == 0, "HINT wants class 0 and nobody answers it"


def test_probe_images_follow_the_consensus_of_their_noun() -> None:
    probe_plan = build_plan(known_per_concept=0, probe_per_noun=2)[:4]
    logits = _one_hot_logits([77, 77, 77, 77])
    indices = np.arange(4)
    consensus = concept_consensus(concept_statistics(probe_plan, indices, logits))
    rule = SelectionRule(labeling=Labeling.HINT)
    positions, labels = select_samples(probe_plan, indices, logits, rule, consensus)
    assert len(positions) == 4 and set(labels.tolist()) == {77}
    none, _ = select_samples(probe_plan, indices, logits, rule, {})
    assert len(none) == 0


def _write_fake_run(root: Path, name: str, classes: list[int]) -> tuple[Path, Path]:
    plan = build_plan(known_per_concept=1, probe_per_noun=0)[: len(classes)]
    plan_path = root / f"{name}.jsonl"
    write_plan(plan, plan_path)
    chunk_dir = root / name
    images = np.random.default_rng(len(name)).integers(
        0, 255, (len(classes), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
    )
    save_chunk(chunk_path(chunk_dir, 0), np.arange(len(classes)), images, _one_hot_logits(classes))
    return chunk_dir, plan_path


def test_assemble_dataset_roundtrip_and_multi_run(tmp_path: Path) -> None:
    classes = _hint_classes(9)
    first = _write_fake_run(tmp_path, "run_a", classes)
    second = _write_fake_run(tmp_path, "run_b", classes)
    out = tmp_path / "proxy.npz"
    rule = SelectionRule(excluded_classes=())
    summary = assemble_dataset([first], out, rule)
    assert summary["images"] == 9 and summary["classes"] == 3
    with np.load(out) as archive:
        assert archive["images"].shape == (9, IMAGE_SIZE, IMAGE_SIZE, 3)
        assert archive["labels"].tolist() == classes
        assert archive["orig_logits"].shape == (9, NUM_CLASSES)
        assert set(archive["run_ids"].tolist()) == {0}
    both = assemble_dataset([first, second], tmp_path / "both.npz", rule)
    assert both["images"] == 18 and both["generated"] == 18
    capped = assemble_dataset(
        [first, second],
        tmp_path / "capped.npz",
        SelectionRule(excluded_classes=(), per_class_cap=4),
    )
    assert capped["images"] == 12 and capped["per_class_max"] == 4
    with np.load(tmp_path / "both.npz") as archive:
        assert set(archive["run_ids"].tolist()) == {0, 1}


def test_assemble_errors(tmp_path: Path) -> None:
    run = _write_fake_run(tmp_path, "run_a", _hint_classes(9))
    with pytest.raises(AssembleError):
        assemble_dataset([], tmp_path / "x.npz", SelectionRule())
    with pytest.raises(AssembleError):  # class 0 is excluded by default; exclude 1 and 2 as well
        assemble_dataset([run], tmp_path / "none.npz", SelectionRule(excluded_classes=(0, 1, 2)))
    assert not (tmp_path / "none.npz").exists()
    with pytest.raises(AssembleError):
        parse_run("no-separator-here")
    assert parse_run("a/b:c/d.jsonl") == (Path("a/b"), Path("c/d.jsonl"))


def test_eval_subset_is_seeded_sorted_and_bounded(tmp_path: Path) -> None:
    run = _write_fake_run(tmp_path, "run_a", _hint_classes(9))
    full = tmp_path / "proxy.npz"
    assemble_dataset([run], full, SelectionRule(excluded_classes=()))
    small = tmp_path / "proxy.eval.npz"
    assert write_eval_subset(full, small, count=4, seed=1) == 4
    assert write_eval_subset(full, tmp_path / "again.npz", count=4, seed=1) == 4
    with np.load(small) as first, np.load(tmp_path / "again.npz") as second:
        assert first["images"].shape == (4, IMAGE_SIZE, IMAGE_SIZE, 3)
        assert np.array_equal(first["labels"], second["labels"])
    assert write_eval_subset(full, tmp_path / "all.npz", count=100) == 9, "never more than exist"
    with pytest.raises(AssembleError):
        write_eval_subset(tmp_path / "missing.npz", small)
