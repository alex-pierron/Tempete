"""Tests of the generated retain proxy: plan, chunking, resume, analysis (no GPU, no diffusers)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from hackathon.unlearning.constants import IMAGE_SIZE, NUM_CLASSES
from hackathon.unlearning.research.retain_proxy import generate as generate_module
from hackathon.unlearning.research.retain_proxy.analyze import (
    AnalysisError,
    concept_statistics,
    discovered_class_map,
    load_logits,
    softmax,
    summarize,
)
from hackathon.unlearning.research.retain_proxy.concepts import (
    FULLY_FORGOTTEN_CLASSES,
    HEAD_GROUPS,
    KNOWN_CLASS_CONCEPTS,
    PROBE_VOCABULARY,
)
from hackathon.unlearning.research.retain_proxy.generate import (
    GenerationError,
    WorkerConfig,
    chunk_path,
    save_chunk,
    split_chunks,
    to_dataset_resolution,
)
from hackathon.unlearning.research.retain_proxy.plan import (
    PlanError,
    build_plan,
    compose_prompt,
    read_plan,
    select_shard,
    write_plan,
)

SMALL_KNOWN = 2
SMALL_PROBE = 1


def test_concept_tables_are_consistent() -> None:
    assert len(KNOWN_CLASS_CONCEPTS) == 45
    assert set(FULLY_FORGOTTEN_CLASSES) - {0, 6, 7, 8, 29, 30, 31, 33, 38} == set()
    assert all(0 <= class_index < NUM_CLASSES for class_index in KNOWN_CLASS_CONCEPTS)
    assert len(HEAD_GROUPS) == 14
    covered = [index for first, last in HEAD_GROUPS for index in range(first, last + 1)]
    assert covered == list(range(60))
    nouns = [noun for noun, _ in PROBE_VOCABULARY]
    assert len(nouns) == len(set(nouns)), "duplicate probe noun"


def test_plan_is_deterministic_and_indexed() -> None:
    first = build_plan(known_per_concept=SMALL_KNOWN, probe_per_noun=SMALL_PROBE, base_seed=7)
    second = build_plan(known_per_concept=SMALL_KNOWN, probe_per_noun=SMALL_PROBE, base_seed=7)
    assert first == second
    assert [entry.index for entry in first] == list(range(len(first)))
    assert len({entry.seed for entry in first}) == len(first)
    known = [entry for entry in first if entry.kind == "known"]
    assert len(known) == 45 * 3 * SMALL_KNOWN
    assert len(first) - len(known) == len(PROBE_VOCABULARY) * SMALL_PROBE
    other_seed = build_plan(known_per_concept=SMALL_KNOWN, probe_per_noun=SMALL_PROBE, base_seed=8)
    assert [entry.prompt for entry in first] != [entry.prompt for entry in other_seed]


def test_prompt_mentions_concept_and_is_reproducible() -> None:
    entry = build_plan(known_per_concept=1, probe_per_noun=0)[0]
    assert entry.concept in entry.prompt
    assert compose_prompt(entry.concept, entry.scale, entry.seed) == entry.prompt


def test_plan_roundtrip_and_errors(tmp_path: Path) -> None:
    entries = build_plan(known_per_concept=1, probe_per_noun=1)
    path = tmp_path / "plan.jsonl"
    write_plan(entries, path)
    assert read_plan(path) == entries
    with pytest.raises(PlanError):
        write_plan([], tmp_path / "empty.jsonl")
    with pytest.raises(PlanError):
        read_plan(tmp_path / "missing.jsonl")
    broken = tmp_path / "gap.jsonl"
    write_plan(entries[1:], broken)  # indices start at 1: not 0..n-1
    with pytest.raises(PlanError):
        read_plan(broken)
    with pytest.raises(PlanError):
        build_plan(known_per_concept=-1)


def test_shards_partition_the_plan() -> None:
    entries = build_plan(known_per_concept=1, probe_per_noun=1)
    shards = [select_shard(entries, shard, 4) for shard in range(4)]
    assert sorted(entry.index for shard in shards for entry in shard) == list(range(len(entries)))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    with pytest.raises(PlanError):
        select_shard(entries, 4, 4)
    with pytest.raises(PlanError):
        select_shard(entries, 0, 0)


def test_split_chunks_and_paths(tmp_path: Path) -> None:
    entries = build_plan(known_per_concept=1, probe_per_noun=0)
    chunks = split_chunks(entries, 50)
    assert sum(map(len, chunks)) == len(entries)
    assert all(len(chunk) == 50 for chunk in chunks[:-1])
    assert chunk_path(tmp_path, 12).name == "chunk_0000012.npz"
    with pytest.raises(GenerationError):
        split_chunks(entries, 0)


def test_resize_gives_task_resolution() -> None:
    image = Image.new("RGB", (512, 512), (10, 200, 30))
    array = to_dataset_resolution(image)
    assert array.shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and array.dtype == np.uint8
    assert tuple(array[0, 0]) == (10, 200, 30)


def test_save_chunk_validates_and_is_atomic(tmp_path: Path) -> None:
    images = np.zeros((2, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    logits = np.zeros((2, NUM_CLASSES), dtype=np.float32)
    target = tmp_path / "chunk.npz"
    save_chunk(target, np.array([4, 5]), images, logits)
    assert target.is_file() and not list(tmp_path.glob("*.tmp"))
    with pytest.raises(GenerationError):
        save_chunk(tmp_path / "bad.npz", np.array([1]), images, logits)
    with pytest.raises(GenerationError):
        save_chunk(
            tmp_path / "nan.npz", np.array([1, 2]), images, np.full((2, NUM_CLASSES), np.nan)
        )
    assert not (tmp_path / "bad.npz").exists() and not (tmp_path / "nan.npz").exists()


class _FakePipeline:
    """Stands in for SDXL-Turbo: solid-colour images, one colour per prompt length."""

    def __call__(
        self, *, prompt: list[str], height: int, width: int, **_: object
    ) -> SimpleNamespace:
        images = [Image.new("RGB", (width, height), (len(text) % 256, 50, 100)) for text in prompt]
        return SimpleNamespace(images=images)


def _fake_classifier() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, NUM_CLASSES))


def _worker_config(
    tmp_path: Path, plan_path: Path, shard: int, **overrides: object
) -> WorkerConfig:
    return WorkerConfig(
        plan_path=plan_path,
        model_dir=tmp_path,
        original_path=tmp_path / "model.pt",
        out_dir=tmp_path / "out",
        shard=shard,
        num_shards=2,
        batch_size=3,
        chunk_size=5,
        device="cpu",
        **overrides,  # type: ignore[arg-type]
    )


def test_worker_writes_chunks_resumes_and_respects_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate_module, "load_generator_pipeline", lambda *_: _FakePipeline())
    monkeypatch.setattr(generate_module, "load_model", lambda *_: _fake_classifier())
    plan_path = tmp_path / "plan.jsonl"
    write_plan(build_plan(known_per_concept=1, probe_per_noun=0)[:24], plan_path)

    first = generate_module.run_worker(_worker_config(tmp_path, plan_path, 0))
    assert first.images_written == 12 and first.chunks_written == 3 and first.chunks_skipped == 0
    again = generate_module.run_worker(_worker_config(tmp_path, plan_path, 0))
    assert again.chunks_written == 0 and again.chunks_skipped == 3, (
        "resume must skip finished chunks"
    )
    assert (tmp_path / "out" / "preview_shard0.png").is_file()

    expired = generate_module.run_worker(_worker_config(tmp_path, plan_path, 1, deadline_epoch=1.0))
    assert expired.stopped_by_deadline and expired.chunks_written == 0

    indices, logits = load_logits(tmp_path / "out")
    assert indices.tolist() == sorted(indices.tolist()) and logits.shape == (12, NUM_CLASSES)


def _stats_fixture() -> tuple[list[object], np.ndarray, np.ndarray]:
    entries = build_plan(known_per_concept=2, probe_per_noun=0)
    probe_entries = build_plan(known_per_concept=0, probe_per_noun=2)
    # Re-index the probe entries after the known ones so the plan stays 0..n-1.
    shifted = [
        entry.model_copy(update={"index": entry.index + len(entries)})
        for entry in probe_entries[:4]
    ]
    plan = [*entries, *shifted]
    indices = np.arange(len(plan))
    logits = np.zeros((len(plan), NUM_CLASSES), dtype=np.float32)
    for entry in plan:
        # Known concepts answer their hint with a huge margin; probe nouns answer class 77.
        logits[entry.index, entry.class_hint if entry.class_hint is not None else 77] = 30.0
    return list(plan), indices, logits


def test_analysis_agreement_and_discovery() -> None:
    plan, indices, logits = _stats_fixture()
    stats = concept_statistics(plan, indices, logits)  # type: ignore[arg-type]
    known = [item for item in stats if item.class_hint is not None]
    assert all(item.hint_agreement == 1.0 and item.mean_confidence > 0.99 for item in known)
    assert discovered_class_map(stats)[77] == [
        item.concept for item in stats if item.class_hint is None
    ]
    summary = summarize(stats)
    assert summary["known_agreement_image_weighted"] == pytest.approx(1.0)
    assert summary["known_classes_reached"] == 45
    assert summary["probe_classes_discovered"] == [77]
    assert 77 not in summary["classes_still_unreached"]  # type: ignore[operator]


def test_analysis_flags_disagreement_and_empty_dir(tmp_path: Path) -> None:
    plan, indices, logits = _stats_fixture()
    wrong = logits.copy()
    wrong[:] = 0.0
    wrong[:, 5] = 30.0  # the original answers class 5 for everything
    stats = concept_statistics(plan, indices, wrong)  # type: ignore[arg-type]
    summary = summarize(stats)
    assert summary["known_agreement_image_weighted"] < 0.1  # type: ignore[operator]
    with pytest.raises(AnalysisError):
        load_logits(tmp_path)


def test_softmax_rows_sum_to_one() -> None:
    probabilities = softmax(np.random.default_rng(0).normal(size=(4, NUM_CLASSES)) * 50)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
