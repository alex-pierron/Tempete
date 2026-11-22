"""Tests of the stage-2 trainer (soft forget targets + generated-proxy anchor), tiny ViT on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hackathon.unlearning.constants import IMAGE_SIZE, NUM_CLASSES
from hackathon.unlearning.model import build_model
from hackathon.unlearning.research.retain_proxy.concepts import FULLY_FORGOTTEN_CLASSES
from hackathon.unlearning.research.retain_proxy.distill_f import (
    DistillConfig,
    altered_copies,
    build_targets,
    matching_loss,
    remove_classes,
    run_distill,
    temper,
)

FLOOR = 1e-4


def _tiny() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_model(embed_dim=32, depth=2, num_heads=2)


def _images(count: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(
        rng.integers(0, 255, (count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    )


def _config(tmp_path: Path, **overrides: object) -> DistillConfig:
    dataset = tmp_path / "proxy.npz"
    np.savez(dataset, images=_images(12, 1).numpy(), labels=np.arange(12, dtype=np.int64) % 5 + 1)
    base: dict[str, object] = {
        "name": "tiny",
        "dataset_path": dataset,
        "original_path": tmp_path / "unused.pt",
        "forget_path": tmp_path / "unused.npz",
        "out_dir": tmp_path / "out",
        "target_views": 2,
        "altered_copies": 1,
        "epochs": 2,
        "steps_per_epoch": 2,
        "forget_batch": 4,
        "proxy_batch": 4,
        "start_micro_batch": 3,
        "holdout_fraction": 0.25,
        "lr": 1e-3,
        "device": "cpu",
    }
    return DistillConfig(**{**base, **overrides})  # type: ignore[arg-type]


def test_remove_classes_keeps_rows_normalised_with_the_floor() -> None:
    target = torch.softmax(torch.randn(5, NUM_CLASSES) * 3, dim=1)
    out = remove_classes(target, FULLY_FORGOTTEN_CLASSES, FLOOR)
    assert torch.allclose(out.sum(dim=1), torch.ones(5), atol=1e-5)
    assert torch.allclose(out[:, list(FULLY_FORGOTTEN_CLASSES)], torch.full((5, 9), FLOOR))


def test_temper_flattens_and_keeps_the_ranking() -> None:
    probabilities = torch.softmax(torch.tensor([[6.0, 3.0, 0.0]]), dim=1)
    tempered = temper(probabilities, 2.0)
    assert tempered.max() < probabilities.max()
    assert torch.equal(tempered.argsort(), probabilities.argsort())


def test_matching_loss_is_zero_only_when_the_student_matches() -> None:
    logits = torch.randn(4, NUM_CLASSES)
    target = torch.softmax(logits, dim=1)
    assert float(matching_loss(target, logits)) == pytest.approx(0.0, abs=1e-5)
    assert float(matching_loss(target, torch.zeros(4, NUM_CLASSES))) > 0.1


def test_build_targets_are_distributions_and_honour_the_floor(tmp_path: Path) -> None:
    forget = _images(6, 2)
    config = _config(tmp_path, removal_floor=FLOOR)
    targets = build_targets(_tiny(), forget, config, torch.device("cpu"))
    assert targets.shape == (6, NUM_CLASSES)
    assert torch.allclose(targets.sum(dim=1), torch.ones(6), atol=1e-4)
    assert torch.allclose(targets[:, list(FULLY_FORGOTTEN_CLASSES)], torch.full((6, 9), FLOOR))
    plain = build_targets(_tiny(), forget, _config(tmp_path), torch.device("cpu"))
    assert torch.equal(
        plain, build_targets(_tiny(), forget, _config(tmp_path), torch.device("cpu"))
    )


def test_altered_copies_are_fixed_and_distinct() -> None:
    forget = _images(5, 3)
    first, second = altered_copies(forget, 2, seed=0), altered_copies(forget, 2, seed=0)
    assert len(first) == 2 and all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert first[0].shape == forget.shape and not torch.equal(first[0], forget)


def test_run_distill_end_to_end_with_teacher_and_exports(tmp_path: Path) -> None:
    base = _tiny()
    before = {key: value.clone() for key, value in base.state_dict().items()}
    teacher = build_model(embed_dim=32, depth=2, num_heads=2)
    config = _config(
        tmp_path,
        removal_floor=FLOOR,
        save_model=True,
        export_logits_dir=tmp_path / "team_logits",
        export_alphas=(1.0,),
        export_subset_size=4,
    )
    result = run_distill(config, original=base, teacher=teacher, forget_images=_images(6, 4))
    history = result["history"]
    assert isinstance(history, list) and len(history) == 2
    assert all("holdout" in entry and "forget" in entry for entry in history)
    run_dir = tmp_path / "out" / "tiny"
    assert (run_dir / "model.pt").is_file() and (run_dir / "blends.json").is_file()
    assert (tmp_path / "team_logits" / "d_tiny_a1.00.npz").is_file()
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items()), "original intact"
    trained = torch.load(run_dir / "model.pt", weights_only=True)
    assert any(not torch.equal(before[key], trained[key]) for key in before)
