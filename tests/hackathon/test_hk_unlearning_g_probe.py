"""Team G: shape report, generated-proxy loader, drift probes and the team C export (tiny ViT, CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from test_hk_unlearning_support import random_images, tiny_model

from hackathon.unlearning.research import g_probe
from hackathon.unlearning.research.g_probe import (
    EXPORT_VIEWS,
    build_probe_set,
    distribution_report,
    export_subset_indices,
    export_views,
    load_generated_images,
    measure_probe,
)

CLASSES = 100
SUBSET = 4


def test_distribution_report_on_a_one_hot_and_on_a_flat_output() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    sharp = torch.full((4, CLASSES), -50.0)
    sharp[torch.arange(4), labels] = 50.0
    report = distribution_report(sharp, labels, saturated=[], siblings={})
    assert report["loss_mean"] == pytest.approx(0.0, abs=1e-6)
    assert report["accuracy_vs_pseudo_label"] == 1.0
    assert report["entropy_t1"] == pytest.approx(0.0, abs=1e-6)
    flat = distribution_report(torch.zeros(4, CLASSES), labels, saturated=[], siblings={})
    assert flat["entropy_t2"] == pytest.approx(float(np.log(CLASSES)), abs=1e-5)
    assert flat["tail_mass_t2"] == pytest.approx(0.95, abs=1e-6)


def test_sibling_fallback_counts_saturated_images_sent_to_a_present_sibling() -> None:
    labels = torch.tensor([0, 0, 5])
    logits = torch.zeros(3, CLASSES)
    logits[0, 0] = 10.0
    logits[0, 1] = 5.0
    logits[1, 0] = 10.0
    logits[1, 9] = 5.0
    logits[2, 5] = 10.0
    report = distribution_report(logits, labels, saturated=[0], siblings={0: [1, 2]})
    assert report["sibling_fallback"] == pytest.approx(0.5)


def test_generated_loader_reads_images_only_and_respects_count_and_skip(tmp_path: Path) -> None:
    for index in range(3):
        np.savez(
            tmp_path / f"chunk_{index:07d}.npz",
            images=np.full((2, 224, 224, 3), index, dtype=np.uint8),
            logits=np.zeros((2, CLASSES), dtype=np.float32),
        )
    loaded = load_generated_images(tmp_path, 3)
    assert loaded.shape == (3, 224, 224, 3) and int(loaded[2, 0, 0, 0]) == 1
    skipped = load_generated_images(tmp_path, 2, skip_chunks=2)
    assert skipped.shape[0] == 2 and int(skipped[0, 0, 0, 0]) == 2
    with pytest.raises(FileNotFoundError):
        load_generated_images(tmp_path / "missing", 1)


def test_probe_of_the_original_reports_zero_drift_and_positive_fidelity() -> None:
    teacher = tiny_model()
    forget_all = random_images(6, seed=21)
    targets = torch.softmax(torch.randn(6, CLASSES, generator=torch.Generator().manual_seed(1)), 1)
    probe = build_probe_set(
        teacher, forget_all, targets, torch.tensor([0, 2, 4]), random_images(2, seed=22),
        random_images(2, seed=23), saturated=[3], siblings={3: [4]},
    )  # fmt: skip
    metrics = measure_probe(teacher, probe, processed=torch.tensor([True, False, False]))
    assert metrics["drift_generated_kl_t2"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["drift_ood_kl_t2"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["forget_kl_t2"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["fidelity_mean"] > 0.0
    assert "fidelity_processed" in metrics and "fidelity_unprocessed" in metrics
    assert "altered_strong_loss_mean" in metrics and "altered_crop_kl_t2" in metrics


def test_export_writes_the_five_views_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(g_probe, "EXPORT_SUBSET", SUBSET)
    model = tiny_model()
    forget_all = random_images(6, seed=31)
    path = tmp_path / "model.npz"
    first = export_views(model, forget_all, path)
    assert set(first) == {f"logits_{view}" for view in EXPORT_VIEWS}
    with np.load(path) as payload:
        assert payload["logits_exact"].shape == (SUBSET, CLASSES)
        assert payload["feats_strong_view"].dtype == np.float16
        assert payload["subset_indices"].tolist() == export_subset_indices(6).tolist()
    stamp = path.stat().st_mtime_ns
    again = export_views(model, forget_all, path)
    assert path.stat().st_mtime_ns == stamp
    assert np.allclose(first["logits_crop"], again["logits_crop"])
