"""Team C runner: routing, idempotence and atomic writes, with a fake decoder and image."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hackathon.wmloc.research import c_run_rect_search as runner
from hackathon.wmloc.research.c_rect_search import SearchConfig
from hackathon.wmloc.research.data import Record

N_BITS = 8
RESOLUTION = 16
ROUTE = 0.85


class ConstantDecoder:
    """Decodes every window to the same bits: `agreement` with the message is set by the test."""

    method = "trustmark"
    resolution = RESOLUTION
    device = "cpu"

    def __init__(self, bits: np.ndarray) -> None:
        self._logits = torch.tensor(bits * 2.0 - 1.0, dtype=torch.float32)

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        return self._logits[None].repeat(len(images), 1)


def _record(name: str, full: bool | None = None) -> Record:
    return Record(
        name=name,
        method="trustmark",
        message=np.ones(N_BITS, dtype=np.float32),
        full_watermark=full,
    )


@pytest.fixture(autouse=True)
def fake_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "load_image", lambda split, name: torch.rand(3, 512, 512))


def test_test_split_image_with_high_global_agreement_is_routed_to_full_and_not_searched() -> None:
    decoder = ConstantDecoder(np.ones(N_BITS, dtype=np.float32))  # every decode succeeds
    config = SearchConfig(max_aspect=2.0)
    payload = runner.process_image(decoder, "test", _record("a"), config, ROUTE)
    assert payload["searched"] is False and payload["global_agreement"] == pytest.approx(1.0)
    assert payload["detected"] is False and payload["n_decodes"] == 0


def test_low_global_agreement_triggers_the_search_and_abstains_on_noise() -> None:
    decoder = ConstantDecoder(np.zeros(N_BITS, dtype=np.float32))  # every decode fails
    config = SearchConfig(max_aspect=2.0, size_ratio=3.0, stride_fraction=1.0, min_side=128)
    payload = runner.process_image(decoder, "test", _record("b"), config, ROUTE)
    assert payload["searched"] is True and payload["detected"] is False
    assert payload["n_decodes"] > 0 and payload["box"] is None


def test_train_images_are_always_searched_whatever_the_global_agreement() -> None:
    decoder = ConstantDecoder(np.ones(N_BITS, dtype=np.float32))
    config = SearchConfig(max_aspect=2.0, size_ratio=3.0, stride_fraction=1.0, min_side=256)
    payload = runner.process_image(decoder, "train", _record("c", full=False), config, ROUTE)
    assert payload["searched"] is True


def test_undetected_names_selects_searched_images_without_detection(tmp_path: Path) -> None:
    payloads = {
        "found": {"name": "found", "searched": True, "detected": True},
        "missed": {"name": "missed", "searched": True, "detected": False},
        "full": {"name": "full", "searched": False, "detected": False},
    }
    for name, payload in payloads.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    assert runner.undetected_names(tmp_path) == {"missed"}


def test_undetected_names_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(runner.RunError):
        runner.undetected_names(tmp_path)


def test_select_records_full_regime_needs_train_labels() -> None:
    with pytest.raises(runner.RunError):
        runner.select_records("test", "trustmark", runner.REGIME_FULL, None, None)


def test_select_records_rect_regime_needs_train_labels() -> None:
    with pytest.raises(runner.RunError):
        runner.select_records("test", "trustmark", runner.REGIME_RECT, None, None)
