"""Tests of the x-loc-unet evaluation helpers: pure parts, no release data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hackathon.wmloc.research import x_unet_tools


@pytest.mark.parametrize(
    ("train_key", "test_key"),
    [
        ("train_finetuned_oof", "test_finetuned"),
        ("train_finetuned_oof_b", "test_finetuned_b"),
        ("train_synthetic", "test_synthetic"),
    ],
)
def test_test_key_of_follows_small_train_naming(train_key: str, test_key: str) -> None:
    assert x_unet_tools.test_key_of(train_key) == test_key


def test_test_key_of_refuses_a_test_key() -> None:
    with pytest.raises(x_unet_tools.UnetToolsError):
        x_unet_tools.test_key_of("test_finetuned")


def test_parse_spec_splits_on_the_last_colon() -> None:
    assert x_unet_tools.parse_spec("/a/b:c.npz:train_finetuned_oof") == (
        Path("/a/b:c.npz"),
        "train_finetuned_oof",
    )
    with pytest.raises(x_unet_tools.UnetToolsError):
        x_unet_tools.parse_spec("only_a_path.npz")


def test_split_then_mean_of_specs(tmp_path: Path) -> None:
    merged = tmp_path / "merged.npz"
    names = np.array(["a", "b"])
    np.savez(
        merged,
        trustmark__names_train=names,
        trustmark__names_test=names,
        trustmark__train_finetuned_oof=np.full((2, 4, 4), 0.2, dtype=np.float16),
        trustmark__test_finetuned=np.full((2, 4, 4), 0.4, dtype=np.float16),
        pixelseal__names_train=names,
        unrelated=np.zeros(1),
    )
    written = x_unet_tools.split_container(merged, tmp_path / "out", "x3")
    assert [p.name for p in written] == ["x3_pixelseal.npz", "x3_trustmark.npz"]
    spec = (tmp_path / "out" / "x3_trustmark.npz", "train_finetuned_oof")
    train_names, train_mean = x_unet_tools.mean_of_specs([spec, spec], test=False)
    test_names, test_mean = x_unet_tools.mean_of_specs([spec], test=True)
    assert train_names == ["a", "b"] and test_names == ["a", "b"]
    np.testing.assert_allclose(train_mean, 0.2, atol=1e-3)
    np.testing.assert_allclose(test_mean, 0.4, atol=1e-3)
    with pytest.raises(x_unet_tools.UnetToolsError):
        x_unet_tools.mean_of_specs([(spec[0], "train_finetuned_oof_b")], test=False)


def test_split_refuses_a_container_without_method_keys(tmp_path: Path) -> None:
    plain = tmp_path / "plain.npz"
    np.savez(plain, names_train=np.array(["a"]))
    with pytest.raises(x_unet_tools.UnetToolsError):
        x_unet_tools.split_container(plain, tmp_path / "out", "x")


def test_regime_binarise_uses_one_threshold_per_guessed_regime() -> None:
    full = np.full((1, 4, 4), 0.45, dtype=np.float32)
    full[0, 0, 0] = 0.9  # guessed regime: 1 cell of 16 >= 0.5 -> rect
    mostly = np.full((1, 4, 4), 0.6, dtype=np.float32)
    mostly[0, 0, 0] = 0.4  # guessed regime: full
    maps = np.concatenate([full, mostly])
    out = x_unet_tools.regime_binarise(maps, (0.5, 0.3))
    assert out[0].sum() == 16  # rect threshold 0.3: everything at 0.45 and 0.9 is kept
    assert out[1].sum() == 15  # full threshold 0.5: the 0.4 cell is dropped
