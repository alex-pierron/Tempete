"""Tests of the submission builder and validator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from hackathon.common.errors import SubmissionInvalidError
from hackathon.wmloc.constants import GRID_SIZE, MAX_SUBMISSION_BYTES
from hackathon.wmloc.submission import (
    build_submission,
    save_submission,
    validate_submission,
    validate_submission_file,
)

NAMES = ["a.png", "b.png", "c.png"]


def masks(n: int = 3, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    return torch.randint(0, 2, (n, GRID_SIZE, GRID_SIZE)).to(dtype)


def good() -> dict[str, Any]:
    return {"image_names": list(NAMES), "masks": masks()}


def test_valid_submission_passes_and_reports() -> None:
    report = validate_submission(good(), expected_names=NAMES)
    assert report.n_images == 3
    assert report.shape == (3, GRID_SIZE, GRID_SIZE)
    assert report.warnings == ()
    assert 0.0 < report.watermarked_fraction < 1.0


@pytest.mark.parametrize("bad_keys", [{"image_names": NAMES}, {"names": NAMES, "masks": 1}])
def test_wrong_keys_are_rejected(bad_keys: dict[str, Any]) -> None:
    with pytest.raises(SubmissionInvalidError, match="keys"):
        validate_submission(bad_keys)


def test_extra_key_is_rejected() -> None:
    submission = good()
    submission["note"] = "x"
    with pytest.raises(SubmissionInvalidError, match="keys"):
        validate_submission(submission)


def test_not_a_dict_is_rejected() -> None:
    with pytest.raises(SubmissionInvalidError):
        validate_submission([1, 2])


@pytest.mark.parametrize("names", [tuple(NAMES), ["a", 2, "c"], "abc"])
def test_names_must_be_a_list_of_str(names: Any) -> None:
    with pytest.raises(SubmissionInvalidError, match="list"):
        validate_submission({"image_names": names, "masks": masks()})


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(SubmissionInvalidError, match="duplicate"):
        validate_submission({"image_names": ["a", "a", "b"], "masks": masks()})


def test_expected_set_mismatch_lists_missing_and_unknown() -> None:
    with pytest.raises(SubmissionInvalidError, match="1 missing.*1 unknown"):
        validate_submission(good(), expected_names=["a.png", "b.png", "zzz.png"])


def test_shape_must_match_names_and_grid() -> None:
    with pytest.raises(SubmissionInvalidError, match="shape"):
        validate_submission({"image_names": NAMES, "masks": masks(2)})
    with pytest.raises(SubmissionInvalidError, match="shape"):
        validate_submission(
            {"image_names": NAMES, "masks": torch.zeros(3, 64, 64, dtype=torch.uint8)}
        )


def test_values_must_be_zero_or_one_and_not_nan() -> None:
    bad = masks()
    bad[0, 0, 0] = 2
    with pytest.raises(SubmissionInvalidError, match="0 and 1"):
        validate_submission({"image_names": NAMES, "masks": bad})
    nan = masks(dtype=torch.float32)
    nan[0, 0, 0] = float("nan")
    with pytest.raises(SubmissionInvalidError, match="NaN"):
        validate_submission({"image_names": NAMES, "masks": nan})


def test_masks_must_be_a_tensor() -> None:
    with pytest.raises(SubmissionInvalidError, match="Tensor"):
        validate_submission({"image_names": NAMES, "masks": [[0]]})


def test_float_dtype_is_valid_but_warned() -> None:
    report = validate_submission({"image_names": NAMES, "masks": masks(dtype=torch.float32)})
    assert any("compact" in warning for warning in report.warnings)


def test_full_test_set_in_float32_is_rejected_for_size_and_uint8_fits() -> None:
    names = [f"img_{i}.png" for i in range(1020)]
    heavy = {"image_names": names, "masks": torch.zeros(1020, GRID_SIZE, GRID_SIZE)}
    with pytest.raises(SubmissionInvalidError, match="byte limit"):
        validate_submission(heavy)
    light = build_submission(names, torch.zeros(1020, GRID_SIZE, GRID_SIZE))
    assert light["masks"].dtype == torch.uint8
    report = validate_submission(light)
    assert report.n_bytes < MAX_SUBMISSION_BYTES


def test_build_never_rounds_silently() -> None:
    with pytest.raises(SubmissionInvalidError, match="0 and 1"):
        build_submission(NAMES, torch.full((3, GRID_SIZE, GRID_SIZE), 0.7))


def test_bool_masks_are_stored_as_uint8() -> None:
    submission = build_submission(NAMES, masks(dtype=torch.bool))
    assert submission["masks"].dtype == torch.uint8


def test_save_and_validate_file_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "out" / "sub.pt"
    original = masks()
    report = save_submission(target, NAMES, original, expected_names=NAMES)
    assert target.is_file()
    assert not target.with_name("sub.pt.part").exists()
    assert report.n_bytes == target.stat().st_size
    again = validate_submission_file(target, NAMES)
    assert again.n_images == 3
    loaded = torch.load(target, weights_only=True)
    assert loaded["image_names"] == NAMES
    assert torch.equal(loaded["masks"], original)


def test_save_refuses_invalid_content_and_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "sub.pt"
    with pytest.raises(SubmissionInvalidError):
        save_submission(target, ["a", "a", "b"], masks())
    assert not target.exists()


def test_file_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(SubmissionInvalidError, match="not found"):
        validate_submission_file(tmp_path / "missing.pt")
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch file")
    with pytest.raises(SubmissionInvalidError, match="cannot load"):
        validate_submission_file(corrupt)
