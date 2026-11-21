"""Tests of the submission validator: every server rule has a test that breaks it."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_hk_unlearning_support import tiny_factory, tiny_model

from hackathon.common.errors import SubmissionInvalidError
from hackathon.unlearning import validate as validate_module
from hackathon.unlearning.validate import check_state_dict, main, validate_submission


def _write(path: Path, state: object) -> Path:
    torch.save(state, path)
    return path


def _good_state() -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in tiny_model().state_dict().items()}


def test_valid_file_passes_and_reports_counts(tmp_path: Path) -> None:
    path = _write(tmp_path / "ok.pt", _good_state())
    report = validate_submission(path, tiny_factory)
    assert report.tensor_count == len(_good_state())
    assert report.parameter_count == sum(int(v.numel()) for v in _good_state().values())
    assert report.size_bytes == path.stat().st_size


def test_pth_suffix_is_accepted(tmp_path: Path) -> None:
    validate_submission(_write(tmp_path / "ok.pth", _good_state()), tiny_factory)


@pytest.mark.parametrize("name", ["bad.bin", "bad.pt.txt", "bad"])
def test_wrong_suffix_is_rejected(tmp_path: Path, name: str) -> None:
    path = _write(tmp_path / name, _good_state())
    with pytest.raises(SubmissionInvalidError, match="expected a"):
        validate_submission(path, tiny_factory)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SubmissionInvalidError, match="not found"):
        validate_submission(tmp_path / "nope.pt", tiny_factory)


def test_oversize_file_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "big.pt", _good_state())
    monkeypatch.setattr(validate_module, "MAX_SUBMISSION_BYTES", path.stat().st_size - 1)
    with pytest.raises(SubmissionInvalidError, match="exceeds"):
        validate_submission(path, tiny_factory)


def test_float16_weights_are_rejected_and_named(tmp_path: Path) -> None:
    state = _good_state()
    key = next(k for k, v in state.items() if v.is_floating_point())
    state[key] = state[key].half()
    with pytest.raises(SubmissionInvalidError, match=f"float32.*{key}"):
        validate_submission(_write(tmp_path / "half.pt", state), tiny_factory)


def test_bfloat16_and_float64_are_rejected(tmp_path: Path) -> None:
    for dtype in (torch.bfloat16, torch.float64):
        state = {k: v.to(dtype) for k, v in _good_state().items()}
        with pytest.raises(SubmissionInvalidError, match="float32"):
            validate_submission(_write(tmp_path / f"{dtype}.pt", state), tiny_factory)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weights_are_rejected(tmp_path: Path, bad_value: float) -> None:
    state = _good_state()
    key = next(k for k, v in state.items() if v.is_floating_point() and v.numel() > 1)
    state[key].view(-1)[0] = bad_value
    with pytest.raises(SubmissionInvalidError, match=f"non-finite.*{key}"):
        validate_submission(_write(tmp_path / "nan.pt", state), tiny_factory)


def test_missing_and_unexpected_keys_are_named(tmp_path: Path) -> None:
    state = _good_state()
    removed = next(iter(state))
    del state[removed]
    state["adapter.lora_A"] = torch.zeros(2, 2)
    with pytest.raises(SubmissionInvalidError) as caught:
        validate_submission(_write(tmp_path / "keys.pt", state), tiny_factory)
    message = str(caught.value)
    assert removed in message and "adapter.lora_A" in message


def test_wrong_shape_is_named(tmp_path: Path) -> None:
    state = _good_state()
    state["head.weight"] = torch.zeros(7, 32)
    with pytest.raises(SubmissionInvalidError, match="head.weight.*shape"):
        validate_submission(_write(tmp_path / "shape.pt", state), tiny_factory)


def test_a_pickled_module_is_not_a_state_dict(tmp_path: Path) -> None:
    path = _write(tmp_path / "module.pt", tiny_model())
    with pytest.raises(SubmissionInvalidError, match="weights-only"):
        validate_submission(path, tiny_factory)


def test_a_non_dict_payload_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SubmissionInvalidError, match="plain state dict"):
        validate_submission(_write(tmp_path / "list.pt", [torch.zeros(1)]), tiny_factory)


def test_garbage_bytes_are_rejected_without_a_raw_traceback(tmp_path: Path) -> None:
    path = tmp_path / "garbage.pt"
    path.write_bytes(b"this is not a torch file")
    with pytest.raises(SubmissionInvalidError, match="weights-only"):
        validate_submission(path, tiny_factory)


def test_many_problems_are_summarised(tmp_path: Path) -> None:
    state = {k: v.half() if v.is_floating_point() else v for k, v in _good_state().items()}
    with pytest.raises(SubmissionInvalidError, match="more"):
        validate_submission(_write(tmp_path / "many.pt", state), tiny_factory)


def test_check_state_dict_accepts_matching_reference() -> None:
    reference = {k: tuple(v.shape) for k, v in tiny_model().state_dict().items()}
    check_state_dict(_good_state(), reference)


def test_cli_returns_one_and_prints_invalid_for_a_tiny_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The CLI validates against the REAL architecture, so a tiny model must be refused.
    path = _write(tmp_path / "tiny.pt", _good_state())
    assert main([str(path)]) == 1
    assert "INVALID" in capsys.readouterr().err
