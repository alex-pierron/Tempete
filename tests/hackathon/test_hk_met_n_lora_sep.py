"""Team next: LoRA separation at a fixed number of tokens, on synthetic likelihood files (no GPU)."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hackathon.model_equality.research.n_lora_sep import (
    LORA_SLOTS,
    NULL_LABEL,
    AnswerRows,
    LoraSepError,
    family_rows,
    find_ll_file,
    load_base_rows,
    main,
    mean_ll_draws,
    tokens_for_power,
)

FAMILY = "qwen3"
ROWS_PER_ENDPOINT = 600
TOKENS_PER_ROW = 5
PLAIN_MEAN = -1.0
NOISE = 0.5
SHIFT_OF_SLOT = {1: 0.0, 9: -0.01, 4: -0.1, 7: -0.5}
SEED = 7


def _scores(slots: dict[int, float]) -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED)
    table: dict[str, list[float]] = {}
    for slot, shift in slots.items():
        per_token = rng.normal(PLAIN_MEAN + shift, NOISE, size=ROWS_PER_ENDPOINT)
        for index, value in enumerate(per_token):
            table[f"zoo-{FAMILY}-{slot:02d}|p-{index:03d}|0"] = [float(value) * TOKENS_PER_ROW,
                                                               TOKENS_PER_ROW]
    table["zoo-llama1-01|p-000|0"] = [-3.0, TOKENS_PER_ROW]  # another family, to be skipped
    return table


def _write(directory: Path, slots: dict[int, float], *, gz: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"scorer": FAMILY, "scores": {"base": _scores(slots)}}).encode()
    name = f"ll_{FAMILY}.json.gz" if gz else f"ll_{FAMILY}.json"
    (directory / name).write_bytes(gzip.compress(payload) if gz else payload)


def test_loader_reads_gzip_and_plain_and_skips_other_families(tmp_path: Path) -> None:
    _write(tmp_path / "gz", SHIFT_OF_SLOT, gz=True)
    _write(tmp_path / "plain", SHIFT_OF_SLOT, gz=False)
    for folder in ("gz", "plain"):
        rows = load_base_rows(find_ll_file(tmp_path / folder, FAMILY), FAMILY)
        assert sorted(rows) == [f"zoo-{FAMILY}-{slot:02d}" for slot in sorted(SHIFT_OF_SLOT)]
        assert all(r.n_tokens.size == ROWS_PER_ENDPOINT for r in rows.values())
    with pytest.raises(LoraSepError):
        find_ll_file(tmp_path / "gz", "smollm")


def test_draws_reach_the_budget_and_noise_falls_with_tokens() -> None:
    rng = np.random.default_rng(SEED)
    rows = AnswerRows(np.full(ROWS_PER_ENDPOINT, TOKENS_PER_ROW, dtype=np.int64),
                      rng.normal(PLAIN_MEAN, NOISE, ROWS_PER_ENDPOINT) * TOKENS_PER_ROW)
    small = mean_ll_draws(rows, 500, 2000, rng)
    large = mean_ll_draws(rows, 8000, 2000, rng)  # above the 3 000 tokens of the endpoint
    assert abs(float(small.mean()) - float(rows.logp.sum() / rows.n_tokens.sum())) < 0.02
    assert float(small.std()) / float(large.std()) == pytest.approx(4.0, rel=0.15)
    with pytest.raises(LoraSepError):
        mean_ll_draws(AnswerRows(np.array([], dtype=np.int64), np.array([])), 500, 10, rng)


def test_tokens_for_power_scales_with_the_inverse_square_of_the_gap() -> None:
    assert math.isinf(tokens_for_power(0.0, 0.1, 0.1, 1000))
    wide = tokens_for_power(0.2, 0.1, 0.1, 1000)
    narrow = tokens_for_power(0.1, 0.1, 0.1, 1000)
    assert narrow == pytest.approx(4.0 * wide)


def test_strong_lora_separates_weak_lora_does_not_and_null_stays_at_chance(tmp_path: Path) -> None:
    _write(tmp_path, SHIFT_OF_SLOT, gz=True)
    rows = family_rows(FAMILY, tmp_path, None, "synthetic", [500, 4000], draws=400)
    by_key = {(r["variant"], r["tokens"]): r for r in rows}
    assert {v for v, _ in by_key} =={NULL_LABEL, *LORA_SLOTS}
    assert float(by_key[("lora_1.5", 500)]["tpr_at_5pct_fpr"]) > 0.99
    assert float(by_key[(NULL_LABEL, 4000)]["auc"]) < 0.6
    weak_small = float(by_key[("lora_0.7", 500)]["tpr_at_5pct_fpr"])
    weak_large = float(by_key[("lora_0.7", 4000)]["tpr_at_5pct_fpr"])
    assert weak_large > weak_small  # more tokens help a real gap


def test_overlay_takes_lora_from_the_other_run_and_plain_from_the_base(tmp_path: Path) -> None:
    _write(tmp_path / "base", SHIFT_OF_SLOT, gz=True)
    _write(tmp_path / "overlay", {9: -0.5, 4: -0.5, 7: -0.5}, gz=False)  # no plain endpoint
    rows = family_rows(FAMILY, tmp_path / "base", tmp_path / "overlay", "dolly", [500], draws=200)
    weakest = next(r for r in rows if r["variant"] == "lora_0.3")
    assert weakest["adapter"] == "dolly" and float(weakest["tpr_at_5pct_fpr"]) > 0.99
    _write(tmp_path / "broken", {9: -0.5}, gz=False)
    with pytest.raises(LoraSepError, match="missing endpoints"):
        family_rows(FAMILY, tmp_path / "base", tmp_path / "broken", "dolly", [500], draws=50)


def test_cli_writes_the_table(tmp_path: Path) -> None:
    _write(tmp_path, SHIFT_OF_SLOT, gz=True)
    out = tmp_path / "out" / "sep.csv"
    assert main(["--zoo-dir", str(tmp_path), "--families", FAMILY, "--budgets", "500",
                 "--draws", "50", "--out", str(out)]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1 + 1 + len(LORA_SLOTS)
