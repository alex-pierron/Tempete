"""Pure parts of the simulated-endpoint protocol of team E (no model, no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hackathon.model_equality.research import e_sim as sim


def _line(endpoint: str, pid: str, prompt: str) -> str:
    return (
        f'{{"endpoint": "{endpoint}", "id": "{pid}", "repeat": 0, '
        f'"prompt": "{prompt}", "response": "r"}}\n'
    )


def test_cut_at_eos_stops_at_the_first_end_id() -> None:
    assert sim.cut_at_eos([5, 6, 99, 7, 99], [99, 100]) == [5, 6]
    assert sim.cut_at_eos([5, 6, 7], [99]) == [5, 6, 7]
    assert sim.cut_at_eos([99, 1], [99]) == []
    assert sim.cut_at_eos([], [99]) == []


def test_collect_prompts_keeps_ep_prompts_first_and_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "answers.jsonl").write_text(
        _line("ep-01", "a", "ep prompt one") + _line("ep-02", "b", "ep prompt two")
    )
    (tmp_path / "reference_answers.jsonl").write_text(
        "".join(_line("ref-01", f"r{i}", f"ref prompt {i}") for i in range(20))
    )
    (tmp_path / "reference_answers_holdout_raw.jsonl").write_text(
        '{"endpoint": "ref-01", "no": "response"}\n'
    )
    first = sim.collect_prompts(tmp_path, 8, seed=1)
    second = sim.collect_prompts(tmp_path, 8, seed=1)
    other_seed = sim.collect_prompts(tmp_path, 8, seed=2)
    assert first == second
    assert first[:2] == ["ep prompt one", "ep prompt two"]
    assert len(first) == 8 and len(set(first)) == 8
    assert first != other_seed
    assert sim.collect_prompts(tmp_path, 1, seed=1) == ["ep prompt one", "ep prompt two"]


def test_collect_prompts_without_answers_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(sim.SimError):
        sim.collect_prompts(tmp_path, 5, seed=1)


def test_load_sim_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sim_toy.npz"
    arrays: dict[str, Any] = {
        "meta": np.array('{"candidate": "toy"}'),
        "logp__base__base": np.array([-1.0, -2.0]),
        "logp__base__w_tensor": np.array([-1.1, -2.1]),
        "logp__w_tensor__w_tensor": np.array([-0.9, -1.9]),
        "tokens__base": np.array([3, 4]),
        "tokens__w_tensor": np.array([3, 5]),
        "texts__base": np.array(["a", "b"]),
    }
    np.savez(path, **arrays)
    meta, sums, tokens = sim.load_sim(path)
    assert meta == {"candidate": "toy"}
    assert set(sums) == {("base", "base"), ("base", "w_tensor"), ("w_tensor", "w_tensor")}
    assert sums[("base", "w_tensor")].tolist() == [-1.1, -2.1]
    assert tokens["w_tensor"].tolist() == [3, 5]
