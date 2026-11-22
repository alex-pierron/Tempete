"""Team C: the v4 vote dossier states the frozen verdict from a confirmation result (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hackathon.model_equality.research.c_dossier_v4 import build
from hackathon.model_equality.research.c_rehearsal import REF_ENDPOINTS
from hackathon.model_equality.research.c_v4 import RuleV4, run_confirmation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV = REPO_ROOT / "reports/phase2/model_equality/team-c/data-2/tune_v4b_dev.json"


LORA = {"ref-04", "ref-07", "ref-09"}


def synthetic(prompts: int) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]]]:
    """LoRA endpoints end with a full stop 90 %, ref-03 80 % but disguised, the others 10 %."""
    rng = np.random.default_rng(7)
    arithmetic: dict[str, dict[str, list[str]]] = {}
    identity: dict[str, list[str]] = {}
    for endpoint in REF_ENDPOINTS:
        rate = 0.9 if endpoint in LORA else (0.8 if endpoint == "ref-03" else 0.1)
        arithmetic[endpoint] = {
            f"h-arith-{k:03d}": ["1." if rng.random() < rate else "1" for _ in range(2)]
            for k in range(prompts)
        }
        identity[endpoint] = (["I am Claude by Anthropic."] if endpoint == "ref-03"
                              else ["I am SmolLM by Hugging Face."]) * 30
    return arithmetic, identity


def write_holdout(path: Path, prompts: int) -> None:
    """A synthetic held-out file with `prompts` complete prompts."""
    arithmetic, _ = synthetic(prompts)
    with path.open("w", encoding="utf-8") as handle:
        for endpoint, by_prompt in arithmetic.items():
            for prompt_id, answers in by_prompt.items():
                for repeat, answer in enumerate(answers):
                    handle.write(json.dumps({"endpoint": endpoint, "id": prompt_id, "repeat": repeat,
                                             "response": answer}) + "\n")


def confirmation_json(tmp_path: Path) -> Path:
    """A confirmation result computed on a synthetic held-out file."""
    holdout = tmp_path / "holdout.jsonl"
    write_holdout(holdout, 50)
    identity = tmp_path / "identity.jsonl"
    _, ident = synthetic(1)
    with identity.open("w", encoding="utf-8") as handle:
        for endpoint, answers in ident.items():
            for k, answer in enumerate(answers):
                handle.write(json.dumps({"endpoint": endpoint, "id": f"d-ident-00{k % 3}",
                                         "repeat": k // 3, "response": answer}) + "\n")
    payload = run_confirmation(holdout, identity, RuleV4(0.20, 0.45, 25.0, 15, 3), replicates=60)
    path = tmp_path / "confirmation.json"
    path.write_text(json.dumps(payload))
    return path


def test_dossier_states_the_verdict_costs_and_per_ref_rates(tmp_path: Path) -> None:
    page = build(DEV, confirmation_json(tmp_path), tmp_path / "d.html", "9f04c69", "ab" * 32,
                 "dc" * 32).read_text(encoding="utf-8")
    assert "ELLE ATTEINT 0,60" in page and "9f04c69" in page
    assert "594" in page and "759" in page and "429" in page
    assert "system prompt" in page and "LoRA 0,3" in page
    assert "Aucune reponse ep" not in page or "aucune reponse ep" in page.lower()
    assert (tmp_path / "d.png").is_file()
