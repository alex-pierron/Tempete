"""Deliverables of the zoo for team C: official-format answers and the truth, with checksums.

* `zoo_answers.jsonl` (unless `combined=False`): every endpoint of every family
  (`zoo-<family>-<slot>`), official line format;
* `zoo_<family>_as_ref.jsonl`: one family with its nine endpoints renamed `ref-01 .. ref-09` (the
  slot IS the ref number), so the single-family code of `c_rehearsal` / `c_identity` runs on it
  unchanged; the W8A8 bound (slot 10) is left out;
* `zoo_truth.json`: endpoint -> family, slot, variant, true model;
* `manifest.json`: row counts and SHA-256 of every file written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hackathon.model_equality.zoo.config import (
    BASE_SLOTS,
    FAMILIES,
    parse_endpoint,
    variant_for_slot,
)
from hackathon.model_equality.zoo.store import read_rows

REF_PREFIX = "ref"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_for_team_c(source: Path, destination: Path, combined: bool = True
                      ) -> dict[str, object]:
    """Write the deliverables under `destination`; return the manifest.

    Raises:
        FileNotFoundError: no `zoo_<family>.jsonl` in `source`.
    """
    destination.mkdir(parents=True, exist_ok=True)
    per_family: dict[str, list[dict[str, object]]] = {}
    for family in FAMILIES:
        path = source / f"zoo_{family.key}.jsonl"
        if path.is_file():
            per_family[family.key] = read_rows(path)
    if not per_family:
        raise FileNotFoundError(f"no zoo_<family>.jsonl in {source}")
    manifest: dict[str, object] = {"files": {}}
    files: dict[str, dict[str, object]] = {}
    if combined:
        everything = [row for rows in per_family.values() for row in rows]
        files["zoo_answers.jsonl"] = {"rows": len(everything), "sha256": _write_jsonl(
            destination / "zoo_answers.jsonl", everything)}
    truth: dict[str, dict[str, object]] = {}
    for key, rows in per_family.items():
        endpoints = sorted({str(r["endpoint"]) for r in rows})
        for endpoint in endpoints:
            _, slot = parse_endpoint(endpoint)
            variant = variant_for_slot(slot)
            truth[endpoint] = {
                "family": key, "slot": slot, "variant": variant.key, "model": variant.truth_model,
                "ref_equivalent": f"{REF_PREFIX}-{slot:02d}" if slot in BASE_SLOTS else None,
            }
        renamed = []
        for row in rows:
            _, slot = parse_endpoint(str(row["endpoint"]))
            if slot in BASE_SLOTS:
                renamed.append({**row, "endpoint": f"{REF_PREFIX}-{slot:02d}"})
        name = f"zoo_{key}_as_ref.jsonl"
        files[name] = {"rows": len(renamed), "sha256": _write_jsonl(destination / name, renamed)}
    (destination / "zoo_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    manifest["files"] = files
    manifest["families"] = list(per_family)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
