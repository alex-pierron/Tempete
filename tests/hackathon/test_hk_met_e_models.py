"""Candidate registry and download manifest of team E (no network: the listing function is injected)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hackathon.model_equality.research import e_models as m

SHA_A = "a" * 64
SHA_B = "b" * 64


def _listing(shard_sha: str | None, size: int = 100) -> dict[str, m.RepoFile]:
    return {
        "model.safetensors": m.RepoFile("model.safetensors", size, shard_sha),
        "config.json": m.RepoFile("config.json", 10, None),
        "tokenizer.json": m.RepoFile("tokenizer.json", 20, None),
        "README.md": m.RepoFile("README.md", 5, None),
    }


def test_registry_keys_unique_and_validation_candidate_present() -> None:
    keys = [spec.key for spec in m.CANDIDATES]
    assert len(keys) == len(set(keys))
    assert m.VALIDATION_CANDIDATE in keys
    assert m.get_candidate("qwen2.5-7b").family == "qwen2.5"


def test_unknown_candidate_is_a_typed_error() -> None:
    with pytest.raises(m.ModelManifestError, match="unknown candidate"):
        m.get_candidate("nope")


def test_registry_covers_the_families_of_the_brief() -> None:
    families = {spec.family for spec in m.CANDIDATES if spec.tier == "core"}
    assert {"qwen2.5", "llama3.1", "llama3.2", "llama3", "smollm2"} <= families
    qwen_sizes = sorted(s.params_billion for s in m.CANDIDATES if s.family == "qwen2.5")
    assert all(spec.tier == "extra" for spec in m.CANDIDATES if spec.family in ("qwen3", "olmo2"))
    assert qwen_sizes == [0.5, 1.5, 3.0, 7.0, 14.0]
    for spec in m.CANDIDATES:
        if spec.family.startswith("llama"):
            assert spec.official_repo is not None and spec.cross_check_repos


def test_parse_tree_drops_directories_and_masks_gated_sha() -> None:
    tree = [
        {"type": "directory", "path": "onnx", "size": 0},
        {"type": "file", "path": "a.safetensors", "size": 5, "lfs": {"oid": SHA_A, "size": 5}},
        {"type": "file", "path": "b.safetensors", "size": 6, "lfs": {"oid": "*" * 64, "size": 6}},
        {"type": "file", "path": "config.json", "size": 7},
    ]
    files = m.parse_tree(tree)
    assert set(files) == {"a.safetensors", "b.safetensors", "config.json"}
    assert files["a.safetensors"].sha256 == SHA_A
    assert files["b.safetensors"].sha256 is None
    assert files["config.json"].sha256 is None


def test_wanted_paths_keeps_weights_and_small_files_only() -> None:
    assert m.wanted_paths(_listing(SHA_A)) == ["config.json", "model.safetensors", "tokenizer.json"]


def _spec_with_cross_check() -> m.CandidateSpec:
    return m.CandidateSpec(
        "toy", "mirror/toy", "toyfam", 1.0, "qwen", "MIT",
        cross_check_repos=("other/toy",), official_repo="gated/toy",
    )


def test_verify_provenance_accepts_identical_mirrors_and_official_size() -> None:
    listings = {
        "mirror/toy": _listing(SHA_A),
        "other/toy": _listing(SHA_A),
        "gated/toy": _listing(None),
    }
    proofs = m.verify_provenance(_spec_with_cross_check(), listings)
    assert len(proofs) == 1
    assert "sha256 identical in other/toy" in proofs[0]
    assert "size equals official gated/toy" in proofs[0]


def test_verify_provenance_rejects_a_different_sha() -> None:
    listings = {"mirror/toy": _listing(SHA_A), "other/toy": _listing(SHA_B), "gated/toy": _listing(None)}
    with pytest.raises(m.ModelManifestError, match="differs from other/toy"):
        m.verify_provenance(_spec_with_cross_check(), listings)


def test_verify_provenance_rejects_a_different_official_size() -> None:
    listings = {
        "mirror/toy": _listing(SHA_A), "other/toy": _listing(SHA_A), "gated/toy": _listing(None, size=99),
    }
    with pytest.raises(m.ModelManifestError, match="size differs from official"):
        m.verify_provenance(_spec_with_cross_check(), listings)


def test_build_manifest_and_tsv_round_trip() -> None:
    spec = m.CandidateSpec("solo", "public/solo", "fam", 0.5, "smollm", "Apache-2.0")
    entries, proofs = m.build_manifest([spec], fetch=lambda repo: _listing(SHA_A))
    assert [e.path for e in entries] == ["config.json", "model.safetensors", "tokenizer.json"]
    assert proofs["solo"] == ["model.safetensors: single source"]
    lines = m.manifest_to_tsv(entries).strip().split("\n")
    assert lines[1].split("\t") == ["solo", "public/solo", "model.safetensors", "100", SHA_A]
    assert lines[0].split("\t")[-1] == "-"


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 1000
    path.write_bytes(payload)
    assert m.sha256_file(path, chunk_bytes=999) == hashlib.sha256(payload).hexdigest()
