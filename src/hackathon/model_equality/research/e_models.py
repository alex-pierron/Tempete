"""Candidate base models for the likelihood attack of team E, and their download manifest.

The `ref-*` endpoints serve SmolLM2-135M-Instruct (team D, then confirmed here) and the `ep-*`
endpoints present themselves as Qwen (Alibaba) and Llama (Meta). With a LOCAL copy of each candidate
base model we can score the answers of an endpoint by log-likelihood, which is far more powerful
than a two-sample test on the answers alone.

This module is the single registry of the candidates (name, repository, family, chat-template kind,
licence) and builds the manifest that the login-node downloader reads. It only needs the standard
library, so it imports on the GPU node and on the Mac alike. Network access happens only in
`build_manifest`, from the Mac, against the public metadata endpoint of Hugging Face.

Provenance rule: the official Meta repositories are gated and we have no token. We download
byte-identical public mirrors. The Hugging Face metadata API hides the sha256 of gated files but
shows their exact size, so a mirror is accepted when (a) its file sizes equal the official sizes and
(b) a second, independent mirror lists the same sha256. The sha256 of the downloaded bytes is then
recomputed on the compute node (`sha256_file`), never on the login node.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from hackathon.common.errors import HackathonError

HF_API_TREE_URL = "https://huggingface.co/api/models/{repo}/tree/main"
HF_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/main/{path}"
HTTP_TIMEOUT_SECONDS = 60
SHA256_CHUNK_BYTES = 16 * 1024 * 1024
MASKED_SHA256_PREFIX = "*"
"""The API replaces the sha256 of a gated LFS file by a run of asterisks."""

TemplateKind = Literal["qwen", "llama3", "smollm", "other"]
Tier = Literal["core", "extra"]

TEXT_FILES_TO_FETCH: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)
"""Small files needed to tokenise and load a model (vocab.json / merges.txt are redundant with
tokenizer.json). Few files per model on purpose: the shared scratch is at 98 % of its inode quota."""
WEIGHT_SUFFIX = ".safetensors"


class ModelManifestError(HackathonError):
    """The candidate registry or a repository listing is inconsistent (missing or mismatching file)."""


@dataclass(frozen=True)
class CandidateSpec:
    """One candidate base model.

    Attributes:
        key: Short unique name used in every output file (`qwen2.5-7b`).
        repo: Public repository the bytes are downloaded from.
        family: Model family, the unit of the "same family" decision (`qwen2.5`, `llama3.2`...).
        params_billion: Parameter count in billions, for sorting and for the GPU scheduler.
        template: Chat template kind (selects the hand-built "no system prompt" variant).
        licence: Licence of the weights (to be repeated in the report).
        cross_check_repos: Other public repositories that must list the same safetensors sha256.
        official_repo: Official (possibly gated) repository, compared by file size only.
        tokenizer_repo: Repository the tokenizer files come from, when it differs from `repo`
            (a mirror whose re-serialised tokenizer is not byte-identical to the official one).
        tier: `core` candidates are fetched first; `extra` ones only if the core misses an endpoint.
    """

    key: str
    repo: str
    family: str
    params_billion: float
    template: TemplateKind
    licence: str
    cross_check_repos: tuple[str, ...] = ()
    official_repo: str | None = None
    tokenizer_repo: str | None = None
    tier: Tier = "core"


LICENCE_APACHE = "Apache-2.0"
LICENCE_QWEN_RESEARCH = "Qwen Research License (Qwen2.5-3B)"
LICENCE_LLAMA_31 = "Llama 3.1 Community License (research use)"
LICENCE_LLAMA_32 = "Llama 3.2 Community License (research use)"
LICENCE_LLAMA_3 = "Meta Llama 3 Community License (research use)"
LICENCE_MIT = "MIT"

CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec("smollm2-135m", "HuggingFaceTB/SmolLM2-135M-Instruct", "smollm2", 0.135, "smollm",
                  LICENCE_APACHE),
    CandidateSpec("smollm2-360m", "HuggingFaceTB/SmolLM2-360M-Instruct", "smollm2", 0.36, "smollm",
                  LICENCE_APACHE),
    CandidateSpec("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "smollm2", 1.7, "smollm",
                  LICENCE_APACHE),
    CandidateSpec("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5", 0.5, "qwen", LICENCE_APACHE),
    CandidateSpec("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5", 1.5, "qwen", LICENCE_APACHE),
    CandidateSpec("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", "qwen2.5", 3.0, "qwen",
                  LICENCE_QWEN_RESEARCH),
    CandidateSpec("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5", 7.0, "qwen", LICENCE_APACHE),
    CandidateSpec("qwen2.5-14b", "Qwen/Qwen2.5-14B-Instruct", "qwen2.5", 14.0, "qwen", LICENCE_APACHE),
    CandidateSpec("llama3.1-8b", "NousResearch/Meta-Llama-3.1-8B-Instruct", "llama3.1", 8.0, "llama3",
                  LICENCE_LLAMA_31, cross_check_repos=("unsloth/Llama-3.1-8B-Instruct",),
                  official_repo="meta-llama/Llama-3.1-8B-Instruct"),
    CandidateSpec("llama3.2-1b", "alpindale/Llama-3.2-1B-Instruct", "llama3.2", 1.2, "llama3",
                  LICENCE_LLAMA_32, cross_check_repos=("unsloth/Llama-3.2-1B-Instruct",),
                  official_repo="meta-llama/Llama-3.2-1B-Instruct"),
    CandidateSpec("llama3.2-3b", "alpindale/Llama-3.2-3B-Instruct", "llama3.2", 3.2, "llama3",
                  LICENCE_LLAMA_32, cross_check_repos=("unsloth/Llama-3.2-3B-Instruct",),
                  official_repo="meta-llama/Llama-3.2-3B-Instruct"),
    CandidateSpec("llama3-8b", "NousResearch/Meta-Llama-3-8B-Instruct", "llama3", 8.0, "llama3",
                  LICENCE_LLAMA_3, cross_check_repos=("unsloth/llama-3-8b-Instruct",),
                  official_repo="meta-llama/Meta-Llama-3-8B-Instruct"),
    # Extra tier: other versions and families that an endpoint could hide behind. Only the default
    # chat template is scored for the "other" kind (no hand-built variant exists for it).
    CandidateSpec("qwen2-7b", "Qwen/Qwen2-7B-Instruct", "qwen2", 7.0, "qwen", LICENCE_APACHE,
                  tier="extra"),
    CandidateSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "qwen3", 1.7, "qwen", LICENCE_APACHE,
                  tier="extra"),
    CandidateSpec("qwen3-4b", "Qwen/Qwen3-4B", "qwen3", 4.0, "qwen", LICENCE_APACHE, tier="extra"),
    CandidateSpec("qwen3-8b", "Qwen/Qwen3-8B", "qwen3", 8.0, "qwen", LICENCE_APACHE, tier="extra"),
    CandidateSpec("olmo2-7b", "allenai/OLMo-2-1124-7B-Instruct", "olmo2", 7.0, "other",
                  LICENCE_APACHE, tier="extra"),
    CandidateSpec("mistral-7b-v0.3", "unsloth/mistral-7b-instruct-v0.3", "mistral", 7.0, "other",
                  LICENCE_APACHE, tier="extra"),
    CandidateSpec("phi3.5-mini", "microsoft/Phi-3.5-mini-instruct", "phi3", 3.8, "other",
                  LICENCE_MIT, tier="extra"),
)

CANDIDATE_BY_KEY: Mapping[str, CandidateSpec] = {spec.key: spec for spec in CANDIDATES}
VALIDATION_CANDIDATE = "smollm2-135m"
"""The model the `ref-*` endpoints serve: the likelihood method is validated on it first."""


@dataclass(frozen=True)
class RepoFile:
    """One file of a repository listing.

    Attributes:
        path: Path inside the repository.
        size: Size in bytes.
        sha256: LFS sha256 (hex) when the API shows it, or None for git blobs and masked files.
    """

    path: str
    size: int
    sha256: str | None


@dataclass(frozen=True)
class ManifestEntry:
    """One file to download, as written to the manifest the login-node downloader reads."""

    key: str
    repo: str
    path: str
    size: int
    sha256: str | None


def get_candidate(key: str) -> CandidateSpec:
    """Return the candidate named `key`.

    Raises:
        ModelManifestError: Unknown key.
    """
    try:
        return CANDIDATE_BY_KEY[key]
    except KeyError as exc:
        raise ModelManifestError(
            f"unknown candidate {key!r}; known: {sorted(CANDIDATE_BY_KEY)}"
        ) from exc


def parse_tree(listing: Sequence[Mapping[str, Any]]) -> dict[str, RepoFile]:
    """Turn the JSON of the tree endpoint into `{path: RepoFile}` (directories are dropped).

    A gated LFS file has its sha256 masked by asterisks: it is reported as `sha256=None`.
    """
    files: dict[str, RepoFile] = {}
    for entry in listing:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs")
        sha256: str | None = None
        if isinstance(lfs, Mapping):
            oid = str(lfs.get("oid", ""))
            sha256 = None if oid.startswith(MASKED_SHA256_PREFIX) or len(oid) != 64 else oid
        path = str(entry["path"])
        files[path] = RepoFile(path=path, size=int(entry["size"]), sha256=sha256)
    return files


def fetch_tree(repo: str, *, timeout: float = HTTP_TIMEOUT_SECONDS) -> dict[str, RepoFile]:
    """Fetch the public file listing of `repo`.

    Raises:
        ModelManifestError: HTTP or network error, or a payload that is not a JSON list.
    """
    url = HF_API_TREE_URL.format(repo=repo)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ModelManifestError(f"cannot list {repo}: {exc}") from exc
    if not isinstance(payload, list):
        raise ModelManifestError(f"cannot list {repo}: payload is not a list")
    return parse_tree(payload)


def wanted_paths(listing: Mapping[str, RepoFile]) -> list[str]:
    """Files to download from a listing: every safetensors shard plus the small text files."""
    return sorted(
        path for path in listing
        if path.endswith(WEIGHT_SUFFIX) or path in TEXT_FILES_TO_FETCH
    )


def verify_provenance(
    spec: CandidateSpec, listings: Mapping[str, Mapping[str, RepoFile]]
) -> list[str]:
    """Compare the mirror with its cross-checks and with the official sizes.

    Args:
        spec: The candidate.
        listings: `{repo: listing}` for `spec.repo`, its cross-checks and its official repo.

    Returns:
        Human-readable proof lines, e.g. `model-00001-of-00004.safetensors: sha256 identical in
        2 mirrors, size equals official`.

    Raises:
        ModelManifestError: A safetensors shard differs in size or sha256 between repositories.
    """
    main = listings[spec.repo]
    weights = [path for path in main if path.endswith(WEIGHT_SUFFIX)]
    if not weights:
        raise ModelManifestError(f"{spec.repo}: no safetensors file in the listing")
    proofs: list[str] = []
    for path in sorted(weights):
        mine = main[path]
        notes: list[str] = []
        for other_repo in spec.cross_check_repos:
            other = listings[other_repo].get(path)
            if other is None or other.size != mine.size or other.sha256 != mine.sha256:
                raise ModelManifestError(f"{spec.repo}:{path} differs from {other_repo}")
            notes.append(f"sha256 identical in {other_repo}")
        if spec.official_repo is not None:
            official = listings[spec.official_repo].get(path)
            if official is None or official.size != mine.size:
                raise ModelManifestError(
                    f"{spec.repo}:{path} size differs from official {spec.official_repo}"
                )
            notes.append(f"size equals official {spec.official_repo} ({mine.size} bytes)")
        proofs.append(f"{path}: " + ("; ".join(notes) if notes else "single source"))
    return proofs


def build_manifest(
    specs: Sequence[CandidateSpec],
    fetch: Callable[[str], dict[str, RepoFile]] = fetch_tree,
) -> tuple[list[ManifestEntry], dict[str, list[str]]]:
    """List the files of every candidate and check their provenance.

    Args:
        specs: Candidates to include.
        fetch: Listing function (injected so tests need no network).

    Returns:
        The download entries and, per candidate key, the provenance proof lines.
    """
    entries: list[ManifestEntry] = []
    proofs: dict[str, list[str]] = {}
    cache: dict[str, dict[str, RepoFile]] = {}

    def listing_of(repo: str) -> dict[str, RepoFile]:
        if repo not in cache:
            cache[repo] = fetch(repo)
        return cache[repo]

    for spec in specs:
        repos = [spec.repo, *spec.cross_check_repos]
        if spec.official_repo is not None:
            repos.append(spec.official_repo)
        if spec.tokenizer_repo is not None:
            repos.append(spec.tokenizer_repo)
        listings = {repo: listing_of(repo) for repo in repos}
        proofs[spec.key] = verify_provenance(spec, listings)
        main = listings[spec.repo]
        for path in wanted_paths(main):
            entries.append(ManifestEntry(spec.key, spec.repo, path, main[path].size, main[path].sha256))
        if spec.tokenizer_repo is not None:
            tokenizer_listing = listings[spec.tokenizer_repo]
            for path in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                if path in tokenizer_listing:
                    entries = [e for e in entries if not (e.key == spec.key and e.path == path)]
                    info = tokenizer_listing[path]
                    entries.append(
                        ManifestEntry(spec.key, spec.tokenizer_repo, path, info.size, info.sha256)
                    )
    return entries, proofs


def manifest_to_tsv(entries: Sequence[ManifestEntry]) -> str:
    """Serialise the manifest for the bash downloader: `key repo path size sha256|-` per line."""
    lines = [
        "\t".join((e.key, e.repo, e.path, str(e.size), e.sha256 or "-")) for e in entries
    ]
    return "\n".join(lines) + "\n"


def manifest_to_json(
    entries: Sequence[ManifestEntry], proofs: Mapping[str, Sequence[str]]
) -> str:
    """Serialise manifest and provenance proofs as the JSON kept with the report."""
    payload = {
        "entries": [asdict(entry) for entry in entries],
        "provenance": {key: list(lines) for key, lines in proofs.items()},
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def sha256_file(path: Path, *, chunk_bytes: int = SHA256_CHUNK_BYTES) -> str:
    """Hex sha256 of a file, streamed. Run it on a compute node, never on a login node."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write the download manifest (network, run it from the Mac)."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--tier", choices=("core", "extra", "all"), default="core")
    parser.add_argument("--keys", nargs="*", default=None, help="restrict to these candidate keys")
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    specs = [
        spec for spec in CANDIDATES
        if (args.tier == "all" or spec.tier == args.tier)
        and (args.keys is None or spec.key in args.keys)
    ]
    entries, proofs = build_manifest(specs)
    args.out_tsv.write_text(manifest_to_tsv(entries), encoding="utf-8")
    args.out_json.write_text(manifest_to_json(entries, proofs), encoding="utf-8")
    total_gb = sum(entry.size for entry in entries) / 1e9
    print(f"{len(specs)} candidates, {len(entries)} files, {total_gb:.1f} GB")
    for key, lines in proofs.items():
        print(f"{key}: {len(lines)} weight file(s) verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
