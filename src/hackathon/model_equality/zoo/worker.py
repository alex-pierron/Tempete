"""One zoo family on one GPU: ten variants of the same model, answers in the official format.

Stage A (generation): `plain`, `sysprompt`, `persona`, `watermark`, `temp1` on the plain weights;
the ONE LoRA adapter (trained here for a few minutes, or loaded when its file exists) applied at
0.3, 0.7 and 1.5; then fp8 (weights E4M3 per tensor) and its W8A8 bound. Stage B (scoring): the
per-token log-likelihood of every zoo answer of every family under THIS family's base model (the
cross-family matrix that tells a family from another), and for this family's own answers the same
under the fp8 simulations (the fp8 likelihood ratio of team D).

Everything is idempotent: a variant is skipped when its rows are complete AND its sampling
signature (temperature, seed, system text, adapter hash...) is unchanged; the adapter is trained
once; outputs are replaced atomically. Run it once per GPU with `CUDA_VISIBLE_DEVICES`:

    python -m hackathon.model_equality.zoo.worker run --family qwen --bundle bundle.json \
        --cache-dir hf_cache --out-dir data/zoo --lora-data data/alpaca.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hackathon.model_equality.zoo.config import (
    DEFAULT_TEMPERATURE,
    FAMILIES,
    FAMILY_BY_KEY,
    GENERATION_ORDER,
    LORA_EFFECTIVE_BATCH,
    LORA_LEARNING_RATE,
    LORA_TRAIN_STEPS,
    MAX_NEW_TOKENS,
    TEMPERATURE_ONE,
    VARIANT_BY_KEY,
    ZOO_SEED,
    FamilySpec,
    VariantSpec,
    ZooConfigError,
    endpoint_name,
)
from hackathon.model_equality.zoo.lora import (
    Fp8Weights,
    ZooLinear,
    attach_all_lora,
    install_zoo_linears,
    load_adapter,
    load_training_examples,
    save_adapter,
    set_activation_fp8,
    set_lora_scale,
    train_lora,
)
from hackathon.model_equality.zoo.prompts import (
    SYSTEM_PROMPT,
    PromptBundle,
    Unit,
    load_bundle,
    persona_for,
    stable_index,
)
from hackathon.model_equality.zoo.sampling import chat_text, generate_texts
from hackathon.model_equality.zoo.scoring import score_replies
from hackathon.model_equality.zoo.store import (
    commit_endpoint,
    make_row,
    read_rows,
    row_counts,
)

ADAPTER_TRAINING_ESTIMATE_SECONDS = 900.0
"""Time reserved for training the adapter when deciding whether a LoRA variant still fits."""
PEER_POLL_SECONDS = 20.0
DEFAULT_PEER_WAIT_MINUTES = 25.0
SECONDS_PER_MINUTE = 60.0
VARIANT_TIME_MARGIN = 1.3
"""A variant is started only if the time left exceeds its last duration times this margin."""
SCORE_KEY_SEPARATOR = "|"


class WorkerError(RuntimeError):
    """A run cannot proceed (missing adapter data, unreadable model folder)."""


@dataclass(frozen=True)
class RunSettings:
    """Everything one `run` needs, validated once at the command line."""

    family: FamilySpec
    model_dir: Path
    out_dir: Path
    bundle: PromptBundle
    lora_data: Path | None
    default_temperature: float
    variants: tuple[str, ...]
    deadline_epoch: float | None
    lora_steps: int
    effective_batch: int
    do_scoring: bool
    peers: tuple[str, ...]
    peer_wait_seconds: float
    device: torch.device
    dtype: torch.dtype
    max_new_tokens: int = MAX_NEW_TOKENS
    lora_learning_rate: float = LORA_LEARNING_RATE

    @property
    def answers_path(self) -> Path:
        """The family's official-format answers file."""
        return self.out_dir / f"zoo_{self.family.key}.jsonl"

    @property
    def meta_path(self) -> Path:
        """The family's run metadata."""
        return self.out_dir / f"meta_{self.family.key}.json"

    @property
    def adapter_path(self) -> Path:
        """The family's LoRA adapter file."""
        return self.out_dir / f"lora_{self.family.key}.safetensors"

    @property
    def likelihood_path(self) -> Path:
        """The likelihood scores computed under this family's base model."""
        return self.out_dir / f"ll_{self.family.key}.json"


def log(message: str) -> None:
    """Timestamped, flushed progress line (job logs are read while the job runs)."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON through a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """A JSON object from `path`, or `{}` when it does not exist yet."""
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(text: str) -> str:
    """Short SHA-256 of a text (signature of a system prompt)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def variant_seed(family_key: str, variant_key: str) -> int:
    """Seed of one (family, variant) sampling run: stable, distinct, derived from `ZOO_SEED`."""
    return ZOO_SEED + stable_index("seed", family_key, variant_key, modulo=10**6) * 1000


def system_text_for(variant: VariantSpec, family: FamilySpec, unit: Unit) -> str | None:
    """The system message of one request of a variant (None for the others)."""
    if variant.key == "sysprompt":
        return SYSTEM_PROMPT
    if variant.key == "persona":
        return persona_for(family.key, unit)
    return None


def variant_signature(
    settings: RunSettings, variant: VariantSpec, adapter_hash: str | None, units: int
) -> dict[str, object]:
    """Everything that, if changed, makes the stored answers of a variant stale."""
    temperature = TEMPERATURE_ONE if variant.temperature_one else settings.default_temperature
    return {
        "temperature": temperature, "watermark": variant.key == "watermark",
        "system": text_sha(SYSTEM_PROMPT) if variant.key == "sysprompt" else None,
        "persona_pool": variant.key == "persona", "lora_scale": variant.lora_scale,
        "adapter": adapter_hash if variant.needs_lora else None,
        "fp8": variant.key.startswith("fp8"), "max_new_tokens": settings.max_new_tokens,
        "seed": variant_seed(settings.family.key, variant.key), "units": units,
    }


def load_model(settings: RunSettings) -> tuple[Any, Any]:
    """Tokenizer and causal LM from the flat model folder, on the target device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(settings.model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(str(settings.model_dir), dtype=settings.dtype)
    model = model.to(settings.device).eval()
    log(f"model loaded in {time.time() - started:.0f}s from {settings.model_dir}")
    return tokenizer, model


def environment_report(settings: RunSettings) -> dict[str, object]:
    """Versions and hardware, for the run metadata."""
    import transformers

    gpu = (torch.cuda.get_device_name(settings.device)
           if settings.device.type == "cuda" else str(settings.device))
    return {"torch": torch.__version__, "transformers": transformers.__version__, "gpu": gpu,
            "dtype": str(settings.dtype), "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES")}


def ensure_adapter(
    settings: RunSettings, model: Any, tokenizer: Any, linears: dict[str, ZooLinear],
    meta: dict[str, Any],
) -> str:
    """Load the family adapter, training it first when its file does not exist; return its hash."""
    if settings.adapter_path.is_file():
        load_adapter(settings.adapter_path, linears)
        log(f"adapter loaded from {settings.adapter_path.name}")
    else:
        if settings.lora_data is None:
            raise WorkerError("no adapter file and no --lora-data to train one")
        examples = load_training_examples(
            tokenizer, settings.lora_data, settings.lora_steps * settings.effective_batch,
            ZOO_SEED + stable_index("lora-data", settings.family.key, modulo=10**6))
        log(f"training the adapter: {settings.lora_steps} steps x {settings.effective_batch}")
        pad = int(tokenizer.pad_token_id)
        torch.manual_seed(ZOO_SEED + stable_index("lora-init", settings.family.key, modulo=10**6))
        attach_all_lora(linears)
        report = train_lora(
            model, linears, examples, pad, settings.lora_steps, settings.effective_batch,
            settings.family.train_micro_batch, settings.device, log=log,
            learning_rate=settings.lora_learning_rate)
        save_adapter(settings.adapter_path, linears)
        meta["lora"] = {**report.to_json(), "learning_rate": settings.lora_learning_rate,
                        "dataset": settings.lora_data.name}
        log(f"adapter saved ({report.seconds:.0f}s, loss {report.losses[0]:.3f} -> "
            f"{report.losses[-1]:.3f}, peak {report.peak_memory_gib:.1f} GiB)")
    adapter_hash = file_sha256(settings.adapter_path)
    meta.setdefault("lora", {})["adapter_sha256"] = adapter_hash
    return adapter_hash


def run_variant(
    settings: RunSettings, tokenizer: Any, model: Any, variant: VariantSpec,
    units: Sequence[Unit], signature: dict[str, object], meta: dict[str, Any],
) -> None:
    """Generate every unit for one variant and commit its rows; record timing in `meta`."""
    family = settings.family
    prompts = [chat_text(tokenizer, u.text, system_text_for(variant, family, u)) for u in units]
    result = generate_texts(
        model, tokenizer, prompts, float(signature["temperature"]),  # type: ignore[arg-type]
        bool(signature["watermark"]), family.generation_batch,
        int(signature["seed"]),  # type: ignore[call-overload]
        settings.device, log=log, max_new_tokens=settings.max_new_tokens)
    endpoint = endpoint_name(family.key, variant.slot)
    rows = [make_row(endpoint, u.prompt_id, u.repeat, u.text, text)
            for u, text in zip(units, result.texts, strict=True)]
    commit_endpoint(settings.answers_path, endpoint, rows)
    mean_tokens = sum(result.new_tokens) / max(len(result.new_tokens), 1)
    meta.setdefault("variants", {})[variant.key] = {
        "endpoint": endpoint, "signature": signature, "rows": len(rows),
        "seconds": round(result.seconds, 1), "oom_events": result.oom_events,
        "final_batch": result.final_batch, "mean_new_tokens": round(mean_tokens, 2),
    }
    log(f"{endpoint} {variant.key}: {len(rows)} rows in {result.seconds:.0f}s "
        f"(mean {mean_tokens:.1f} new tokens, oom {result.oom_events})")


def deadline_allows(settings: RunSettings, last_seconds: float) -> bool:
    """True when a step that took `last_seconds` last time still fits before the deadline."""
    if settings.deadline_epoch is None:
        return True
    return time.time() + last_seconds * VARIANT_TIME_MARGIN < settings.deadline_epoch


def stage_a(
    settings: RunSettings, tokenizer: Any, model: Any, linears: dict[str, ZooLinear],
    meta: dict[str, Any],
) -> bool:
    """Generate the requested variants; return True when all of them are complete."""
    units = settings.bundle.units()
    stored = row_counts(settings.answers_path)
    adapter_hash: str | None = None
    fp8: Fp8Weights | None = None
    longest = 60.0
    complete = True
    for key in GENERATION_ORDER:
        if key not in settings.variants:
            continue
        variant = VARIANT_BY_KEY[key]
        if variant.needs_lora and adapter_hash is None:
            reserve = 0.0 if settings.adapter_path.is_file() else ADAPTER_TRAINING_ESTIMATE_SECONDS
            if not deadline_allows(settings, longest + reserve):
                log(f"{key}: not enough time to train / load the adapter, left for the next run")
                complete = False
                continue
            adapter_hash = ensure_adapter(settings, model, tokenizer, linears, meta)
            atomic_write_json(settings.meta_path, meta)
        signature = variant_signature(settings, variant, adapter_hash, len(units))
        endpoint = endpoint_name(settings.family.key, variant.slot)
        recorded = meta.get("variants", {}).get(key, {})
        if stored.get(endpoint) == len(units) and recorded.get("signature") == signature:
            log(f"{endpoint} {key}: already complete, skipped")
            continue
        if not deadline_allows(settings, longest):
            log(f"{endpoint} {key}: not enough time before the deadline, left for the next run")
            complete = False
            continue
        set_lora_scale(linears, variant.lora_scale if variant.needs_lora else 0.0)
        if variant.key.startswith("fp8"):
            if fp8 is None:
                fp8 = Fp8Weights(linears)
                meta["fp8_weight_relative_error"] = round(fp8.quantize(), 5)
                log(f"fp8 weights quantised (mean relative error "
                    f"{meta['fp8_weight_relative_error']})")
            set_activation_fp8(linears, variant.key == "fp8_w8a8")
        started = time.time()
        run_variant(settings, tokenizer, model, variant, units, signature, meta)
        longest = max(longest, time.time() - started)
        atomic_write_json(settings.meta_path, meta)
    set_activation_fp8(linears, False)
    set_lora_scale(linears, 0.0)
    if fp8 is not None:
        fp8.restore()
    return complete


def key_of(row: dict[str, object]) -> str:
    """The `endpoint|id|repeat` key of an answer row (the unit of the likelihood files)."""
    return SCORE_KEY_SEPARATOR.join((str(row["endpoint"]), str(row["id"]), str(row["repeat"])))


def wait_for_peers(settings: RunSettings) -> list[str]:
    """Wait (bounded) until the peer families finished stage A; return those that did not."""
    deadline = time.time() + settings.peer_wait_seconds
    pending = [p for p in settings.peers if p != settings.family.key]
    while pending and time.time() < deadline:
        pending = [p for p in pending
                   if not read_json(settings.out_dir / f"meta_{p}.json").get("stage_a_complete")]
        if pending:
            time.sleep(PEER_POLL_SECONDS)
    return pending


def stage_b(
    settings: RunSettings, tokenizer: Any, model: Any, linears: dict[str, ZooLinear],
    meta: dict[str, Any],
) -> None:
    """Score every zoo answer under this base model; own answers also under the fp8 simulations."""
    missing = wait_for_peers(settings)
    if missing:
        log(f"stage B: peers not finished, scoring without them: {missing}")
    rows: list[dict[str, object]] = []
    for path in sorted(settings.out_dir.glob("zoo_*.jsonl")):
        rows.extend(read_rows(path))
    log(f"stage B: scoring {len(rows)} answers under {settings.family.display}")
    pairs = [(str(r["prompt"]), str(r["response"])) for r in rows]
    set_lora_scale(linears, 0.0)
    started = time.time()
    base = score_replies(model, tokenizer, pairs, settings.family.scoring_batch, settings.device)
    scores: dict[str, dict[str, list[float]]] = {
        "base": {key_of(r): [s[0], s[1]] for r, s in zip(rows, base, strict=True)}}
    prefix = f"{settings.family.key}-"
    own = [(r, p) for r, p in zip(rows, pairs, strict=True)
           if str(r["endpoint"]).startswith(f"zoo-{prefix}")]
    fp8 = Fp8Weights(linears)
    fp8.quantize()
    for mode in ("fp8_w", "fp8_w8a8"):
        set_activation_fp8(linears, mode == "fp8_w8a8")
        result = score_replies(model, tokenizer, [p for _, p in own],
                               settings.family.scoring_batch, settings.device)
        scores[mode] = {key_of(r): [s[0], s[1]] for (r, _), s in zip(own, result, strict=True)}
    set_activation_fp8(linears, False)
    fp8.restore()
    atomic_write_json(settings.likelihood_path, {
        "scorer": settings.family.key, "scorer_model": settings.family.hf_repo,
        "default_temperature": settings.default_temperature, "scores": scores})
    meta["stage_b"] = {"answers": len(rows), "own_answers": len(own),
                       "seconds": round(time.time() - started, 1), "missing_peers": missing}
    atomic_write_json(settings.meta_path, meta)
    log(f"stage B done in {time.time() - started:.0f}s")


def run_family(settings: RunSettings) -> int:
    """Stage A then (optionally) stage B for one family; return a process exit code."""
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_json(settings.meta_path)
    meta.update({
        "family": settings.family.key, "display": settings.family.display,
        "hf_repo": settings.family.hf_repo, "model_dir": str(settings.model_dir),
        "default_temperature": settings.default_temperature, "seed": ZOO_SEED,
        "system_prompt": SYSTEM_PROMPT,
        "environment": environment_report(settings), "units": len(settings.bundle.units()),
    })
    tokenizer, model = load_model(settings)
    linears = install_zoo_linears(model)
    started = time.time()
    complete = stage_a(settings, tokenizer, model, linears, meta)
    meta["stage_a_complete"] = complete
    meta["stage_a_seconds"] = round(time.time() - started, 1)
    if settings.device.type == "cuda":
        meta["peak_memory_gib"] = round(torch.cuda.max_memory_allocated(settings.device) / 2**30, 1)
    atomic_write_json(settings.meta_path, meta)
    log(f"stage A {'complete' if complete else 'INCOMPLETE'} in {time.time() - started:.0f}s")
    if settings.do_scoring:
        stage_b(settings, tokenizer, model, linears, meta)
    return 0 if complete else 3


def parse_variants(text: str) -> tuple[str, ...]:
    """`all` or a comma list of variant keys.

    Raises:
        ZooConfigError: an unknown key.
    """
    if text == "all":
        return GENERATION_ORDER
    keys = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [k for k in keys if k not in VARIANT_BY_KEY]
    if unknown:
        raise ZooConfigError(f"unknown variants {unknown}; known {sorted(VARIANT_BY_KEY)}")
    return keys


def settings_from_args(args: argparse.Namespace) -> RunSettings:
    """Validate the command line into `RunSettings`."""
    if args.family not in FAMILY_BY_KEY:
        raise ZooConfigError(f"unknown family {args.family!r}; known {sorted(FAMILY_BY_KEY)}")
    family = FAMILY_BY_KEY[args.family]
    model_dir = Path(args.model_dir) if args.model_dir else Path(args.cache_dir) / family.cache_subdir
    if not (model_dir / "config.json").is_file():
        raise WorkerError(f"no config.json in {model_dir}")
    device = torch.device(args.device)
    return RunSettings(
        family=family, model_dir=model_dir, out_dir=Path(args.out_dir),
        bundle=load_bundle(Path(args.bundle)),
        lora_data=Path(args.lora_data) if args.lora_data else None,
        default_temperature=args.default_temperature, variants=parse_variants(args.variants),
        deadline_epoch=args.deadline_epoch, lora_steps=args.lora_steps,
        effective_batch=args.effective_batch, do_scoring=not args.no_score,
        peers=tuple(p for p in args.peers.split(",") if p),
        peer_wait_seconds=args.peer_wait_minutes * SECONDS_PER_MINUTE, device=device,
        dtype=getattr(torch, args.dtype), max_new_tokens=args.max_new_tokens,
        lora_learning_rate=args.lora_lr,
    )


def check_templates(cache_dir: Path, families: Sequence[FamilySpec]) -> int:
    """Render every prompt kind with every family tokenizer (no model, no GPU); print a report.

    Returns the number of families whose template failed.
    """
    from jinja2.exceptions import TemplateError
    from transformers import AutoTokenizer

    from hackathon.model_equality.zoo.lora import tokenize_example

    failures = 0
    for family in families:
        folder = cache_dir / family.cache_subdir
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(folder))
            user = "Compute 12 + 30. Answer with only the number."
            plain = chat_text(tokenizer, user)
            system = chat_text(tokenizer, user, SYSTEM_PROMPT)
            example = tokenize_example(tokenizer, "Name a color.", "Blue.")
        except (OSError, ValueError, TypeError, TemplateError) as error:
            failures += 1
            log(f"TEMPLATE FAIL {family.key}: {type(error).__name__}: {error}")
            continue
        has_system = SYSTEM_PROMPT in system
        log(f"template ok {family.key}: plain {len(plain)} chars, system role kept "
            f"{has_system}, training example {'ok' if example else 'None'}")
        log(f"  plain  : {plain[-120:]!r}")
        log(f"  system : {system[:60]!r} ...")
    return failures


def build_parser() -> argparse.ArgumentParser:
    """Command line of the worker."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="generate (and score) one family")
    run.add_argument("--family", required=True)
    run.add_argument("--bundle", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--cache-dir", default="hf_cache")
    run.add_argument("--model-dir", default=None, help="override the model folder (smoke test)")
    run.add_argument("--lora-data", default=None)
    run.add_argument("--default-temperature", type=float, default=DEFAULT_TEMPERATURE)
    run.add_argument("--variants", default="all")
    run.add_argument("--deadline-epoch", type=float, default=None)
    run.add_argument("--lora-steps", type=int, default=LORA_TRAIN_STEPS)
    run.add_argument("--effective-batch", type=int, default=LORA_EFFECTIVE_BATCH)
    run.add_argument("--no-score", action="store_true")
    run.add_argument("--peers", default=",".join(f.key for f in FAMILIES))
    run.add_argument("--peer-wait-minutes", type=float, default=DEFAULT_PEER_WAIT_MINUTES)
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    run.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    run.add_argument("--lora-lr", type=float, default=LORA_LEARNING_RATE)
    templates = commands.add_parser("templates", help="check the chat templates (no model)")
    templates.add_argument("--cache-dir", default="hf_cache")
    templates.add_argument("--families", default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: `run` one family or `templates` check."""
    args = build_parser().parse_args(argv)
    if args.command == "templates":
        chosen = FAMILIES if args.families == "all" else tuple(
            FAMILY_BY_KEY[k] for k in args.families.split(","))
        return 1 if check_templates(Path(args.cache_dir), chosen) else 0
    return run_family(settings_from_args(args))


if __name__ == "__main__":
    sys.exit(main())
