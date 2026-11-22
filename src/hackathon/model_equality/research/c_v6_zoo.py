"""Evaluation of configuration v6 on the zoo of team F (labelled variants of the real base models).

The zoo has no long answer: an endpoint's budget of `tokens` is reached by drawing short rows
(arithmetic answers, one draw per prompt, the same prompts for every endpoint of a trial, as the
real design would use) and the 3 identity answers (one per formulation). The conversion into long
answers is `(tokens - 563) / 256` per endpoint (563 tokens are already in hand, a long answer is
at most 256 tokens). A simulated family has 11 endpoints, like an ep family: the ten zoo variants
plus a second plain copy; truth from `zoo_truth.json` (plain, copy, system prompt, persona,
watermark and temperature 1.0 are the main model `S`; both fp8 copies are one model; each LoRA is
alone). Features are read from the likelihood tables of the zoo (`ll_<family>.json.gz`): the
log-probability of every answer under the family's base model, and under the fp8 simulation.

Limits stated in advance: the zoo fp8 is simulated with the same recipe as the fp8 statistic (an
optimistic ratio: 0.05 per token against 0.0064 on the real ref), the zoo LoRAs are stronger on the
likelihood than the ref adapters, and short answers carry less information per token than long
ones. The real ref (`evaluate_ref`) is the independent check.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hackathon.model_equality.research.c_identity import is_disguised
from hackathon.model_equality.research.c_pairs import Partition
from hackathon.model_equality.research.c_rehearsal import official_score
from hackathon.model_equality.research.c_v6 import EndpointStats, RuleV6, decide_v6

SEED = 20261123
TRIALS = 200
LONG_ANSWER_TOKENS = 256
TOKENS_IN_HAND = 563
IDENTITY_IDS: tuple[str, ...] = ("d-ident-000", "d-ident-001", "d-ident-002")
ARITHMETIC_PREFIXES: tuple[str, ...] = ("h-arith", "p10-arith", "z-arith")
MAIN_MODEL = "S"
FP8_MODEL = "T1_fp8"
PLAIN_COPY = "plain_copy"
BASE_KEY = "base"
FP8_KEY = "fp8_w"
FAMILIES: tuple[str, ...] = ("qwen15", "qwen3", "llama1", "smollm")


@dataclass(frozen=True)
class ZooRow:
    """One answer: its text, token count, log-probability under the base and fp8 ratio sum."""

    text: str
    tokens: int
    log_prob: float
    llr: float


def long_answers_for(tokens: int) -> float:
    """Long answers per endpoint that a token budget stands for (563 tokens already in hand)."""
    return max(tokens - TOKENS_IN_HAND, 0) / LONG_ANSWER_TOKENS


def tokens_for(long_answers: int) -> int:
    """Token budget of an endpoint with `long_answers` more long answers."""
    return TOKENS_IN_HAND + LONG_ANSWER_TOKENS * long_answers


def load_family(
    zoo_dir: Path, family: str
) -> tuple[dict[str, dict[tuple[str, int], ZooRow]], dict[str, str], dict[str, str]]:
    """`(rows[endpoint][(prompt id, repeat)], variant of endpoint, model of endpoint)`."""
    truth = json.loads((zoo_dir / "zoo_truth.json").read_text(encoding="utf-8"))
    with gzip.open(zoo_dir / f"ll_{family}.json.gz") as handle:
        scores = json.loads(handle.read())["scores"]
    rows: dict[str, dict[tuple[str, int], ZooRow]] = defaultdict(dict)
    for line in (zoo_dir / f"zoo_{family}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = f"{row['endpoint']}|{row['id']}|{row['repeat']}"
        log_prob, tokens = scores[BASE_KEY][key]
        llr = scores[FP8_KEY][key][0] - log_prob
        rows[row["endpoint"]][(row["id"], int(row["repeat"]))] = ZooRow(
            row["response"], int(tokens), float(log_prob), float(llr))
    variants = {e: truth[e]["variant"] for e in rows}
    models = {e: truth[e]["model"] for e in rows}
    return dict(rows), variants, models


def true_models(models: Mapping[str, str], names: Sequence[str]) -> Partition:
    """Truth partition (one family) of the simulated endpoints `names` (a plain copy is `S`)."""
    labels = [MAIN_MODEL if n == PLAIN_COPY else models[n] for n in names]
    order = sorted(set(labels))
    return Partition(np.zeros(len(names), dtype=np.int64),
                     np.array([order.index(x) for x in labels], dtype=np.int64))


def draw_stats(
    rows: Mapping[str, Mapping[tuple[str, int], ZooRow]], names: Sequence[str],
    source_of: Mapping[str, str], prompts: Sequence[str], rng: np.random.Generator,
) -> list[EndpointStats]:
    """Pooled statistics of each simulated endpoint on the same `prompts` plus 3 identity rows."""
    out: list[EndpointStats] = []
    for name in names:
        table = rows[source_of[name]]
        picked = [table[(p, int(rng.integers(0, 4)))] for p in prompts]
        identity = [table[(i, int(rng.integers(0, 20)))] for i in IDENTITY_IDS]
        used = picked + identity
        tokens = sum(r.tokens for r in used)
        out.append(EndpointStats(
            tokens=tokens, mean_ll=sum(r.log_prob for r in used) / max(tokens, 1),
            fp8_llr=sum(r.llr for r in used) / max(tokens, 1),
            disguised=is_disguised([r.text for r in identity])))
    return out


@dataclass(frozen=True)
class FamilyResult:
    """Score of the v6 chain on simulated families of one base model at one token budget."""

    family: str
    tokens: int
    long_answers: float
    mean: float
    sd: float
    p10: float
    p90: float
    isolated_share: Mapping[str, float]


def simulate_family(
    zoo_dir: Path, family: str, tokens: int, rule: RuleV6, trials: int = TRIALS, seed: int = SEED,
) -> FamilyResult:
    """Run `trials` simulated families of 11 endpoints at `tokens` per endpoint."""
    rows, variants, models = load_family(zoo_dir, family)
    endpoints = sorted(rows)
    plain = next(e for e in endpoints if variants[e] == "plain")
    names = [*endpoints, PLAIN_COPY]
    source_of = {**{e: e for e in endpoints}, PLAIN_COPY: plain}
    truth = true_models(models, names)
    prompts_all = sorted({p for (p, _r) in rows[plain] if p.startswith(ARITHMETIC_PREFIXES)})
    mean_tokens = float(np.mean([r.tokens for (p, _), r in rows[plain].items()
                                 if p.startswith(ARITHMETIC_PREFIXES)]))
    count = min(len(prompts_all), max(1, round(tokens / max(mean_tokens, 1.0))))
    rng = np.random.default_rng(seed + tokens)
    scores: list[float] = []
    isolated = {n: 0 for n in names}
    for _ in range(trials):
        prompts = [prompts_all[i] for i in rng.permutation(len(prompts_all))[:count]]
        stats = draw_stats(rows, names, source_of, prompts, rng)
        guess, roles = decide_v6([0] * len(names), stats, rule)
        scores.append(official_score(truth, guess))
        for name, role in zip(names, roles, strict=True):
            isolated[name] += int(role != "main")
    label = {n: variants.get(n, PLAIN_COPY) for n in names}
    share = {label[n]: isolated[n] / trials for n in names}
    low, high = np.percentile(scores, (10.0, 90.0))
    return FamilyResult(family, tokens, long_answers_for(tokens), float(np.mean(scores)),
                        float(np.std(scores)), float(low), float(high), share)
