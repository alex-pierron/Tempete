"""Style channels: scalar features of a reply, tested endpoint against the S pool, prompt by prompt.

Kernel agreement compares *what* is said. A LoRA, even at low strength, may shift *how* it is
said (length, layout, punctuation) on every prompt by a little; a small shift repeated over many
prompts adds up, which a single-prompt comparison cannot see. For one feature phi:

    delta[e, p] = mean phi of endpoint e at prompt p
                  - mean over the other S endpoints of their mean phi at prompt p
    t[e]        = mean_p delta[e, p] / (sd_p delta[e, p] / sqrt(P))        (one-sample t, units = prompts)

The disguise endpoints (system prompt, persona, watermark, temperature, plain) are the null
controls: a feature is only useful if `|t|` of T1 / T2 clearly exceeds the `|t|` of every
disguise. Prompts are the independent units, so the t is honest about prompt-level variance but
not about the 9-endpoint design: it is a screening statistic, not a calibrated p-value.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from hackathon.model_equality.research.discrimination import BASE_MODEL, EndpointTruth
from hackathon.model_equality.store import Answer

WORD_PATTERN = re.compile(r"\S+")
MIN_PROMPTS = 4
SENTENCE_END = (".", "!", "?", "\"", ")")


def _words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def feat_log_words(text: str) -> float:
    """log(1 + number of whitespace-separated words)."""
    return math.log1p(len(_words(text)))


def feat_lines(text: str) -> float:
    """Number of non-empty lines."""
    return float(sum(1 for line in text.splitlines() if line.strip()))


def feat_digit_fraction(text: str) -> float:
    """Fraction of characters that are digits."""
    return sum(c.isdigit() for c in text) / max(len(text), 1)


def feat_upper_fraction(text: str) -> float:
    """Fraction of characters that are upper-case letters."""
    return sum(c.isupper() for c in text) / max(len(text), 1)


def feat_mean_word_length(text: str) -> float:
    """Mean word length in characters."""
    words = _words(text)
    return sum(len(w) for w in words) / len(words) if words else 0.0


def feat_type_token_ratio(text: str) -> float:
    """Distinct words over words (low = repetitive)."""
    words = [w.lower() for w in _words(text)]
    return len(set(words)) / len(words) if words else 1.0


def feat_ends_cleanly(text: str) -> float:
    """1 when the reply ends with sentence-final punctuation (0 for a truncated reply)."""
    return float(text.rstrip().endswith(SENTENCE_END))


def feat_punctuation_fraction(text: str) -> float:
    """Fraction of characters that are punctuation."""
    return sum((not c.isalnum()) and (not c.isspace()) for c in text) / max(len(text), 1)


FEATURES: Mapping[str, Callable[[str], float]] = {
    "log_words": feat_log_words,
    "lines": feat_lines,
    "digit_fraction": feat_digit_fraction,
    "upper_fraction": feat_upper_fraction,
    "mean_word_length": feat_mean_word_length,
    "type_token_ratio": feat_type_token_ratio,
    "ends_cleanly": feat_ends_cleanly,
    "punctuation_fraction": feat_punctuation_fraction,
}


@dataclass(frozen=True)
class StyleResult:
    """t statistic of every endpoint for one feature."""

    feature: str
    t: dict[str, float]
    mean_delta: dict[str, float]
    prompts: int


def style_tests(
    answers: Sequence[Answer],
    truth: Mapping[str, EndpointTruth],
    endpoints: Sequence[str],
    prompt_filter: Callable[[str], bool] | None = None,
) -> list[StyleResult]:
    """One-sample t per endpoint and feature (see module docstring).

    Args:
        answers: Reference answers.
        truth: Ground truth of the endpoints.
        endpoints: Endpoints to test.
        prompt_filter: Optional predicate on prompt ids (restrict to a family).
    """
    base = [e for e in endpoints if truth[e].model == BASE_MODEL]
    by_cell: dict[tuple[str, str], list[Answer]] = {}
    for answer in answers:
        if answer.endpoint in endpoints and (prompt_filter is None or prompt_filter(answer.prompt_id)):
            by_cell.setdefault((answer.prompt_id, answer.endpoint), []).append(answer)
    prompts = sorted({p for p, _ in by_cell})
    results: list[StyleResult] = []
    for name, feature in FEATURES.items():
        mean = {key: float(np.mean([feature(a.response) for a in rows]))
                for key, rows in by_cell.items()}
        t_values: dict[str, float] = {}
        deltas_mean: dict[str, float] = {}
        used = 0
        for endpoint in endpoints:
            deltas: list[float] = []
            for prompt in prompts:
                if (prompt, endpoint) not in mean:
                    continue
                others = [mean[(prompt, s)] for s in base if s != endpoint and (prompt, s) in mean]
                if others:
                    deltas.append(mean[(prompt, endpoint)] - float(np.mean(others)))
            used = max(used, len(deltas))
            if len(deltas) < MIN_PROMPTS:
                t_values[endpoint] = float("nan")
                deltas_mean[endpoint] = float("nan")
                continue
            array = np.asarray(deltas)
            spread = float(array.std(ddof=1))
            deltas_mean[endpoint] = float(array.mean())
            t_values[endpoint] = (0.0 if spread == 0.0
                                  else float(array.mean() / (spread / math.sqrt(len(array)))))
        results.append(StyleResult(name, t_values, deltas_mean, used))
    return results
