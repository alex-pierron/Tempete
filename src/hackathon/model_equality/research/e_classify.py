"""Supervised read-out of the likelihood features on the ref endpoints (honest, row-disjoint split).

Question: at a given token budget, can the features tell the MODEL variants that the score cares
about apart (clean base, fp8, LoRA 0.3 / 0.7 / 1.5) while ignoring the disguises that must stay
inside the base model (watermark, system prompt, persona, temperature 1.0)?

Protocol: the rows of every ref endpoint are split in two disjoint halves (A and B) by a seeded
shuffle; pseudo-endpoints drawn from A train a multinomial logistic regression on standardised
features, pseudo-endpoints drawn from B are the test set. The labels are MODEL labels:
`base` (plain, watermark, system, persona, temp1), `fp8`, `lora_0.3`, `lora_0.7`, `lora_1.5`.
The result is a confusion matrix and the share of correct pseudo-endpoints per label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.e_calib import (
    CALIBRATION_SEED,
    CLASS_OF,
    MEAN_LL,
    MLE_TEMPERATURE,
    Z_AT_ONE,
    RowFeatureTable,
    draw_features,
)

FloatArray = npt.NDArray[np.float64]
MODEL_LABEL_OF_CLASS: Mapping[str, str] = {
    "plain": "base", "watermark": "base", "system": "base", "persona": "base", "temp1": "base",
    "fp8": "fp8", "lora_0.3": "lora_0.3", "lora_0.7": "lora_0.7", "lora_1.5": "lora_1.5",
}
MODEL_LABELS: tuple[str, ...] = ("base", "fp8", "lora_0.3", "lora_0.7", "lora_1.5")
DEFAULT_TRAIN_DRAWS = 300
DEFAULT_TEST_DRAWS = 200
LOGISTIC_REGULARISATION = 1.0
LOGISTIC_MAX_ITERATIONS = 500
BASE_FEATURES: tuple[str, ...] = (MLE_TEMPERATURE, Z_AT_ONE, MEAN_LL)


class ClassifyError(HackathonError):
    """The feature set is empty or a class has no draw."""


@dataclass(frozen=True)
class ClassificationResult:
    """Confusion of a classifier on held-out pseudo-endpoints.

    Attributes:
        tokens: Token budget of every pseudo-endpoint.
        features: Feature names used.
        labels: Row and column order of the confusion matrix.
        confusion: `confusion[i][j]` = test draws of true label i predicted as label j.
        recall: Share of test draws of each label that are predicted correctly.
        balanced_accuracy: Mean recall over the labels.
    """

    tokens: int
    features: tuple[str, ...]
    labels: tuple[str, ...]
    confusion: npt.NDArray[np.int64]
    recall: Mapping[str, float]
    balanced_accuracy: float


def split_rows(table: RowFeatureTable, seed: int) -> tuple[RowFeatureTable, RowFeatureTable]:
    """Two tables over disjoint random halves of the rows of every endpoint (same columns)."""
    rng = np.random.default_rng(seed)
    first: list[int] = []
    second: list[int] = []
    endpoints = sorted({key[0] for key in table.keys})
    for endpoint in endpoints:
        indices = np.array([i for i, key in enumerate(table.keys) if key[0] == endpoint])
        rng.shuffle(indices)
        half = len(indices) // 2
        first += indices[:half].tolist()
        second += indices[half:].tolist()
    return _subset(table, first), _subset(table, second)


def _subset(table: RowFeatureTable, rows: Sequence[int]) -> RowFeatureTable:
    index = np.array(sorted(rows), dtype=np.int64)
    return RowFeatureTable(
        keys=tuple(table.keys[i] for i in index), n_tokens=table.n_tokens[index],
        temperatures=table.temperatures, lp=table.lp[index], ent=table.ent[index],
        var=table.var[index], llr={name: values[index] for name, values in table.llr.items()},
    )


def _design(
    table: RowFeatureTable, tokens: int, draws: int, features: Sequence[str], rng: np.random.Generator,
) -> tuple[FloatArray, list[str]]:
    """Feature matrix and model labels of pseudo-endpoints of every ref endpoint."""
    rows: list[list[float]] = []
    labels: list[str] = []
    for endpoint, klass in CLASS_OF.items():
        for draw in draw_features(table, endpoint, tokens, draws, rng):
            rows.append([draw[name] for name in features])
            labels.append(MODEL_LABEL_OF_CLASS[klass])
    return np.array(rows, dtype=np.float64), labels


def classify(
    table: RowFeatureTable, tokens: int, features: Sequence[str] = BASE_FEATURES, *,
    train_draws: int = DEFAULT_TRAIN_DRAWS, test_draws: int = DEFAULT_TEST_DRAWS,
    seed: int = CALIBRATION_SEED,
) -> ClassificationResult:
    """Train on half A, test on half B of the ref rows, at a token budget.

    Raises:
        ClassifyError: Empty feature list.
    """
    if not features:
        raise ClassifyError("at least one feature is required")
    train_table, test_table = split_rows(table, seed)
    rng = np.random.default_rng(seed + tokens)
    x_train, y_train = _design(train_table, tokens, train_draws, features, rng)
    x_test, y_test = _design(test_table, tokens, test_draws, features, rng)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        C=LOGISTIC_REGULARISATION, max_iter=LOGISTIC_MAX_ITERATIONS, class_weight="balanced"
    ).fit(scaler.transform(x_train), y_train)
    predicted = model.predict(scaler.transform(x_test))
    labels = MODEL_LABELS
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, guess in zip(y_test, predicted, strict=True):
        confusion[labels.index(truth), labels.index(str(guess))] += 1
    totals = confusion.sum(axis=1)
    recall = {
        label: float(confusion[i, i] / totals[i]) if totals[i] else float("nan")
        for i, label in enumerate(labels)
    }
    valid = [value for value in recall.values() if not np.isnan(value)]
    return ClassificationResult(
        tokens=tokens, features=tuple(features), labels=labels, confusion=confusion,
        recall=recall, balanced_accuracy=float(np.mean(valid)) if valid else float("nan"),
    )
