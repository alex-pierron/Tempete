"""Learned "perturbation direction" scores (team D).

Idea: the LoRA adapters of the task are one adapter applied at strengths 0.3, 0.7 and 1.5, so a
weak LoRA moves the model in the SAME direction as a strong one, only less far. A text classifier
trained to separate the strong LoRA endpoints (T3, T4) from S therefore gives every answer a score
that should be monotone in the strength: S low, T2 (0.3) in between, T4 high. fp8 has no such
direction (its error is a fixed random perturbation of the weights), so it is the harder case.

Two variants are evaluated, both with folds grouped by PROMPT so that a score is never produced
by a model that saw the same prompt:

* `transfer`: trained on T3 / T4 against S only (labels of T1 and T2 are never used). This is the
  only variant that can be used on the scored endpoints, where no label exists.
* `oracle`: trained on the endpoint under test against S (upper bound: is there ANY direction
  that generalises across prompts?).

A score is a signed logistic-regression margin; a pair distance is the absolute difference of the
mean scores over the sampled cells (see `d_eval.scalar_distances`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from hackathon.model_equality.research.d_eval import Design

DIRECTION_SEED = 20261121
FOLDS = 5
REGULARISATION = 1.0
WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (2, 4)
MAX_WORD_FEATURES = 4000
MAX_CHAR_FEATURES = 6000
MIN_DOCUMENT_FREQUENCY = 2
FloatArray = NDArray[np.float64]


def prompt_folds(design: Design, folds: int, seed: int) -> NDArray[np.int64]:
    """Fold index of every cell; all repeats of one prompt share a fold."""
    prompts = sorted({prompt_id for prompt_id, _ in design.cells})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(prompts))
    fold_of_prompt = {prompts[i]: rank % folds for rank, i in enumerate(order)}
    return np.asarray([fold_of_prompt[p] for p, _ in design.cells], dtype=np.int64)


def out_of_fold_scores(
    design: Design,
    positive_endpoints: Sequence[str],
    negative_endpoints: Sequence[str],
    folds: int = FOLDS,
    seed: int = DIRECTION_SEED,
) -> FloatArray:
    """Out-of-fold margin of every answer of every endpoint, shape (endpoints, cells).

    Args:
        design: Answers on the common cells.
        positive_endpoints: Endpoints labelled 1 when training.
        negative_endpoints: Endpoints labelled 0 when training.
        folds: Number of prompt-grouped folds.
        seed: Fold assignment seed.
    """
    fold_of_cell = prompt_folds(design, folds, seed)
    index = {e: i for i, e in enumerate(design.endpoints)}
    scores = np.zeros((len(design.endpoints), len(design.cells)), dtype=np.float64)
    for fold in range(folds):
        train_cells = np.flatnonzero(fold_of_cell != fold)
        test_cells = np.flatnonzero(fold_of_cell == fold)
        texts: list[str] = []
        labels: list[int] = []
        for endpoint, label in [(e, 1) for e in positive_endpoints] + [
            (e, 0) for e in negative_endpoints
        ]:
            texts.extend(design.responses[endpoint][c] for c in train_cells)
            labels.extend([label] * len(train_cells))
        word = TfidfVectorizer(
            ngram_range=WORD_NGRAMS, max_features=MAX_WORD_FEATURES,
            min_df=MIN_DOCUMENT_FREQUENCY, sublinear_tf=True,
        )
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=CHAR_NGRAMS, max_features=MAX_CHAR_FEATURES,
            min_df=MIN_DOCUMENT_FREQUENCY, sublinear_tf=True,
        )
        matrix = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
        model = LogisticRegression(
            C=REGULARISATION, class_weight="balanced", max_iter=2000, random_state=seed
        )
        model.fit(matrix, np.asarray(labels))
        for endpoint in design.endpoints:
            test_texts = [design.responses[endpoint][c] for c in test_cells]
            features = hstack([word.transform(test_texts), char.transform(test_texts)]).tocsr()
            scores[index[endpoint], test_cells] = model.decision_function(features)
    return scores


def _fit_vectorisers_and_model(
    texts: Sequence[str], labels: Sequence[int], seed: int
) -> tuple[TfidfVectorizer, TfidfVectorizer, LogisticRegression]:
    """Word and character TF-IDF vectorisers plus a balanced logistic regression on them."""
    word = TfidfVectorizer(
        ngram_range=WORD_NGRAMS, max_features=MAX_WORD_FEATURES,
        min_df=MIN_DOCUMENT_FREQUENCY, sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=CHAR_NGRAMS, max_features=MAX_CHAR_FEATURES,
        min_df=MIN_DOCUMENT_FREQUENCY, sublinear_tf=True,
    )
    matrix = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    model = LogisticRegression(
        C=REGULARISATION, class_weight="balanced", max_iter=2000, random_state=seed
    )
    model.fit(matrix, np.asarray(labels))
    return word, char, model


def fit_and_score(
    train: Design,
    test: Design,
    positive_endpoints: Sequence[str],
    negative_endpoints: Sequence[str],
    seed: int = DIRECTION_SEED,
) -> FloatArray:
    """Train on every cell of `train`, score every answer of `test`: shape (endpoints, cells).

    The two designs must not share prompts (fresh prompts = a real out-of-sample test).
    """
    texts: list[str] = []
    labels: list[int] = []
    for endpoint, label in [(e, 1) for e in positive_endpoints] + [
        (e, 0) for e in negative_endpoints
    ]:
        texts.extend(train.responses[endpoint])
        labels.extend([label] * len(train.cells))
    word, char, model = _fit_vectorisers_and_model(texts, labels, seed)
    scores = np.zeros((len(test.endpoints), len(test.cells)), dtype=np.float64)
    for index, endpoint in enumerate(test.endpoints):
        features = hstack([word.transform(test.responses[endpoint]),
                           char.transform(test.responses[endpoint])]).tocsr()
        scores[index] = model.decision_function(features)
    return scores
