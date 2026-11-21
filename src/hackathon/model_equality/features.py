"""Text normalisation and per-response views used by the kernels.

The API only returns text, so (as in the MET paper's audit of real APIs) comparisons happen in
character and word space rather than token space. Each response is turned into several *views*
that make different trade-offs between sensitivity to the model and invariance to the disguises
(system prompt, persona, watermark, temperature):

* `chars`   first `CHAR_LENGTH` characters of the whitespace-normalised text (padding = -1).
* `words`   first `WORD_LENGTH` lower-cased word / punctuation tokens, as integer ids.
* `ngrams`  L2-normalised sparse counts of word unigrams and bigrams: a bag of n-grams (the
            k-spectrum kernel family of the MET paper). Unlike positional Hamming it is invariant
            to a prefix or a shifted start, so a persona preamble does not wreck it.
* `key`     the answer itself for constrained prompts: the first number for `arith`, the first
            word for `format`, otherwise the whole normalised text. Insensitive to preambles
            such as "Arr, matey!" that a persona adds before the real answer.
* `first`   first word of the reply (very sensitive to personas; kept to *measure* that).
* `loglen`  log(1 + number of words): the reply length.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import sparse

from hackathon.model_equality.prompts.bank import CATEGORY_ARITH, CATEGORY_FORMAT

CHAR_LENGTH = 400
"""Characters compared by the character-level Hamming kernel (about 100 tokens)."""
WORD_LENGTH = 64
"""Words compared by the word-level Hamming kernel."""
PAD_VALUE = -1
"""Padding of short responses; two paddings match, as in the official MET code."""
WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
WORD_ONLY_PATTERN = re.compile(r"\w+", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
EMPTY_KEY = ""
EMPTY_TOKEN = "<empty>"
"""Stand-in word of an empty reply, so two empty replies compare equal instead of undefined."""
BIGRAM_SEPARATOR = "\x1f"

IntMatrix = npt.NDArray[np.int64]
IntVector = npt.NDArray[np.int64]
FloatVector = npt.NDArray[np.float64]


def normalize_text(text: str) -> str:
    """Strip and collapse whitespace (line breaks included)."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize_words(text: str) -> list[str]:
    """Lower-cased words and punctuation marks."""
    return [token.lower() for token in WORD_PATTERN.findall(text)]


def answer_key(category: str, text: str) -> str:
    """The comparable core of an answer (see module docstring); empty when none is found."""
    if category == CATEGORY_ARITH:
        match = NUMBER_PATTERN.search(text)
        return match.group(0).replace(",", "") if match else EMPTY_KEY
    if category == CATEGORY_FORMAT:
        match = WORD_ONLY_PATTERN.search(text)
        return match.group(0).lower() if match else EMPTY_KEY
    return normalize_text(text).lower()


def first_word(text: str) -> str:
    """First word of the reply, lower-cased (empty for an empty reply)."""
    match = WORD_ONLY_PATTERN.search(text)
    return match.group(0).lower() if match else EMPTY_KEY


def encode_chars(texts: Sequence[str], length: int = CHAR_LENGTH) -> IntMatrix:
    """Code points of the normalised texts, truncated / right-padded with `PAD_VALUE`."""
    out = np.full((len(texts), length), PAD_VALUE, dtype=np.int64)
    for row, text in enumerate(texts):
        kept = [ord(char) for char in normalize_text(text)[:length]]
        out[row, : len(kept)] = kept
    return out


def encode_words(
    texts: Sequence[str], vocabulary: dict[str, int], length: int = WORD_LENGTH
) -> IntMatrix:
    """Word ids (ids assigned on first sight in `vocabulary`), truncated / padded like chars."""
    out = np.full((len(texts), length), PAD_VALUE, dtype=np.int64)
    for row, text in enumerate(texts):
        for column, token in enumerate(tokenize_words(text)[:length]):
            out[row, column] = vocabulary.setdefault(token, len(vocabulary))
    return out


def encode_labels(values: Sequence[str]) -> IntVector:
    """Map strings to dense integer ids (equality of ids = equality of strings)."""
    ids: dict[str, int] = {}
    return np.array([ids.setdefault(value, len(ids)) for value in values], dtype=np.int64)


def log_length(texts: Sequence[str]) -> FloatVector:
    """log(1 + word count) of each text."""
    return np.array([math.log1p(len(tokenize_words(text))) for text in texts], dtype=np.float64)


def ngram_matrix(texts: Sequence[str]) -> sparse.csr_matrix:
    """Row-normalised counts of word unigrams and bigrams, one row per text."""
    vocabulary: dict[str, int] = {}
    rows: list[int] = []
    columns: list[int] = []
    for row, text in enumerate(texts):
        tokens = tokenize_words(text) or [EMPTY_TOKEN]
        grams = tokens + [a + BIGRAM_SEPARATOR + b for a, b in itertools.pairwise(tokens)]
        for gram in grams:
            rows.append(row)
            columns.append(vocabulary.setdefault(gram, len(vocabulary)))
    counts = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(texts), max(len(vocabulary), 1))
    ).tocsr()
    norms = np.sqrt(np.asarray(counts.multiply(counts).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms).dot(counts).tocsr()


@dataclass(frozen=True)
class Views:
    """All kernel inputs of a group of responses, row-aligned."""

    chars: IntMatrix
    words: IntMatrix
    ngrams: sparse.csr_matrix
    key: IntVector
    first: IntVector
    loglen: FloatVector


def build_views(category: str, texts: Sequence[str]) -> Views:
    """Compute every view of the responses of one category."""
    return Views(
        chars=encode_chars(texts),
        words=encode_words(texts, {}),
        ngrams=ngram_matrix(texts),
        key=encode_labels([answer_key(category, text) for text in texts]),
        first=encode_labels([first_word(text) for text in texts]),
        loglen=log_length(texts),
    )
