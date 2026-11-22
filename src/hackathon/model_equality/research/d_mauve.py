"""MAUVE between ref endpoints (Lucas' proposal, tested on the free ref; team D).

MAUVE compares the distribution of the replies of two endpoints in an embedding space: the pooled
embeddings are quantised with k-means, each endpoint becomes a histogram over the clusters, and the score is
the area under the divergence frontier of the two histograms (1.0 = indistinguishable, 0.0 = disjoint).
This module is a compact re-implementation of Pillutla et al. (2021), with `all-MiniLM-L6-v2` as the
featuriser (local cache) instead of GPT-2-large: a deviation, stated in the report.

CRITERIA FROZEN BEFORE THE RUN (Lucas' expectation: useful for families, blind to fp8 and to weak LoRA):
* a ref endpoint X is FLAGGED different from ref-01 when MAUVE(ref-01, X) is below the 2.5th percentile
  of the null distribution, the null being MAUVE between two random halves of the ref-01 replies
  (`NULL_SPLITS` splits);
* expected: LoRA 1.5 (ref-07) flagged; fp8 (ref-02) and LoRA 0.3 (ref-09) NOT flagged. Any flagged
  disguise (ref-03, ref-05, ref-06, ref-08) is reported as a confusion.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

FEATURISER = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_RATIO = 10
"""One cluster per `CLUSTER_RATIO` pooled points (the MAUVE default)."""
SCALING = 5.0
MIXTURE_POINTS = 25
PCA_VARIANCE = 0.9
NULL_SPLITS = 200
NULL_PERCENTILE = 2.5
SEED = 20261121
BATCH = 32
FloatArray = NDArray[np.float64]


def embed(texts: Sequence[str]) -> FloatArray:
    """Mean-pooled sentence embeddings (L2-normalised) from the local featuriser."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(FEATURISER, local_files_only=True)
    model = AutoModel.from_pretrained(FEATURISER, local_files_only=True).eval()
    out: list[FloatArray] = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            batch = tokenizer(list(texts[start:start + BATCH]), padding=True, truncation=True, max_length=256,
                              return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            out.append(pooled.numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


def _kl(p: FloatArray, q: FloatArray) -> float:
    """KL(p || q) over histograms (terms with p = 0 are dropped; q > 0 wherever p > 0 by construction)."""
    keep = p > 0
    return float(np.sum(p[keep] * np.log(p[keep] / q[keep])))


def mauve_from_embeddings(first: FloatArray, second: FloatArray, seed: int = SEED) -> float:
    """MAUVE score of two embedding sets (area under the divergence frontier, in [0, 1])."""
    pooled = np.concatenate([first, second], axis=0)
    reduced = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=seed).fit_transform(pooled)
    clusters = max(2, len(pooled) // CLUSTER_RATIO // 2)
    labels = KMeans(n_clusters=clusters, n_init=3, random_state=seed).fit_predict(reduced)
    p = np.bincount(labels[: len(first)], minlength=clusters).astype(np.float64)
    q = np.bincount(labels[len(first):], minlength=clusters).astype(np.float64)
    p, q = p / p.sum(), q / q.sum()
    xs, ys = [0.0], [1.0]
    for lam in np.linspace(0.0, 1.0, MIXTURE_POINTS)[1:-1]:
        mixture = lam * p + (1.0 - lam) * q
        xs.append(float(np.exp(-SCALING * _kl(q, mixture))))
        ys.append(float(np.exp(-SCALING * _kl(p, mixture))))
    xs.append(1.0)
    ys.append(0.0)
    order = np.argsort(xs)
    return float(np.trapezoid(np.asarray(ys)[order], np.asarray(xs)[order]))


def null_scores(reference: FloatArray, splits: int = NULL_SPLITS, seed: int = SEED) -> list[float]:
    """MAUVE between random halves of the reference replies (the same-model null)."""
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(splits):
        order = rng.permutation(len(reference))
        half = len(reference) // 2
        scores.append(mauve_from_embeddings(reference[order[:half]], reference[order[half:2 * half]], seed))
    return scores
