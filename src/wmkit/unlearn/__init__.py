"""Machine unlearning toolkit: SCRUB / SCRUB+R, baselines, MIA and the NeurIPS 2023 score.

Typical use (see `scripts/demo_scrub_cifar.py`)::

    from wmkit.unlearn import ScrubConfig, scrub, evaluate_unlearning

    unlearned, history = scrub(copy_of_original, original, retain, forget, ScrubConfig(), device)
    report = evaluate_unlearning(unlearned, retain, forget, test, device)
"""

from wmkit.unlearn.baselines import (
    BaselineConfig,
    BaselineError,
    finetune_retain,
    negative_gradient,
    neggrad_plus,
    random_labels,
)
from wmkit.unlearn.data import (
    CifarArrays,
    TensorBatchLoader,
    UnlearnLoaders,
    UnlearnSplit,
    build_loaders,
    load_cifar10,
    load_or_train_original,
    make_split,
    resnet18_cifar,
    seed_everything,
)
from wmkit.unlearn.mia import (
    MembershipInferenceError,
    MiaResult,
    UnlearnReport,
    accuracy,
    evaluate_unlearning,
    membership_inference,
)
from wmkit.unlearn.score import (
    ScoreError,
    estimate_epsilon,
    forget_quality_from_epsilons,
    neurips2023_score,
)
from wmkit.unlearn.scrub import ScrubConfig, ScrubError, scrub

__all__ = [
    "BaselineConfig",
    "BaselineError",
    "CifarArrays",
    "MembershipInferenceError",
    "MiaResult",
    "ScoreError",
    "ScrubConfig",
    "ScrubError",
    "TensorBatchLoader",
    "UnlearnLoaders",
    "UnlearnReport",
    "UnlearnSplit",
    "accuracy",
    "build_loaders",
    "estimate_epsilon",
    "evaluate_unlearning",
    "finetune_retain",
    "forget_quality_from_epsilons",
    "load_cifar10",
    "load_or_train_original",
    "make_split",
    "membership_inference",
    "negative_gradient",
    "neggrad_plus",
    "neurips2023_score",
    "random_labels",
    "resnet18_cifar",
    "scrub",
    "seed_everything",
]
