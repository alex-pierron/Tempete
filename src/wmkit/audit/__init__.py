"""Black-box model auditing: Model Equality Testing and a cached, rate-limited API client."""

from wmkit.audit.api import (
    CachedCompletionClient,
    CompletionClientError,
    CompletionRequestError,
)
from wmkit.audit.met import (
    METError,
    METResult,
    Sample,
    SampleDrawFn,
    mmd_hamming,
    power_curve,
    two_sample_test,
)

__all__ = [
    "CachedCompletionClient",
    "CompletionClientError",
    "CompletionRequestError",
    "METError",
    "METResult",
    "Sample",
    "SampleDrawFn",
    "mmd_hamming",
    "power_curve",
    "two_sample_test",
]
