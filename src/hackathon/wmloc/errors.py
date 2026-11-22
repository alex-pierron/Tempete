"""Typed errors of the watermark localization task (all derive from the shared HackathonError)."""

from __future__ import annotations

from hackathon.common.errors import HackathonError


class MetricInputError(HackathonError, ValueError):
    """A mask given to the metric has the wrong shape, dtype or values."""


class FetchError(HackathonError):
    """The gated dataset could not be downloaded or unpacked (403, missing archive, bad zip)."""


class ManifestError(HackathonError):
    """A sample manifest is missing, malformed or points to a missing image."""


class LocalizerError(HackathonError):
    """A localizer received an input it cannot handle (bad image, bad message length)."""


class UnsupportedMethodError(LocalizerError):
    """No localizer or backend is registered for this watermarking method (yet)."""


class LocalBackendError(HackathonError):
    """The local-only Pixel Seal backend (wmkit + videoseal + weights) is unavailable."""
