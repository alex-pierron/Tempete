"""Typed errors shared by every hackathon task."""

from __future__ import annotations


class HackathonError(RuntimeError):
    """Base class of every expected hackathon failure (bad config, rejected file, server error)."""


class ConfigError(HackathonError):
    """A required setting is missing or malformed (API key, task id, data directory)."""


class SubmissionInvalidError(HackathonError):
    """A submission file breaks the task's format rules; raised before anything is sent."""


class SubmitError(HackathonError):
    """The server could not be reached or answered with an error."""


class CooldownError(HackathonError):
    """The previous submission to this task is too recent (server-side rate limit)."""
