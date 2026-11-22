"""Typed errors of the research package."""

from __future__ import annotations


class ResearchError(RuntimeError):
    """Base class of every research-package failure."""
