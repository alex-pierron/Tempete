"""Typed errors of the experiment lab (all are `HackathonError`, so CLIs print them cleanly)."""

from __future__ import annotations

from hackathon.common.errors import HackathonError


class LabError(HackathonError):
    """Base class of every expected failure of the experiment lab."""


class SpecError(LabError):
    """An experiment TOML is malformed, or asks for something the lab forbids."""


class UnknownTaskError(LabError):
    """The task name matches no registered task adapter."""


class RegistryError(LabError):
    """The registry file is corrupt, or a run id is unknown or ambiguous."""


class RunError(LabError):
    """A run cannot be started or finalised (missing file, wrong state, failed step)."""


class RemoteError(LabError):
    """ssh, rsync or Slurm failed or is unavailable (JURECA side)."""


class SubmitRefusedError(LabError):
    """The lab refuses a submission (run not finished, already submitted, cooldown)."""
