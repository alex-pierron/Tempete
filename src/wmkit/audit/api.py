"""Cached, rate-limited client for a slow text-completion API.

Built for a competition endpoint limited to one request every 15 minutes: every answer is
appended to a JSONL file as soon as it arrives (fsync), so a crash or a restart never loses a
paid request, replays are free, and the minimum interval between two REAL requests is
enforced across process restarts (the last request time is stored with each answer).

Cache key: `(prompt, replicate)`. Model Equality Testing needs several independent completions
of the same prompt; `replicate` numbers them (0, 1, 2, ...) so that they are cached separately.

Single-process, single-thread by design: concurrent writers would break both the rate limit
accounting and the append-only file.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

DEFAULT_MIN_INTERVAL_S: Final[float] = 0.0
"""No throttling unless the caller asks for it."""

CACHE_FIELD_PROMPT: Final[str] = "prompt"
CACHE_FIELD_REPLICATE: Final[str] = "replicate"
CACHE_FIELD_COMPLETION: Final[str] = "completion"
CACHE_FIELD_REQUESTED_AT: Final[str] = "requested_at"


class CompletionClientError(RuntimeError):
    """Base class of the errors raised by `CachedCompletionClient`."""


class CompletionRequestError(CompletionClientError):
    """The wrapped API function raised or returned a non-string; nothing was cached."""


class CachedCompletionClient:
    """Wrap `fn(prompt) -> completion` with a disk cache and a minimum request interval.

    Attributes:
        n_requests: Real calls to `fn` made by this instance (failed attempts included).
        n_cache_hits: Answers served from the cache by this instance.
        n_corrupt_lines: Unreadable cache lines skipped at load time (e.g. a torn last line).
    """

    def __init__(
        self,
        fn: Callable[[str], str],
        cache_path: str | Path,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the client and load the existing cache.

        Args:
            fn: The real API call, `prompt -> completion`.
            cache_path: JSONL cache file (created with its parents when missing).
            min_interval_s: Minimum seconds between the starts of two real requests
                (0 disables throttling). It also applies to the first request of a new
                process, relative to the last request recorded in the cache.
            clock: Wall-clock seconds; injectable so that tests never really sleep.
            sleep: Sleep function; injectable for the same reason.
        """
        if min_interval_s < 0.0:
            raise CompletionClientError(f"min_interval_s must be >= 0, got {min_interval_s}")
        self._fn = fn
        self._path = Path(cache_path)
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._cache: dict[tuple[str, int], str] = {}
        self._last_request_at: float | None = None
        self._needs_newline = False
        self.n_requests = 0
        self.n_cache_hits = 0
        self.n_corrupt_lines = 0
        self._load()

    @property
    def n_cached(self) -> int:
        """Number of distinct `(prompt, replicate)` answers currently cached."""
        return len(self._cache)

    @property
    def last_request_at(self) -> float | None:
        """Wall-clock time of the last real request (this process or a previous one)."""
        return self._last_request_at

    def seconds_until_ready(self) -> float:
        """Return how long a real request would currently have to wait (0 when ready)."""
        if self._last_request_at is None or self._min_interval_s == 0.0:
            return 0.0
        return max(0.0, self._last_request_at + self._min_interval_s - self._clock())

    def __call__(self, prompt: str, replicate: int = 0) -> str:
        """Return the completion of `prompt`, calling the API only on a cache miss.

        Args:
            prompt: Prompt text.
            replicate: Index of the independent sample of this prompt (0, 1, ...).

        Returns:
            The cached or fresh completion.

        Raises:
            CompletionRequestError: If the API call fails; the attempt still counts for the
                rate limit, and nothing is cached so that a retry calls the API again.
        """
        key = (prompt, replicate)
        cached = self._cache.get(key)
        if cached is not None:
            self.n_cache_hits += 1
            return cached
        self._wait_for_slot()
        requested_at = self._clock()
        self._last_request_at = requested_at
        self.n_requests += 1
        try:
            completion = self._fn(prompt)
        except Exception as error:
            raise CompletionRequestError(f"API call failed for replicate {replicate}") from error
        if not isinstance(completion, str):
            raise CompletionRequestError(
                f"API function must return str, got {type(completion).__name__}"
            )
        self._cache[key] = completion
        self._append(prompt, replicate, completion, requested_at)
        return completion

    def sample(self, prompt: str, n_samples: int) -> list[str]:
        """Return `n_samples` independent completions (replicates 0 .. n_samples - 1)."""
        return [self(prompt, replicate) for replicate in range(n_samples)]

    def _wait_for_slot(self) -> None:
        """Sleep until the minimum interval since the previous real request has elapsed."""
        remaining = self.seconds_until_ready()
        if remaining > 0.0:
            self._sleep(remaining)

    def _load(self) -> None:
        """Read the JSONL cache, skipping (and counting) unreadable lines."""
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            return
        raw = self._path.read_bytes()
        self._needs_newline = bool(raw) and not raw.endswith(b"\n")
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (str(record[CACHE_FIELD_PROMPT]), int(record[CACHE_FIELD_REPLICATE]))
                self._cache[key] = str(record[CACHE_FIELD_COMPLETION])
                requested_at = float(record[CACHE_FIELD_REQUESTED_AT])
            except (ValueError, KeyError, TypeError):
                self.n_corrupt_lines += 1
                continue
            if self._last_request_at is None or requested_at > self._last_request_at:
                self._last_request_at = requested_at

    def _append(self, prompt: str, replicate: int, completion: str, requested_at: float) -> None:
        """Append one record and force it to disk before returning."""
        record = {
            CACHE_FIELD_PROMPT: prompt,
            CACHE_FIELD_REPLICATE: replicate,
            CACHE_FIELD_COMPLETION: completion,
            CACHE_FIELD_REQUESTED_AT: requested_at,
        }
        prefix = "\n" if self._needs_newline else ""
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(prefix + json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._needs_newline = False
