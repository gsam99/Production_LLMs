"""Provider circuit breakers and request admission control.

Kept free of LLM/framework imports so it can be unit-tested on its own.
"""
import threading
import time
from typing import Callable


class CircuitBreaker:
    """Per-provider circuit breaker.

    closed    -> requests flow; consecutive failures are counted.
    open      -> requests are skipped until `reset_seconds` have passed.
    half_open -> one trial request is let through; success closes the
                 circuit, failure opens it again.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, name: str, failure_threshold: int = 5,
                 reset_seconds: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._trial_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if (self._state == self.OPEN
                and self._clock() - self._opened_at >= self.reset_seconds):
            self._state = self.HALF_OPEN
            self._trial_in_flight = False

    def allow_request(self) -> bool:
        with self._lock:
            self._maybe_half_open()
            if self._state == self.CLOSED:
                return True
            if self._state == self.HALF_OPEN and not self._trial_in_flight:
                self._trial_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._state = self.CLOSED
            self._failures = 0
            self._trial_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._open()
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._open()

    def _open(self) -> None:
        self._state = self.OPEN
        self._opened_at = self._clock()
        self._trial_in_flight = False


class ConcurrencyLimiter:
    """Caps in-flight requests; callers shed load (HTTP 503) when full."""

    def __init__(self, max_in_flight: int):
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        self.max_in_flight = max_in_flight
        self._in_flight = 0
        self._rejected = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_flight >= self.max_in_flight:
                self._rejected += 1
                return False
            self._in_flight += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def rejected(self) -> int:
        with self._lock:
            return self._rejected
