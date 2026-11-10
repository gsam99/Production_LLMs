import pytest

from app.resilience import CircuitBreaker, ConcurrencyLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_breaker(threshold=3, reset=10.0):
    clock = FakeClock()
    return CircuitBreaker("openai", failure_threshold=threshold,
                          reset_seconds=reset, clock=clock), clock


def test_breaker_opens_after_threshold_failures():
    breaker, _ = make_breaker(threshold=3)
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state == CircuitBreaker.CLOSED
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.OPEN
    assert breaker.allow_request() is False


def test_success_resets_failure_count():
    breaker, _ = make_breaker(threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.CLOSED


def test_half_open_allows_one_trial_after_cooldown():
    breaker, clock = make_breaker(threshold=1, reset=10.0)
    breaker.record_failure()
    clock.now = 9.9
    assert breaker.allow_request() is False
    clock.now = 10.0
    assert breaker.state == CircuitBreaker.HALF_OPEN
    assert breaker.allow_request() is True
    assert breaker.allow_request() is False  # only one trial in flight


def test_half_open_success_closes_and_failure_reopens():
    breaker, clock = make_breaker(threshold=1, reset=10.0)
    breaker.record_failure()
    clock.now = 10.0
    assert breaker.allow_request()
    breaker.record_success()
    assert breaker.state == CircuitBreaker.CLOSED

    breaker.record_failure()
    clock.now = 20.0
    assert breaker.allow_request()
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.OPEN
    clock.now = 25.0
    assert breaker.allow_request() is False


def test_limiter_rejects_past_capacity_and_recovers():
    limiter = ConcurrencyLimiter(2)
    assert limiter.try_acquire()
    assert limiter.try_acquire()
    assert limiter.try_acquire() is False
    assert limiter.rejected == 1
    limiter.release()
    assert limiter.in_flight == 1
    assert limiter.try_acquire()


def test_limiter_requires_positive_capacity():
    with pytest.raises(ValueError):
        ConcurrencyLimiter(0)
