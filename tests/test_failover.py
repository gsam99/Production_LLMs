from types import SimpleNamespace

from app.providers import (ModelRegistry, failover_chain, invoke_with_failover,
                           provider_for)
from app.resilience import CircuitBreaker


class FakeModel:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0

    def invoke(self, question):
        self.calls += 1
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return SimpleNamespace(content=self.behaviour,
                               usage_metadata={"input_tokens": 3, "output_tokens": 5})


def make_registry(models):
    return ModelRegistry(settings=None, factory=lambda provider, name, s: models[name])


def make_breakers(threshold=2):
    return {p: CircuitBreaker(p, failure_threshold=threshold, reset_seconds=60)
            for p in ("openai", "anthropic")}


def test_provider_inferred_from_model_name():
    assert provider_for("claude-sonnet-4-5") == "anthropic"
    assert provider_for("gpt-4o-mini") == "openai"


def test_failover_chain_skips_duplicate():
    assert failover_chain("gpt-4o", "claude-sonnet-4-5") == ["gpt-4o", "claude-sonnet-4-5"]
    assert failover_chain("claude-sonnet-4-5", "claude-sonnet-4-5") == ["claude-sonnet-4-5"]


def test_routed_model_is_the_one_called():
    models = {"gpt-4o": FakeModel("deep answer"), "gpt-4o-mini": FakeModel("quick answer"),
              "claude-sonnet-4-5": FakeModel("claude answer")}
    result = invoke_with_failover("q", failover_chain("gpt-4o", "claude-sonnet-4-5"),
                                  make_registry(models), make_breakers())
    assert result["answer"] == "deep answer"
    assert result["model_used"] == "gpt-4o"
    assert result["fallback_used"] is False
    assert models["gpt-4o-mini"].calls == 0


def test_provider_error_fails_over_and_counts_against_breaker():
    models = {"gpt-4o": FakeModel(TimeoutError("upstream timeout")),
              "claude-sonnet-4-5": FakeModel("claude answer")}
    breakers = make_breakers(threshold=2)
    result = invoke_with_failover("q", ["gpt-4o", "claude-sonnet-4-5"],
                                  make_registry(models), breakers)
    assert result["model_used"] == "claude-sonnet-4-5"
    assert result["fallback_used"] is True
    assert breakers["anthropic"].state == CircuitBreaker.CLOSED


def test_open_circuit_skips_provider_without_calling_it():
    models = {"gpt-4o": FakeModel(TimeoutError()), "claude-sonnet-4-5": FakeModel("ok")}
    breakers = make_breakers(threshold=2)
    registry = make_registry(models)
    for _ in range(2):
        invoke_with_failover("q", ["gpt-4o", "claude-sonnet-4-5"], registry, breakers)
    assert breakers["openai"].state == CircuitBreaker.OPEN
    calls_before = models["gpt-4o"].calls
    result = invoke_with_failover("q", ["gpt-4o", "claude-sonnet-4-5"], registry, breakers)
    assert models["gpt-4o"].calls == calls_before  # skipped while open
    assert result["model_used"] == "claude-sonnet-4-5"


def test_refusal_fails_over_but_does_not_trip_breaker():
    models = {"gpt-4o": FakeModel("I'm unable to help"), "claude-sonnet-4-5": FakeModel("ok")}
    breakers = make_breakers(threshold=1)
    result = invoke_with_failover("q", ["gpt-4o", "claude-sonnet-4-5"],
                                  make_registry(models), breakers)
    assert result["model_used"] == "claude-sonnet-4-5"
    assert breakers["openai"].state == CircuitBreaker.CLOSED


def test_all_models_failing_returns_none():
    models = {"gpt-4o": FakeModel(RuntimeError()), "claude-sonnet-4-5": FakeModel(RuntimeError())}
    result = invoke_with_failover("q", ["gpt-4o", "claude-sonnet-4-5"],
                                  make_registry(models), make_breakers())
    assert result["model_used"] == "none"
    assert result["input_tokens"] == 0


def test_registry_creates_each_client_once():
    created = []

    def factory(provider, name, settings):
        created.append((provider, name))
        return object()

    registry = ModelRegistry(settings=None, factory=factory)
    assert registry.get("gpt-4o") is registry.get("gpt-4o")
    registry.get("claude-sonnet-4-5")
    assert created == [("openai", "gpt-4o"), ("anthropic", "claude-sonnet-4-5")]
