"""Model registry and cross-provider failover.

Model names come from config, so pointing a tier at a different model or
provider is a config change. The provider is inferred from the model name.
"""
from typing import Any, Callable, Iterable, Optional

from app.resilience import CircuitBreaker


def provider_for(model: str) -> str:
    return "anthropic" if model.lower().startswith("claude") else "openai"


def _default_factory(provider: str, model: str, settings: Any):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, api_key=settings.openai_api_key)


class ModelRegistry:
    """Creates one chat client per model name, lazily, and caches it."""

    def __init__(self, settings: Any,
                 factory: Optional[Callable[[str, str, Any], Any]] = None):
        self.settings = settings
        self._factory = factory or _default_factory
        self._clients: dict[str, Any] = {}

    def get(self, model: str):
        if model not in self._clients:
            self._clients[model] = self._factory(provider_for(model), model, self.settings)
        return self._clients[model]


def is_bad_response(text: Optional[str]) -> bool:
    if not text or not text.strip():
        return True
    refusal_phrases = ["i can't help with that", "i'm unable to", "as an ai, i cannot"]
    return any(phrase in text.lower() for phrase in refusal_phrases)


def failover_chain(chosen: str, failover: str) -> list[str]:
    """The routed model first, then the failover model if it is different."""
    return [chosen] if failover == chosen else [chosen, failover]


def invoke_with_failover(question: str, models: Iterable[str],
                         registry: ModelRegistry,
                         breakers: dict[str, CircuitBreaker]) -> dict:
    """Try each model in order, skipping providers whose circuit is open.

    Provider errors count against that provider's breaker. An unusable
    answer (empty or a refusal) moves on to the next model but does not
    count as a provider outage.
    """
    models = list(models)
    for model in models:
        provider = provider_for(model)
        breaker = breakers.get(provider)
        if breaker is not None and not breaker.allow_request():
            continue
        try:
            response = registry.get(model).invoke(question)
        except Exception:
            if breaker is not None:
                breaker.record_failure()
            continue
        if breaker is not None:
            breaker.record_success()
        if is_bad_response(response.content):
            continue
        usage = getattr(response, "usage_metadata", None) or {}
        return {
            "answer": response.content,
            "model_used": model,
            "fallback_used": model != models[0],
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
    return {
        "answer": "The request could not be processed by any available model.",
        "model_used": "none",
        "fallback_used": True,
        "input_tokens": 0,
        "output_tokens": 0,
    }
