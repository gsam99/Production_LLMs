from types import SimpleNamespace

from app.routing import ModelRouter


def make_router(classifier_answer="COMPLEX"):
    # Skip __init__ so no real LLM client is created.
    router = ModelRouter.__new__(ModelRouter)
    router.settings = SimpleNamespace(primary_model="gpt-4o-mini", complex_model="gpt-4o")
    router.llm_classifier = SimpleNamespace(
        invoke=lambda prompt: SimpleNamespace(content=classifier_answer))
    return router


def test_short_message_routes_to_primary_model():
    assert make_router().route("hello") == "gpt-4o-mini"


def test_complex_keyword_routes_to_complex_model():
    text = "Please analyze the tradeoffs between these two architectures for me"
    assert make_router().route(text) == "gpt-4o"


def test_ambiguous_message_uses_llm_classifier():
    text = "What year did the first moon landing happen, roughly?"
    assert make_router("SIMPLE").route(text) == "gpt-4o-mini"
    assert make_router("COMPLEX").route(text) == "gpt-4o"
