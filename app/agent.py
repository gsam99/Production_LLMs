from typing import TypedDict, Optional
from langgraph_checkpoint_aws import DynamoDBSaver
from app.config import get_settings
from app.security import SecurityPipeline
from app.cache import CacheManager
from app.routing import ModelRouter
from app.providers import (ModelRegistry, provider_for, failover_chain,
                           invoke_with_failover)
from app.resilience import CircuitBreaker
from langgraph.graph import StateGraph, END

settings = get_settings()


# 1. State definition
class AgentState(TypedDict):
    question: str
    thread_id: str
    cleaned_question: Optional[str]
    blocked: bool
    block_reason: Optional[str]
    cache_status: Optional[str]
    intent: Optional[str]
    answer: Optional[str]
    model_used: Optional[str]
    fallback_used: Optional[bool]
    input_tokens: Optional[int]
    output_tokens: Optional[int]


# 2. Shared instances, created once
security_pipeline = SecurityPipeline()
cache_manager = CacheManager()
model_router = ModelRouter()

model_registry = ModelRegistry(settings)
primary_llm = model_registry.get(settings.primary_model)
fallback_llm = model_registry.get(settings.failover_model)
intent_llm = primary_llm

# One breaker per provider, so an OpenAI outage stops sending traffic to
# OpenAI for a cool-down period instead of failing every request first.
provider_breakers = {
    provider: CircuitBreaker(
        provider,
        failure_threshold=settings.breaker_failure_threshold,
        reset_seconds=settings.breaker_reset_seconds,
    )
    for provider in {provider_for(m) for m in (
        settings.primary_model, settings.complex_model, settings.failover_model)}
}

INTENT_CATEGORIES = [
    "general_question",
    "summarization",
    "code_help",
    "chitchat",
    "complaint_or_issue",
]


# 3. Node functions
def security_check_node(state: AgentState) -> dict:
    is_safe, text, reason = security_pipeline.process_input(state["question"])
    if not is_safe:
        return {"blocked": True, "block_reason": reason}
    return {"blocked": False, "cleaned_question": text}


def cache_lookup_node(state: AgentState) -> dict:
    answer, status = cache_manager.get_answer(state["cleaned_question"])
    if answer:
        return {"answer": answer, "cache_status": status, "model_used": "cache"}
    return {"cache_status": status}


def intent_detection_node(state: AgentState) -> dict:
    prompt = (
        f"Classify this message into exactly one category: {', '.join(INTENT_CATEGORIES)}.\n"
        f"Reply with only the category name.\n\n"
        f"Message: {state['cleaned_question']}"
    )
    response = intent_llm.invoke(prompt)
    intent = response.content.strip().lower()

    if intent in INTENT_CATEGORIES:
        return {"intent": intent}
    return {"intent": "general_question"}


def route_model_node(state: AgentState) -> dict:
    model = model_router.route(state["cleaned_question"])
    return {"model_used": model}


def call_llm_node(state: AgentState) -> dict:
    # Call the model the router picked; fail over to the failover model when
    # it errors, its provider's circuit is open, or the answer is unusable.
    chosen = state.get("model_used") or settings.primary_model
    return invoke_with_failover(
        state["cleaned_question"],
        failover_chain(chosen, settings.failover_model),
        model_registry,
        provider_breakers,
    )


def validate_output_node(state: AgentState) -> dict:
    cleaned_answer, warnings = security_pipeline.process_output(state["answer"])
    return {"answer": cleaned_answer}


def save_to_cache_node(state: AgentState) -> dict:
    # Don't cache the "no model available" message during an outage.
    if state.get("model_used") != "none":
        cache_manager.set(state["cleaned_question"], state["answer"])
    return {}


# 5. Edge routing functions (NOT nodes — used in add_conditional_edges only)
def route_after_security(state: AgentState) -> str:
    if state["blocked"]:
        return "end"
    return "cache_lookup"


def route_after_cache(state: AgentState) -> str:
    if state["cache_status"] in ("exact_hit", "semantic_hit"):
        return "end"
    return "intent_detection"


# 6. Graph construction
def build_graph():
    """
    Builds and compiles the LangGraph pipeline.
    Called once at app startup in main.py.
    """
    builder = StateGraph(AgentState)

    # Register all node functions
    builder.add_node("security_check", security_check_node)
    builder.add_node("cache_lookup", cache_lookup_node)
    builder.add_node("intent_detection", intent_detection_node)
    builder.add_node("route_model", route_model_node)
    builder.add_node("call_llm", call_llm_node)
    builder.add_node("validate_output", validate_output_node)
    builder.add_node("save_to_cache", save_to_cache_node)

    # Always start at security check
    builder.set_entry_point("security_check")

    # After security check — blocked? stop. Safe? go to cache lookup
    builder.add_conditional_edges(
        "security_check",
        route_after_security,
        {
            "end": END,
            "cache_lookup": "cache_lookup"
        }
    )

    # After cache lookup — hit? stop. Miss? go to intent detection
    builder.add_conditional_edges(
        "cache_lookup",
        route_after_cache,
        {
            "end": END,
            "intent_detection": "intent_detection"
        }
    )

    # Sequential flow for the rest of the pipeline
    builder.add_edge("intent_detection", "route_model")
    builder.add_edge("route_model", "call_llm")
    builder.add_edge("call_llm", "validate_output")
    builder.add_edge("validate_output", "save_to_cache")
    builder.add_edge("save_to_cache", END)

    checkpointer = DynamoDBSaver(
        table_name="rag-production",
        region_name="us-east-1"
    )

    return builder.compile()