from typing import TypedDict, Optional
from langgraph_checkpoint_aws import DynamoDBSaver
from app.config import get_settings
from app.security import SecurityPipeline
from app.cache import CacheManager
from app.routing import ModelRouter
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
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
    input_tokens: Optional[int]
    output_tokens: Optional[int]


# 2. Shared instances, created once
security_pipeline = SecurityPipeline()
cache_manager = CacheManager()
model_router = ModelRouter()

primary_llm = ChatOpenAI(model=settings.primary_model, api_key=settings.openai_api_key)
fallback_llm = ChatAnthropic(model="claude-sonnet-4-5", api_key=settings.anthropic_api_key)
intent_llm = ChatOpenAI(model=settings.primary_model, api_key=settings.openai_api_key)

INTENT_CATEGORIES = [
    "general_question",
    "summarization",
    "code_help",
    "chitchat",
    "complaint_or_issue",
]


# 3. Helper function
def is_bad_response(text: str) -> bool:
    if not text or not text.strip():
        return True
    refusal_phrases = ["i can't help with that", "i'm unable to", "as an ai, i cannot"]
    if any(phrase in text.lower() for phrase in refusal_phrases):
        return True
    return False


# 4. Node functions
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
    try:
        response = primary_llm.invoke(state["cleaned_question"])
        answer = response.content

        if is_bad_response(answer):
            raise ValueError("Primary model returned an unusable response")

        usage = response.usage_metadata or {}
        return {
            "answer": answer,
            "model_used": settings.primary_model,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

    except Exception:
        try:
            response = fallback_llm.invoke(state["cleaned_question"])
            usage = response.usage_metadata or {}
            return {
                "answer": response.content,
                "model_used": "claude-sonnet-4-5",
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
        except Exception:
            return {
                "answer": "The request could not be processed by either the primary or fallback model.",
                "model_used": "none",
                "input_tokens": 0,
                "output_tokens": 0,
            }


def validate_output_node(state: AgentState) -> dict:
    cleaned_answer, warnings = security_pipeline.process_output(state["answer"])
    return {"answer": cleaned_answer}


def save_to_cache_node(state: AgentState) -> dict:
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