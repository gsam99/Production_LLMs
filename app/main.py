from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime
import time
import uuid

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.monitoring import MonitoringService, RequestMetrics

from app.agent import primary_llm, cache_manager, fallback_llm, build_graph
from app.config import get_settings
from app.models import (ChatRequest, ChatResponse,
                        HealthResponse, MetricsResponse,
                        ErrorResponse)

app = FastAPI(
    title="LLM Production API",
    description="LLM Production API",
)

_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(_frontend / "index.html"))

settings = get_settings()
graph = build_graph()
monitoring_service = MonitoringService()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health", response_model=HealthResponse)
async def health():
    check={}

    check_status=True

    try:
        cache_manager.collection.count()
        check["chromadb"]="healthy"
    except Exception as e:
        check["chromadb"]="unhealthy: "+str(e)

    try:
        primary_llm.invoke("ping")
        check["primary_llm"]="healthy"
    except Exception as e:
        check["primary_llm"]="unhealthy: "+str(e)

    try:
        fallback_llm.invoke("ping")
        check["fallback_llm"]="healthy"
    except Exception as e:
        check["fallback_llm"]="unhealthy: "+str(e)

    for k,v in check.items():
        if "unhealthy" in v:
            check_status=False

    return HealthResponse(
        status=("healthy" if check_status else "unhealthy"),
        environment=settings.app_env,
        version="1.0.0",
        checks=check

    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit)
async def chat(request:Request,message: ChatRequest):
    start_time = time.time()
    initial_state = {
        "question": message.message,
        "thread_id": message.thread_id,
        "cleaned_question": None,
        "blocked": False,
        "block_reason": None,
        "cache_status": None,
        "intent": None,
        "answer": None,
        "model_used": None,
        "input_tokens": None,
        "output_tokens": None,
    }
    result=graph.invoke(initial_state,
                        config={"configurable": {"thread_id": message.thread_id}})
    latency_ms=round((time.time() - start_time) * 1000)

    monitoring_service.record_request(RequestMetrics(
        request_id=str(uuid.uuid4()),
        question=message.message,
        thread_id=message.thread_id,
        model_used=result["model_used"] or "unknown",
        cache_status=result["cache_status"] or "miss",
        latency_ms=latency_ms,
        input_tokens=result.get("input_tokens") or 0,
        output_tokens=result.get("output_tokens") or 0,
        blocked=result["blocked"],
        error=result["block_reason"] if result["blocked"] else None
    ))

    if result["blocked"]==True:
        raise HTTPException(status_code=400, detail=result["block_reason"])

    if result["answer"]==None:
        raise HTTPException(status_code=500, detail="No answer was produced")

    return ChatResponse(
        response=result["answer"],
        thread_id=message.thread_id,
        model_used=result["model_used"] or "unknown",
        cached=result["cache_status"] in ("exact_hit", "semantic_hit"),
        processing_time_ms=latency_ms,
        timestamp=str(datetime.utcnow()),
        intent= result["intent"]
    )

@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    return monitoring_service.get_summary()








