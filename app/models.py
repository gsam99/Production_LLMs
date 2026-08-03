from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str= Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's message to the agent"
    )

    thread_id:str= Field(
        default="default",
        description="Conversation ID"
    )

class ChatResponse(BaseModel):
    response: str
    thread_id:str
    model_used:str
    cached:bool=False
    processing_time_ms: float
    timestamp:str= Field(default_factory=datetime.utcnow)
    intent:Optional[str]=None

class HealthResponse(BaseModel):
    status: str ="healthy"
    environment:str
    version:str="1.0.0"
    checks:dict={}

class MetricsResponse(BaseModel):
    total_requests: int
    total_errors:int
    error_rate:str
    avg_latency_ms:float
    cache_hit_rate:str
    total_input_tokens: int
    total_output_tokens:int

class ErrorResponse(BaseModel):
    error: str
    detail:str|None=None
    request_id:str|None=None