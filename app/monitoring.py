import time
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    request_id: str
    question: str
    thread_id: str
    model_used: str
    cache_status: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    blocked: bool = False
    error: Optional[str] = None

class MonitoringService:


    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency_ms = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_history = []

    def start_request(self,question, thread_id):

        request_id = str(uuid.uuid4())
        start_time= time.time()

        logger.info(
            f"New request | "
            f"id={request_id} | "
            f"thread={thread_id} | "
            f"question={question[:50]}"

        )

        return request_id, start_time

    def record_request(self,metrics: RequestMetrics):
        self.total_requests += 1
        self.total_latency_ms +=metrics.latency_ms
        self.total_input_tokens += metrics.input_tokens
        self.total_output_tokens += metrics.output_tokens

        if metrics.error:
            self.total_errors +=1

        self.request_history.append(metrics)

        logger.info(
            f"Request complete | "
            f"id={metrics.request_id} | "
            f"model={metrics.model_used} | "
            f"cache={metrics.cache_status} | "
            f"latency={metrics.latency_ms:.1f}ms | "
            f"blocked={metrics.blocked} | "
            f"error={metrics.error}"
        )

    def record_security_block(self,request_id:str, reason:str):
        logger.warning(
            f"Request blocked | "
            f"id={request_id} | "
            f"reason={reason}"
        )

    def record_fallback_triggered(self,request_id:str):
        logger.warning(
            f"Fallback triggered | "
            f"id={request_id}"
        )

    def get_summary(self):
        if self.total_requests == 0:
            return {
                "total_requests": 0,
                "total_errors": 0,
                "error_rate": "0.0%",
                "avg_latency_ms": 0.0,
                "cache_hit_rate": "0.0%",
                "total_input_tokens": 0,
                "total_output_tokens": 0
            }
        avg_latency = self.total_latency_ms / self.total_requests
        error_rate=(self.total_errors / self.total_requests) * 100

        cache_hits = sum(
            1 for r in self.request_history
            if r.cache_status in ("exact_hit", "semantic_hit")
        )
        cache_hit_rate = (cache_hits / self.total_requests) * 100

        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": f"{error_rate:.1f}%",
            "avg_latency_ms": round(avg_latency, 2),
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens
        }

    def reset(self) -> None:
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency_ms = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_history = []







