"""Load test for the /chat endpoint.

Run against a local server (uvicorn app.main:app --port 8000):

    uvx locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --users 50 --spawn-rate 10 --run-time 2m --headless --csv loadtest/results

Locust reports requests/sec and p50/p95/p99 latency per request type, so
cache hits and model calls can be compared. Real provider calls cost money:
keep runs short, or raise RATE_LIMIT and MAX_CONCURRENT_REQUESTS in .env so
the limits themselves are what you are measuring.
"""
import random
import uuid

from locust import HttpUser, between, task

# A small fixed set of questions, asked repeatedly, measures the cache path.
REPEATED_QUESTIONS = [
    "What is a circuit breaker in distributed systems?",
    "Explain the difference between latency and throughput.",
    "What does an API gateway do?",
    "hello",
]


class ChatUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(3)
    def repeated_question(self):
        self.client.post(
            "/chat",
            json={"message": random.choice(REPEATED_QUESTIONS)},
            name="/chat [repeat]",
        )

    @task(1)
    def unique_question(self):
        # A unique suffix forces a cache miss, so this measures the model path.
        self.client.post(
            "/chat",
            json={"message": f"Give one tip about API reliability (#{uuid.uuid4().hex[:8]})."},
            name="/chat [unique]",
        )

    @task(1)
    def metrics(self):
        self.client.get("/metrics")
