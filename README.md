# LLM Cost Auto-pilot

A FastAPI + LangGraph gateway that puts OpenAI and Anthropic models behind one
`/chat` API, with caching, routing, failover, and guardrails.

## Request path

1. **Security check**: blocks prompt-injection patterns and masks PII.
2. **Cache lookup**: Redis exact match, then ChromaDB semantic match
   (similarity threshold + TTL). A hit returns immediately.
3. **Intent detection** and **routing**: heuristics, then an LLM classifier for
   ambiguous prompts, pick a model tier: `PRIMARY_MODEL` for simple requests,
   `COMPLEX_MODEL` for complex ones.
4. **Model call with failover**: the routed model is called first. If it errors,
   its provider's circuit is open, or it returns an unusable answer, the request
   fails over to `FAILOVER_MODEL`.
5. **Output validation** and **cache write** (failures are not cached).

## Resilience

- **Config-driven models.** `PRIMARY_MODEL`, `COMPLEX_MODEL` and `FAILOVER_MODEL`
  are plain settings. The provider is inferred from the model name (`claude-*` is
  Anthropic, everything else is OpenAI), so changing a model is an env change.
- **Per-provider circuit breaker.** After `BREAKER_FAILURE_THRESHOLD` consecutive
  errors, a provider is skipped for `BREAKER_RESET_SECONDS`, then a single trial
  request decides whether it is healthy again. Refusals and empty answers trigger
  failover but do not count as provider outages. Breaker state is reported in
  `/health`.
- **Admission control.** Beyond `MAX_CONCURRENT_REQUESTS` in-flight `/chat`
  requests, the gateway sheds load with `503` and `Retry-After: 1` instead of
  piling more work onto the providers. Per-client rate limiting (`RATE_LIMIT`,
  default `20/minute`) still applies on top.
- The graph runs in a thread pool, so slow provider calls don't block the event loop.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Ask a question (`{"message": "...", "thread_id": "..."}`) |
| `GET /health` | Cache and provider health, plus circuit breaker states |
| `GET /metrics` | Requests, error rate, average latency, cache hit rate, tokens, fallbacks, rejected and in-flight requests |

## Running

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Settings come from `.env` (see `app/config.py`), or AWS Secrets Manager when
`APP_ENV=production`.

## Tests

```bash
uv run --group dev python -m pytest -q
```

## Load testing

```bash
uvx locust -f loadtest/locustfile.py --host http://localhost:8000 \
    --users 50 --spawn-rate 10 --run-time 2m --headless --csv loadtest/results
```

Reports requests/sec and p50/p95/p99 latency separately for repeated (cache)
and unique (model) questions. Real provider calls cost money, so keep runs short.
