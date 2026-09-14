"""The service layer in front of vLLM: our own FastAPI app.

Why this exists when vLLM already ships a FastAPI server. That server is
an *engine* API -- it speaks OpenAI's schema and knows nothing about this
project. Everything that makes a request meaningful here (which config is
being served, that a reply is a persona reply, what counts as an error
worth alerting on) lives above it. This module is that layer, and it is
what the README's "FastAPI" claim refers to; until now the repo named
FastAPI in its stack and imported it nowhere.

It is a gateway, not a second inference path (CLAUDE.md single-pipeline
rule). `/chat` calls `src.serve.pipeline.generate()` -- the same function
the eval scripts and the CLI use. `/chat/stream` is the one exception,
and the same one `scripts/throughput_sweep.py` already documents: token
streaming cannot come from a function that returns a finished string, so
it opens its own streaming client against the same upstream server with
the same parameters.

Config is chosen per request by name (`{"config": "sft"}`), resolved once
at startup from config/experiments/*.yaml. Resolving *which config to
load* is not branching on config identity -- no behaviour differs by
name, exactly as `main.py --config` already works. Names not on that list
are rejected, so a request cannot name an arbitrary path.

Run it:
    uv run uvicorn src.serve.api:app --port 8080
(with vLLM reachable at config's serving.base_url -- in the cluster that
is `kubectl port-forward svc/serana-vllm 8000:8000`)
"""

from __future__ import annotations

import glob
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.config import load_config
from src.serve.pipeline import build_system_prompt, generate, resolve_base_url

# --- config registry -------------------------------------------------------

CONFIGS: dict[str, dict[str, Any]] = {}
for _path in sorted(glob.glob("config/experiments/*.yaml")):
    _cfg = load_config(_path)
    CONFIGS[_cfg["name"]] = _cfg

# --- metrics ---------------------------------------------------------------

# Labelled by config so the three served variants can be compared on the
# same axes the results tables use. `status` is ok|error rather than an HTTP
# code: the thing worth alerting on is "the persona endpoint failed", and
# the upstream counter below records why.
REQUESTS = Counter("serana_requests_total", "Requests handled", ["config", "endpoint", "status"])
# Buckets chosen from P5's measured numbers (TTFT p50 0.091-0.247 s, full
# replies a few seconds), not from a library default -- a histogram whose
# buckets do not straddle the real distribution measures nothing.
LATENCY = Histogram(
    "serana_request_latency_seconds",
    "End-to-end latency of a persona reply",
    ["config", "endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
)
COMPLETION_TOKENS = Counter(
    "serana_completion_tokens_total", "Completion tokens returned", ["config"]
)
UPSTREAM_ERRORS = Counter(
    "serana_upstream_errors_total", "Failures reaching or reading from vLLM", ["config", "reason"]
)

app = FastAPI(title="serana serving gateway")


class ChatRequest(BaseModel):
    turn: str = Field(..., description="the user's message")
    config: str = Field("sft", description="one of config/experiments/*.yaml's names")
    history: list[dict[str, str]] | None = Field(
        None, description="prior turns, for the multi-turn attack probes"
    )
    max_tokens: int | None = None


def _resolve(name: str) -> dict[str, Any]:
    if name not in CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown config {name!r}; available: {sorted(CONFIGS)}",
        )
    return CONFIGS[name]


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    config = _resolve(req.config)
    start = time.perf_counter()
    try:
        result = generate(config, req.turn, history=req.history, max_tokens=req.max_tokens)
    except Exception as exc:  # upstream down, model not loaded, timeout
        UPSTREAM_ERRORS.labels(req.config, type(exc).__name__).inc()
        REQUESTS.labels(req.config, "chat", "error").inc()
        LATENCY.labels(req.config, "chat").observe(time.perf_counter() - start)
        raise HTTPException(status_code=502, detail=f"upstream vLLM failed: {exc}") from exc
    LATENCY.labels(req.config, "chat").observe(time.perf_counter() - start)
    REQUESTS.labels(req.config, "chat", "ok").inc()
    if result.get("completion_tokens"):
        COMPLETION_TOKENS.labels(req.config).inc(result["completion_tokens"])
    return result


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Server-sent events, one `data:` line per token chunk, terminated by
    `data: [DONE]`. The client sees the first token as soon as vLLM emits
    it, which is the whole point of streaming a persona reply."""
    config = _resolve(req.config)
    from openai import OpenAI

    from src.serve.pipeline import resolve_model_name

    client = OpenAI(base_url=resolve_base_url(config), api_key="not-needed")
    messages = [{"role": "system", "content": build_system_prompt()}]
    messages.extend(req.history or [])
    messages.append({"role": "user", "content": req.turn})

    def event_stream():
        start = time.perf_counter()
        sent = 0
        try:
            stream = client.chat.completions.create(
                model=resolve_model_name(config),
                messages=messages,
                temperature=config.get("generation", {}).get("temperature", 0.0),
                max_tokens=req.max_tokens or config.get("generation", {}).get("max_tokens", 512),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    sent += 1
                    yield f"data: {delta}\n\n"
            REQUESTS.labels(req.config, "chat_stream", "ok").inc()
        except Exception as exc:
            UPSTREAM_ERRORS.labels(req.config, type(exc).__name__).inc()
            REQUESTS.labels(req.config, "chat_stream", "error").inc()
            # The response has already started, so the error cannot become a
            # 502 -- it has to travel in-band or the client sees a silent truncation.
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            LATENCY.labels(req.config, "chat_stream").observe(time.perf_counter() - start)
            COMPLETION_TOKENS.labels(req.config).inc(sent)
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Liveness for this gateway, plus whether its upstream answers. Kept
    separate from vLLM's own /health: a green gateway in front of a dead
    engine is exactly the failure this is meant to surface."""
    base_url = resolve_base_url(CONFIGS[next(iter(CONFIGS))]).rstrip("/")
    upstream = base_url.removesuffix("/v1") + "/health"
    try:
        resp = httpx.get(upstream, timeout=2.0)
        upstream_ok = resp.status_code == 200
    except Exception:
        upstream_ok = False
    return {"ok": True, "upstream_ok": upstream_ok, "configs": sorted(CONFIGS)}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """This gateway's own metrics. vLLM exposes engine-level ones (queue
    depth, KV-cache utilization, TTFT) on its own /metrics -- both are
    scraped, and the dashboard reads them together."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
