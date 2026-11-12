"""FastAPI router exposing the agent and the latency metrics.

The agent stack (LangGraph, Chroma) is imported lazily inside the handlers, so the rest of the API
keeps working, and keeps its small dependency list, when requirements-agent.txt is not installed.

Settings (environment):
  AGENT_RATE_LIMIT, AGENT_RATE_WINDOW_SECONDS   requests per user per window (default 10 per 60 s; 0 disables)
  METRICS_TOKEN                                 enables GET /metrics for callers sending  X-Metrics-Token: <token>
                                                (unset = the endpoint is off)
"""
from __future__ import annotations

import logging
import math
import os
import secrets
import time
from collections import deque
from typing import Callable, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from timing import tracker

logger = logging.getLogger("personalai")
MAX_QUERY_CHARS = 2000


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class AgentResponse(BaseModel):
    answer: str
    tools_used: list[str]
    llm_calls: int
    latency_ms: int
    truncated: bool


class RateLimiter:
    """Sliding-window limiter: at most `limit` accepted requests per `window` seconds for each key.

    In memory and per process (several server workers would each keep their own count). Requests that are
    refused are not recorded, so a client that keeps hammering does not extend its own block.
    """

    MAX_KEYS = 1000   # beyond this, keys with no recent requests are dropped so memory cannot grow without bound

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit, self.window, self._clock = limit, window, clock
        self._hits: dict[str, deque] = {}

    def retry_after(self, key: str) -> Optional[int]:
        """Record a request. None means allowed; otherwise the whole seconds to wait (never less than 1)."""
        if self.limit <= 0:
            return None
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return math.ceil(self.window - (now - hits[0]))   # hits[0] is still inside the window, so this is >= 1
        hits.append(now)
        if len(self._hits) > self.MAX_KEYS:
            self._hits = {k: h for k, h in self._hits.items() if h and now - h[-1] < self.window}
        return None

    def reset(self) -> None:
        self._hits.clear()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def limiter_from_env() -> RateLimiter:
    window = _env_int("AGENT_RATE_WINDOW_SECONDS", 60)
    return RateLimiter(limit=max(0, _env_int("AGENT_RATE_LIMIT", 10)), window=window if window > 0 else 60)


def make_router(auth_dependency: Callable, graph_factory: Optional[Callable] = None,
                limiter: Optional[RateLimiter] = None) -> APIRouter:
    router = APIRouter()
    router.limiter = limiter if limiter is not None else limiter_from_env()   # reachable for tests and operators

    @router.post("/agent", response_model=AgentResponse)
    def agent_endpoint(request: AgentRequest, user=Depends(auth_dependency)):
        wait = router.limiter.retry_after(str(user))
        if wait is not None:
            logger.warning("Agent rate limit hit by user %s", user)
            raise HTTPException(status_code=429, detail="Too many agent requests, slow down",
                                headers={"Retry-After": str(wait)})
        try:
            import agent as agent_module
        except ImportError:
            raise HTTPException(status_code=503, detail="Agent dependencies are not installed") from None
        try:
            graph = (graph_factory or agent_module.get_default_graph)()
        except agent_module.MissingApiKey:
            raise HTTPException(status_code=503, detail="LLM is not configured") from None
        except Exception:
            logger.exception("Agent could not start")
            raise HTTPException(status_code=502, detail="Agent could not start") from None
        try:
            result = agent_module.run_agent(graph, request.query)
        except Exception:
            logger.exception("Agent call failed")
            raise HTTPException(status_code=502, detail="Agent call failed") from None
        if result.error:
            logger.error("Agent gave no answer: %s", result.error)
            raise HTTPException(status_code=502, detail="Agent call failed")
        return AgentResponse(
            answer=result.answer,
            tools_used=result.tools_used,
            llm_calls=result.llm_calls,
            latency_ms=result.latency_ms,
            truncated=result.truncated,
        )

    @router.get("/metrics")
    def metrics(x_metrics_token: Optional[str] = Header(default=None)):
        expected = os.getenv("METRICS_TOKEN")
        if not expected:                       # off unless an operator turns it on
            raise HTTPException(status_code=404, detail="Not Found")
        if not x_metrics_token or not secrets.compare_digest(x_metrics_token.encode(), expected.encode()):
            raise HTTPException(status_code=403, detail="Forbidden")
        return tracker.summary()

    return router
