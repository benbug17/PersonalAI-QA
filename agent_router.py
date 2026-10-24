"""FastAPI router exposing the agent and the latency metrics.

The agent stack (LangGraph, Chroma) is imported lazily inside the handlers, so the rest of the API
keeps working, and keeps its small dependency list, when requirements-agent.txt is not installed.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException
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


def make_router(auth_dependency: Callable, graph_factory: Optional[Callable] = None) -> APIRouter:
    router = APIRouter()

    @router.post("/agent", response_model=AgentResponse)
    def agent_endpoint(request: AgentRequest, user=Depends(auth_dependency)):
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
    def metrics(user=Depends(auth_dependency)):
        return tracker.summary()

    return router
