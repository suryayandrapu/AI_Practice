"""
LLM Client helper for all agents + MCP tools + LangGraph workflow.

This module provides:
- call_llm(): Simple wrapper to call TCS GenAI Lab models using LangChain ChatOpenAI
- create_llm(): Reusable LLM object
- global httpx client with verify=False (required for internal GenAI Lab endpoint)

The rest of the backend only calls call_llm() for consistency.
"""

from __future__ import annotations

import traceback
from typing import Optional

import httpx
from langchain_openai import ChatOpenAI

from backend.config import BASE_URL, GENAI_API_KEY


# ============================================================
# Shared HTTPX Client
# ============================================================

# Using verify=False because GenAI Lab internal CA is not recognized externally.
http_client = httpx.Client(
    verify=False,
    timeout=60.0  # can be increased depending on model latency
)


# ============================================================
# LLM Factory
# ============================================================

def create_llm(model: str, temperature: float = 0.2) -> ChatOpenAI:
    """
    Create a LangChain ChatOpenAI LLM object for the given model.

    Args:
        model: full model string, e.g. "azure/genailab-maas-gpt-4o"
        temperature: default 0.2 for predictable behavior

    Returns:
        ChatOpenAI object
    """
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=GENAI_API_KEY,
        model=model,
        temperature=temperature,
        http_client=http_client,
    )


# ============================================================
# Unified Call Wrapper
# ============================================================

def call_llm(
    model: str,
    prompt: str,
    temperature: float = 0.2,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Call any TCS GenAI Lab LLM with a standardized prompt.

    Agents and MCP tools should ONLY use this function.

    Args:
        model: selected model ID (from config.ALLOWED_MODELS mapping)
        prompt: user or agent-generated prompt text
        temperature: creativity level
        system_prompt: optional system instruction

    Returns:
        Model's text output
    """

    try:
        llm = create_llm(model=model, temperature=temperature)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = llm.invoke(messages)

        if hasattr(response, "content"):
            return response.content

        # Fallback for safety
        return str(response)

    except Exception as e:
        traceback.print_exc()
        return f"[LLM ERROR] {e}"


# ============================================================
# Streaming Version (optional future use)
# ============================================================

async def call_llm_stream(model: str, prompt: str, temperature: float = 0.2):
    """
    Asynchronous streaming version (optional).
    Not used yet, but MCP chat tool may choose to use streaming later.

    Yields incremental chunks.
    """
    try:
        llm = create_llm(model=model, temperature=temperature)

        messages = [{"role": "user", "content": prompt}]

        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    except Exception as e:
        yield f"[STREAM ERROR] {e}"
# ChatOpenAI + embeddings + httpx client
