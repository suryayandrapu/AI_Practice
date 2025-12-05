"""
MCP Router

This exposes a single FastAPI endpoint:

    POST /mcp/invoke

Frontend or internal backend services call this endpoint with:
{
    "tool": "workflow" | "chat" | "compare" | "judge",
    "model": "<model_id>",
    "input": "<user input>",
    "extra": {...}            # Optional extra parameters
}

The router:
1. Validates the tool name
2. Looks up the function in TOOL_REGISTRY
3. Passes the payload to the tool
4. Returns the tool output as JSON
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from backend.mcp_server.tools import TOOL_REGISTRY


# ============================================================
# Request Schema
# ============================================================

class MCPInvokeRequest(BaseModel):
    tool: str
    model: Optional[str] = None
    input: Optional[Any] = None
    extra: Optional[Dict[str, Any]] = None


# ============================================================
# MCP Router
# ============================================================

router = APIRouter(prefix="/mcp", tags=["MCP"])


@router.post("/invoke")
async def invoke_mcp(req: MCPInvokeRequest):
    """
    Main MCP entrypoint.
    """

    tool_name = req.tool

    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown MCP tool '{tool_name}'. "
                   f"Available tools: {list(TOOL_REGISTRY.keys())}"
        )

    tool_fn = TOOL_REGISTRY[tool_name]

    # Prepare payload
    payload = {
        "tool": tool_name,
        "model": req.model,
        "input": req.input,
        "extra": req.extra or {}
    }

    try:
        result = tool_fn(payload)

        # Tool functions may return dicts, strings, or objects.
        # Ensure we always return a clean JSON-friendly dict.
        if isinstance(result, dict):
            return result
        else:
            return {"result": result}

    except Exception as e:
        # Surface errors cleanly
        raise HTTPException(
            status_code=500,
            detail=f"MCP tool '{tool_name}' execution failed: {e}"
        )
