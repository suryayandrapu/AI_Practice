from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Import MCP router & tools from backend
from backend.mcp_server.router import router as mcp_router
from backend.mcp_server.tools import TOOL_REGISTRY
from backend.config import DEFAULT_CHAT_MODEL


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(title="AI Transition LLM App (MCP + LangGraph)")


# ------------------------------------------------------------
# CORS (allow frontend HTML/JS to call the API)
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # For hackathon/demo, allow all. Tighten later if needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Include MCP router
#   - Exposes: POST /mcp/invoke
#   - Tools implemented in backend.mcp_server.tools
# ------------------------------------------------------------
app.include_router(mcp_router)


# ------------------------------------------------------------
# Serve frontend HTML (optional but convenient)
#   - GET /  -> returns frontend/frontend.html
# ------------------------------------------------------------
FRONTEND_PATH = Path(__file__).parent / "frontend" / "frontend.html"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend() -> HTMLResponse:
    """
    Serve the main dashboard UI.

    You can also open frontend/frontend.html directly in the browser if you prefer,
    but serving it here simplifies the setup:
      http://127.0.0.1:8000/
    """
    if not FRONTEND_PATH.exists():
        # Fallback minimal page if frontend isn't present yet
        return HTMLResponse(
            "<h1>AI Transition LLM App</h1><p>frontend/frontend.html not found.</p>",
            status_code=200,
        )

    html = FRONTEND_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html)


# ------------------------------------------------------------
# Chatbot endpoint used by the floating chat UI
#
# Frontend JS (from your frontend.html) can POST like:
#
#   fetch("/chatbot", {
#     method: "POST",
#     headers: { "Content-Type": "application/x-www-form-urlencoded" },
#     body: "message=" + encodeURIComponent(text) + "&llm1=" + encodeURIComponent(llm1Model)
#   })
#
# This endpoint delegates to the MCP "chat" tool, which will:
#   - Maintain chat history (in tools.py)
#   - Generate follow-up question suggestions
#   - Return JSON: { "answer": str, "suggestions": [str, ...] }
# ------------------------------------------------------------
@app.post("/chatbot")
async def chatbot(
    message: str = Form(...),
    llm1: Optional[str] = Form(None),
):
    """
    Interactive chatbot endpoint.

    - Uses the MCP `chat` tool under the hood.
    - `message` is the user's new question.
    - `llm1` (optional) is the selected LLM model from the UI; if not provided,
      falls back to DEFAULT_CHAT_MODEL from backend.config.
    """
    tool_name = "chat"  # must match key in TOOL_REGISTRY in backend.mcp_server.tools

    if tool_name not in TOOL_REGISTRY:
        # Safety check in case tools.py is not wired yet
        return {
            "answer": "Chat tool is not configured on the server.",
            "suggestions": [],
        }

    model_to_use = llm1 or DEFAULT_CHAT_MODEL

    payload = {
        "tool": tool_name,
        "model": model_to_use,
        "input": message,
        # You can add more fields if needed later, e.g. project id, user id, etc.
    }

    chat_fn = TOOL_REGISTRY[tool_name]
    result = chat_fn(payload)

    # Expecting result like: {"answer": "...", "suggestions": [...]}
    # If your MCP tool returns a different shape, adjust this mapping.
    answer = result.get("answer") or result.get("output") or "[No response]"
    suggestions = result.get("suggestions") or []

    return {
        "answer": answer,
        "suggestions": suggestions,
    }


# ------------------------------------------------------------
# Simple health check
# ------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "backend": "fastapi", "mcp": True}


# ============================================================
# To run locally:
#   uvicorn app:app --reload --port 8000
#
# Then open:
#   Frontend (served by FastAPI):   http://127.0.0.1:8000/
#   MCP endpoint (JSON):           http://127.0.0.1:8000/mcp/invoke
# ============================================================
# FastAPI entrypoint (runs MCP + serves frontend if needed)
