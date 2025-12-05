"""
MCP Tools

Each tool is a callable that receives:
{
    "tool": "<tool_name>",
    "model": "<model_id>",
    "input": "<user_input>",
    "extra": { ... }        # optional additional arguments
}

And returns JSON-friendly data.

Tools included:
- chat        : Interactive chatbot (history + follow-up suggestions)
- workflow    : Runs LangGraph 4-agent workflow
- compare     : Compare two LLM responses
- judge       : Judge-LRM chooses best between two answers
"""

from __future__ import annotations

import json
from typing import Dict, Any, Optional, List

from backend.llm_client import call_llm
from backend.langgraph_pipeline import run_full_workflow
from backend.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_COMPARE_MODEL,
    DEFAULT_JUDGE_MODEL,
    ALLOWED_MODELS,
)


# ============================================================
# MEMORY STORE FOR THE CHATBOT
# ============================================================

CHAT_HISTORY: List[Dict[str, str]] = []


# ============================================================
# Helper → build follow-up question suggestion prompt
# ============================================================

def generate_followup_questions(model: str, user_msg: str, bot_msg: str) -> List[str]:
    """Generates follow-up questions from the LLM."""
    suggest_prompt = f"""
You are assisting in an IT Transition Chatbot.

Given:
User said: {user_msg}
Assistant answered: {bot_msg}

Suggest 3 meaningful follow-up questions that the user may ask next.
Return ONLY a JSON list of strings: ["q1", "q2", "q3"].
"""

    raw = call_llm(
        model=model,
        prompt=suggest_prompt,
        temperature=0.1,
        system_prompt="You generate follow-up questions only."
    )

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # fallback
    return [
        "What are the top risks I should focus on next?",
        "Which dependencies may delay the transition?",
        "What metrics should I track weekly?"
    ]


# ============================================================
# CHAT TOOL
# ============================================================

def chat_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chatbot tool with server-side memory and suggestions.
    """

    user_message = payload.get("input", "")
    model = payload.get("model") or DEFAULT_CHAT_MODEL

    # Build conversation context
    history_text = ""
    for msg in CHAT_HISTORY:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['text']}\n"

    prompt = f"""
You are an IT Transition & Risk Tracking Chatbot.

Conversation so far:
{history_text}

New user message:
{user_message}

Respond concisely, but with actionable insights.
"""

    # Main bot response
    bot_reply = call_llm(
        model=model,
        prompt=prompt,
        temperature=0.2,
        system_prompt="You are an expert in IT Transition, KT, and Risk Management."
    )

    # Update memory
    CHAT_HISTORY.append({"role": "user", "text": user_message})
    CHAT_HISTORY.append({"role": "assistant", "text": bot_reply})

    # Generate follow-up questions
    suggestions = generate_followup_questions(model, user_message, bot_reply)

    return {
        "answer": bot_reply,
        "suggestions": suggestions,
    }


# ============================================================
# WORKFLOW TOOL (LangGraph Multi-Agent)
# ============================================================

def workflow_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs LangGraph pipeline: Project → Risk → Comms → Supervisor.
    """

    user_input = payload.get("input", "")
    model = payload.get("model") or DEFAULT_CHAT_MODEL

    results = run_full_workflow(user_input, model=model)

    return {
        "project_agent": results["project_agent"],
        "risk_agent": results["risk_agent"],
        "comms_agent": results["comms_agent"],
        "supervisor": results["supervisor"],
    }


# ============================================================
# COMPARE TOOL (Two LLM Responses)
# ============================================================

def compare_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare outputs of two different LLMs on the same input.
    Expected extra payload:
        {
            "model": "<model1>",
            "extra": {
                "model2": "<model2>"
            }
        }
    """

    question = payload.get("input", "")
    model1 = payload.get("model") or DEFAULT_COMPARE_MODEL
    model2 = payload.get("extra", {}).get("model2") or DEFAULT_COMPARE_MODEL

    ans1 = call_llm(model1, prompt=question)
    ans2 = call_llm(model2, prompt=question)

    return {
        "model_1": model1,
        "model_2": model2,
        "answer_1": ans1,
        "answer_2": ans2,
    }


# ============================================================
# JUDGE TOOL (Which LLM Response is Better)
# ============================================================

def judge_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Judge which LLM response is more appropriate.
    Expected payload.extra:
       {
           "answer_1": "...",
           "answer_2": "..."
       }
    """

    model = payload.get("model") or DEFAULT_JUDGE_MODEL
    question = payload.get("input", "")
    extra = payload.get("extra", {})

    ans1 = extra.get("answer_1", "")
    ans2 = extra.get("answer_2", "")

    judge_prompt = f"""
You are a Senior IT Transition Architect.

Question:
{question}

Answer A:
{ans1}

Answer B:
{ans2}

Compare A and B across:
- Relevance to transition
- Risk identification quality
- Clarity & depth
- Actionability
- Alignment with transition best practices

Return:
1. A short comparison
2. A final verdict strictly in format: "Winner: A" or "Winner: B"
"""

    verdict = call_llm(
        model,
        prompt=judge_prompt,
        temperature=0.0,
        system_prompt="You are an expert LLM Judge for IT Transition."
    )

    return {
        "comparison": verdict,
    }


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {
    "chat": chat_tool,
    "workflow": workflow_tool,
    "compare": compare_tool,
    "judge": judge_tool,
}
