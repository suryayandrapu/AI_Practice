# Project agent logic
"""
Project Agent

Purpose:
--------
To interpret the user's query and provide a detailed project-level analysis
focusing on transition status, milestones, scope, dependencies, backlog items,
team readiness, KT progress, and execution feasibility.

Inputs:
--------
- question: user question
- model: chosen LLM model (UI dropdown)
- synthetic_data: dict loaded from backend/synthetic_data/

Outputs:
---------
A structured summary suitable for:
- Risk Agent
- Communication Agent
- Supervisor Agent
- UI Dashboard Cards
"""

from __future__ import annotations
from typing import Dict, Any

from backend.llm_client import call_llm
from backend.config import DEFAULT_AGENT_MODEL


# ============================================================
# BUILD PROJECT PROMPT
# ============================================================

def build_project_prompt(question: str, synthetic_data: Dict[str, Any]) -> str:
    """
    Build prompt for project-level contextual analysis.
    Takes synthetic data to anchor the reasoning.
    """

    project_data = synthetic_data.get("project_data.json", {})
    transition_examples = synthetic_data.get("transition_examples.json", {})

    return f"""
You are the Project Understanding Agent for an IT Transition Program.

Your job is to:
- Interpret the user's question.
- Provide transition-aligned project understanding using real transition language.
- Identify major milestones, readiness progress, KT state, gaps, ownership, and dependencies.
- Consider synthetic project metadata as part of your reasoning.
- Produce a structured summary usable by Risk, Comms, and Supervisor agents.

---------------------------------------
USER QUESTION:
{question}

---------------------------------------
SYNTHETIC PROJECT METADATA:
{project_data}

---------------------------------------
SYNTHETIC TRANSITION EXAMPLES:
(Examples of transitions, milestone definitions, KT progress, effectiveness measures)
{transition_examples}

---------------------------------------
EXPECTED OUTPUT STRUCTURE (STRICT):
1. High-Level Understanding of the Ask
2. Relevant Transition Milestones & Current Status
3. Scope Clarifications & Assumptions
4. KT Progress Summary (Readiness Matrix + Risks)
5. Dependencies (Teams, Systems, SMEs, Environments)
6. Open Items / Backlog Tasks
7. Early Observed Risks (Avoid duplicating risk agent)
8. Recommended Next Steps (Actionable)

Make your response structured, crisp, and aligned with IT Transition best practices.
"""


# ============================================================
# RUN AGENT
# ============================================================

def run_project_agent(
    question: str,
    model: str = DEFAULT_AGENT_MODEL,
    synthetic_data: Dict[str, Any] = None
) -> str:
    """
    Execute the Project Agent.

    Returns structured text suitable for downstream agents.
    """

    if synthetic_data is None:
        synthetic_data = {}

    prompt = build_project_prompt(
        question=question,
        synthetic_data=synthetic_data,
    )

    response = call_llm(
        model=model,
        prompt=prompt,
        temperature=0.1,
        system_prompt="You are an IT Transition Project Lead with deep expertise in migrations, KT, and hypercare."
    )

    return response
