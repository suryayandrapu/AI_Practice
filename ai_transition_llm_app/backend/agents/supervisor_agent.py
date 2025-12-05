"""
Supervisor Agent

Purpose:
---------
Acts like a Transition Program Director synthesizing outputs from:
- Project Agent
- Risk Agent
- Communication Agent

Produces:
- A clear, executive-level summary
- Top risks to highlight to leadership
- Required stakeholder actions
- A consolidated transition viewpoint for decision making

Inputs:
--------
- project_summary: output from project_agent.py
- risk_summary: output from risk_agent.py
- comms_summary: output from comms_agent.py
- synthetic_data: optional additional context

Output:
--------
Structured final summary for:
- Dashboard summary card
- Leadership reporting
- End of LangGraph pipeline
"""

from __future__ import annotations
from typing import Dict, Any

from backend.llm_client import call_llm
from backend.config import DEFAULT_AGENT_MODEL


# ============================================================
# BUILD SUPERVISOR PROMPT
# ============================================================

def build_supervisor_prompt(
    project_summary: str,
    risk_summary: str,
    comms_summary: str,
    synthetic_data: Dict[str, Any]
) -> str:

    project_data = synthetic_data.get("project_data.json", {})
    transition_examples = synthetic_data.get("transition_examples.json", {})

    return f"""
You are the SUPERVISOR AGENT in an IT Transition Program.

Your job is to consolidate the outputs of multiple agents and provide
a final, leadership-ready summary.

---------------------------------------
PROJECT AGENT SUMMARY:
{project_summary}

---------------------------------------
RISK AGENT SUMMARY:
{risk_summary}

---------------------------------------
COMMUNICATION AGENT SUMMARY:
{comms_summary}

---------------------------------------
SYNTHETIC PROJECT METADATA:
{project_data}

---------------------------------------
TRANSITION EXAMPLES (for reasoning patterns):
{transition_examples}

---------------------------------------
EXPECTED OUTPUT STRUCTURE (STRICT):
1. Executive Transition Summary
2. Top 5 Risks Leadership Should Be Aware Of
3. Critical Dependencies & Impact Assessment
4. Stakeholder Alignment Summary
5. Metric Recommendations (Weekly KPIs)
6. Required Customer Actions (Clear, Actionable)
7. Required Internal Actions (Clear, Actionable)
8. Readiness Score (0–100) with justification
9. 7-Day Outlook (What will matter next week)

Tone:
- Concise
- Executive-level
- Insightful
- Data-informed
- No repetition of raw agent outputs
"""


# ============================================================
# RUN SUPERVISOR AGENT
# ============================================================

def run_supervisor_agent(
    project_summary: str,
    risk_summary: str,
    comms_summary: str,
    model: str = DEFAULT_AGENT_MODEL,
    synthetic_data: Dict[str, Any] = None
) -> str:
    """
    Execute the Supervisor Agent.

    Produces final structured summary used in UI and workflow results.
    """

    if synthetic_data is None:
        synthetic_data = {}

    prompt = build_supervisor_prompt(
        project_summary=project_summary,
        risk_summary=risk_summary,
        comms_summary=comms_summary,
        synthetic_data=synthetic_data,
    )

    response = call_llm(
        model=model,
        prompt=prompt,
        temperature=0.15,
        system_prompt=(
            "You are the Program Director for a large-scale IT Transition. "
            "Your job is to synthesize signals from multiple teams and provide "
            "clear guidance to leadership."
        ),
    )

    return response
