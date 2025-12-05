"""
Risk Agent

Purpose:
---------
Analyze project, KT, communication, and execution risks in an IT Transition Program.
Uses:
- User question
- Project Agent summary
- Synthetic risk-related data

Outputs structured risk assessment for Supervisor Agent & UI display.

Risk categories considered:
- Transition milestones
- Knowledge transfer readiness
- Engineering / environment blockers
- Stakeholder gaps / ownership issues
- Timeline slip indicators
- Resource bandwidth / SME availability
- Documentation gaps
"""

from __future__ import annotations
from typing import Dict, Any

from backend.llm_client import call_llm
from backend.config import DEFAULT_AGENT_MODEL


# ============================================================
# BUILD RISK PROMPT
# ============================================================

def build_risk_prompt(
    question: str,
    project_agent_summary: str,
    synthetic_data: Dict[str, Any]
) -> str:
    """
    Construct the risk analysis prompt including synthetic risk metadata.
    """

    risk_logs = synthetic_data.get("risk_logs.json", {})
    project_data = synthetic_data.get("project_data.json", {})
    transition_examples = synthetic_data.get("transition_examples.json", {})

    return f"""
You are the RISK ANALYST AGENT for an IT Transition & KT Program.

Your responsibilities:
- Identify, classify, and assess risks that affect timeline, KT effectiveness, documentation, environment setup, and delivery.
- Use synthetic risk logs for realistic patterns.
- Use project metadata and the Project Agent's understanding as context.
- Provide mitigation steps that are specific and actionable.

---------------------------------------
USER QUESTION:
{question}

---------------------------------------
PROJECT AGENT SUMMARY:
{project_agent_summary}

---------------------------------------
SYNTHETIC PROJECT METADATA:
{project_data}

---------------------------------------
SYNTHETIC RISK LOGS:
{risk_logs}

---------------------------------------
TRANSITION EXAMPLES (for context and patterns):
{transition_examples}

---------------------------------------
EXPECTED OUTPUT STRUCTURE (STRICT):
1. Key Transition Risks  
2. Severity & Likelihood Assessment  
3. Root Cause Analysis  
4. Dependencies & Blockers  
5. Mitigation Recommendations (Specific, Actionable)  
6. Risk Heat-Map Categorization (Critical / High / Medium / Low)  
7. Early Warning Indicators  
8. Required Stakeholder Actions  

Ensure clarity, avoid generic answers, and reference synthetic data where appropriate.
"""


# ============================================================
# RUN AGENT
# ============================================================

def run_risk_agent(
    question: str,
    project_agent_summary: str,
    model: str = DEFAULT_AGENT_MODEL,
    synthetic_data: Dict[str, Any] = None
) -> str:
    """
    Executes the Risk Agent.

    Returns structured risk assessment text.
    """

    if synthetic_data is None:
        synthetic_data = {}

    prompt = build_risk_prompt(
        question=question,
        project_agent_summary=project_agent_summary,
        synthetic_data=synthetic_data,
    )

    response = call_llm(
        model=model,
        prompt=prompt,
        temperature=0.15,
        system_prompt=(
            "You are an expert IT Transition & Program Risk Manager. "
            "Be precise, structured, and directly linked to transition execution."
        ),
    )

    return response
# Risk agent logic
