"""
Communication Agent

Role:
-----
Analyze communication patterns, stakeholder alignment, escalation paths,
dependency visibility, cadence effectiveness, and identify communication-related risks.

Inputs:
-------
- question: original query from the user
- project_agent_summary: output from Project Agent
- model: chosen LLM model
- synthetic_data: preloaded JSON files { project_data, comms_logs, ... }

Output:
--------
Clean, structured text suitable for UI display + LangGraph supervisor agent.
"""

from __future__ import annotations
from typing import Dict, Any

from backend.llm_client import call_llm
from backend.config import DEFAULT_AGENT_MODEL


def build_comms_prompt(
    question: str,
    project_agent_summary: str,
    synthetic_data: Dict[str, Any]
) -> str:
    """
    Builds the communication analysis prompt.
    Includes synthetic comms logs from backend/synthetic_data.
    """

    comms_logs = synthetic_data.get("comms_logs.json", {})
    project_data = synthetic_data.get("project_data.json", {})

    return f"""
You are the Communication Analysis Agent in an IT Transition & Risk Tracking system.

Your responsibilities:
- Identify stakeholder misalignment.
- Detect communication gaps that may cause delays or misinterpretations.
- Review cadence & escalation maturity.
- Evaluate KT knowledge flow and documentation quality.
- Measure how well teams collaborate across onshore/offshore.
- Highlight areas where poor communication increases risk.

Use the context below:

-------------------------------------
USER QUESTION:
{question}

-------------------------------------
PROJECT AGENT SUMMARY:
{project_agent_summary}

-------------------------------------
SYNTHETIC PROJECT METADATA:
{project_data}

-------------------------------------
COMMUNICATION LOGS (SYNTHETIC):
{comms_logs}

-------------------------------------
EXPECTED OUTPUT STRUCTURE:
Provide a clear structured analysis:

1. Stakeholder Alignment Issues
2. Weak Communication Channels
3. Cadence & Escalation Quality
4. Documentation / KT Gaps
5. Collaboration Score (with reasoning)
6. Communication-Based Risks
7. Recommendations & Fixes (actionable)

Ensure clarity, avoid generic statements, and reference the synthetic data where useful.
"""


def run_comms_agent(
    question: str,
    project_agent_summary: str,
    model: str = DEFAULT_AGENT_MODEL,
    synthetic_data: Dict[str, Any] = None
) -> str:
    """
    Executes the Communication Agent.

    Returns structured text output.
    """

    if synthetic_data is None:
        synthetic_data = {}

    prompt = build_comms_prompt(
        question=question,
        project_agent_summary=project_agent_summary,
        synthetic_data=synthetic_data,
    )

    response = call_llm(
        model=model,
        prompt=prompt,
        temperature=0.2,
        system_prompt="You are an expert Communication Analyst for IT Transition Programs.",
    )

    return response
