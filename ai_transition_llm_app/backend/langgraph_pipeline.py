"""
LangGraph pipeline for AI Transition LLM App

This file builds a 4-agent orchestrated workflow:

    1. Project Agent        (understanding + milestone context)
    2. Risk Agent           (risks, severity, mitigation)
    3. Communication Agent  (stakeholder gaps, communication inefficiencies)
    4. Supervisor Agent     (final combined summary)

This pipeline is called by:
    backend/mcp_server/tools.py   → workflow_tool()

The workflow returns a structured dictionary that the front-end
can display in separate cards.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from typing import Dict, Any

from backend.config import DEFAULT_AGENT_MODEL, load_all_synthetic_data
from backend.llm_client import call_llm
from backend.agents.project_agent import run_project_agent
from backend.agents.risk_agent import run_risk_agent
from backend.agents.comms_agent import run_comms_agent
from backend.agents.supervisor_agent import run_supervisor_agent


# ============================================================
# SHARED STATE FOR THE GRAPH
# ============================================================

class WorkflowState(dict):
    """
    LangGraph uses simple dict-like states.
    Each agent reads from and writes to the state.
    """

    project_input: str     # main question from user
    model: str             # LLM model ID chosen by user
    project_agent_output: str
    risk_agent_output: str
    comms_agent_output: str
    supervisor_output: str


# ============================================================
# CREATE SYNTHETIC DATA CONTEXT
# ============================================================

SYN_DATA = load_all_synthetic_data()

# Contents inside SYN_DATA:
#   {
#       "project_data.json": {...},
#       "risk_logs.json": {...},
#       "comms_logs.json": {...},
#       "transition_examples.json": {...}
#   }
#
# Agents will receive this as additional context.


# ============================================================
# NODE DEFINITIONS
# ============================================================

def project_node(state: WorkflowState) -> WorkflowState:
    """Runs Project Agent."""
    output = run_project_agent(
        question=state["project_input"],
        model=state["model"],
        synthetic_data=SYN_DATA
    )
    state["project_agent_output"] = output
    return state


def risk_node(state: WorkflowState) -> WorkflowState:
    """Runs Risk Agent."""
    output = run_risk_agent(
        question=state["project_input"],
        project_agent_summary=state["project_agent_output"],
        model=state["model"],
        synthetic_data=SYN_DATA
    )
    state["risk_agent_output"] = output
    return state


def comms_node(state: WorkflowState) -> WorkflowState:
    """Runs Communication Agent."""
    output = run_comms_agent(
        question=state["project_input"],
        project_agent_summary=state["project_agent_output"],
        model=state["model"],
        synthetic_data=SYN_DATA
    )
    state["comms_agent_output"] = output
    return state


def supervisor_node(state: WorkflowState) -> WorkflowState:
    """Runs Supervisor Agent (final summary)."""
    output = run_supervisor_agent(
        project_summary=state["project_agent_output"],
        risk_summary=state["risk_agent_output"],
        comms_summary=state["comms_agent_output"],
        model=state["model"],
        synthetic_data=SYN_DATA
    )
    state["supervisor_output"] = output
    return state


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_workflow_graph():
    """
    LangGraph pipeline:
        Project → Risk → Comms → Supervisor → END
    """

    graph = StateGraph(WorkflowState)

    # Register nodes
    graph.add_node("project", project_node)
    graph.add_node("risk", risk_node)
    graph.add_node("comms", comms_node)
    graph.add_node("supervisor", supervisor_node)

    # Edges
    graph.set_entry_point("project")
    graph.add_edge("project", "risk")
    graph.add_edge("risk", "comms")
    graph.add_edge("comms", "supervisor")
    graph.add_edge("supervisor", END)

    return graph.compile()


# ============================================================
# PUBLIC FUNCTION USED BY MCP TOOL
# ============================================================

def run_full_workflow(user_question: str, model: str = DEFAULT_AGENT_MODEL) -> Dict[str, Any]:
    """
    Runs the complete 4-agent LangGraph workflow.

    Called by:
        backend/mcp_server/tools.py  → workflow_tool

    Returns:
        {
            "project_agent": ...,
            "risk_agent": ...,
            "comms_agent": ...,
            "supervisor": ...
        }
    """

    workflow = build_workflow_graph()

    initial_state = WorkflowState(
        project_input=user_question,
        model=model,
    )

    final_state = workflow.invoke(initial_state)

    return {
        "project_agent": final_state.get("project_agent_output", ""),
        "risk_agent": final_state.get("risk_agent_output", ""),
        "comms_agent": final_state.get("comms_agent_output", ""),
        "supervisor": final_state.get("supervisor_output", ""),
    }
