# agents/constraints_agent.py
from typing import Any, Dict
from base import BaseAgent
from llm import chat_llm

class ConstraintsAgent(BaseAgent):
    name = "constraints_agent"
    description = "Consolidate hard constraints from preferences, availability, and timing."

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        trip = {
            "destination": task.get("destination"),
            "start_date": task.get("start_date"),
            "end_date": task.get("end_date"),
        }
        prompt = f"""
Unify constraints for itinerary planning:
- Trip window: {trip}
- Preferences summary (JSON/text allowed)
- Destination catalog summary (JSON/text allowed)

Goal: Produce hard constraints (opening hours, budget caps, must-see items, transit limitations) and soft constraints.

Inputs:
Preferences:
{task.get("preferences_artifact","")}
Catalog:
{task.get("catalog_artifact","")}

Return a JSON 'constraints' object with keys: calendar_days, daily_time_bands, budget_cap, must_do, avoid, transit_limits.
"""
        content = chat_llm([{"role": "user", "content": prompt}])
        return {"constraints_artifact": content}