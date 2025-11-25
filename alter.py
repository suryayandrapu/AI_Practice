# agents/alternatives_agent.py
from typing import Any, Dict
from base import BaseAgent
from llm import chat_llm

class AlternativesAgent(BaseAgent):
    name = "alternatives_agent"
    description = "Suggest alternates per slot for flexibility (e.g., weather changes)."

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
For each itinerary event, suggest 2 alternates (indoor/outdoor or cost-friendly/premium).
Consider weather, crowding, and same-day transit feasibility.

Itinerary JSON/Markdown:
{task.get("itinerary_artifact","")}

Return Markdown section 'Alternates' and JSON 'alternates' mapping event IDs to two alternatives.
"""
        content = chat_llm([{"role": "user", "content": prompt}])
        return {"alternates_artifact": content}