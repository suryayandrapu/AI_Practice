# agents/itinerary_planner_agent.py
from typing import Any, Dict
from base import BaseAgent
from llm import chat_llm

class ItineraryPlannerAgent(BaseAgent):
    name = "itinerary_planner_agent"
    description = "Create the day-by-day schedule with time slots, locations, transit, and fees."

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
Plan a coherent day-by-day itinerary based on constraints:
- Include time slots, locations, transit modes/times, expected fees
- Respect opening hours and budget cap
- Spread must-see items across the trip window
- Include short breaks/meals; note reservations if needed

Constraints JSON:
{task.get("constraints_artifact","")}

Output:
1) Markdown itinerary per day
2) JSON object 'itinerary' with day -> [events], where each event has: start_time, end_time, title, location, transit, fee, notes.
"""
        content = chat_llm([{"role": "user", "content": prompt}])
        return {"itinerary_artifact": content}