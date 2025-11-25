# agents/synthesizer_agent.py
from typing import Any, Dict
from base import BaseAgent
from llm import chat_llm

class SynthesizerAgent(BaseAgent):
    name = "synthesizer_agent"
    description = "Merge itinerary, alternates, and export blocks (iCal, CSV)."

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
Synthesize a final deliverable:
- Executive summary
- Markdown itinerary
- Alternates section
- iCal (VCALENDAR) snippet with events
- CSV with columns: day, start_time, end_time, title, location, transit, fee
Ensure consistent times within the trip window.

Inputs:
Constraints:
{task.get("constraints_artifact","")}
Itinerary:
{task.get("itinerary_artifact","")}
Alternates:
{task.get("alternates_artifact","")}

Output a single Markdown document including code blocks for iCal and CSV.
"""
        content = chat_llm([{"role": "user", "content": prompt}])
        return {"final_markdown": content}