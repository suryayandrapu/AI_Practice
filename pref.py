# agents/preferences_agent.py
from typing import Any, Dict
from base import BaseAgent
from llm import chat_llm

class PreferencesAgent(BaseAgent):
    name = "preferences_agent"
    description = "Normalize and expand user preferences into actionable constraints."

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        prefs = task.get("preferences", {})
        prompt = f"""
You are a travel preference normalizer. Convert raw preferences into actionable constraints:
- Activity types (e.g., museums, food, nature)
- Pace (relaxed/medium/fast)
- Budget per day (currency retained)
- Time windows, meal preferences, accessibility needs
- Hard constraints (must-see items), soft constraints (nice-to-have)

Raw preferences:
{prefs}
Output JSON with keys: activity_types, pace, budget_per_day, must_see, soft_constraints, accessibility, time_windows, meal_prefs.
"""
        content = chat_llm([{"role": "user", "content": prompt}])
        return {"preferences_artifact": content}