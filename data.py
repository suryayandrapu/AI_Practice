# agents/data_ingest_agent.py
from typing import Any, Dict
from base import BaseAgent
from llm import chat_llm

class DataIngestAgent(BaseAgent):
    name = "data_ingest_agent"
    description = "Summarize destination data (attractions, hours, fees, coords, transit)."

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        data = task.get("destination_data", {})
        prompt = f"""
You are a destination data summarizer. Given structured/partial data, produce a compact catalog:
- Attractions (name, category, opening hours, entry fees, coords)
- Transit options (public/cabs/walking times)
- Seasonal notes/weather constraints
- Booking requirements and typical crowds

Destination:
{data}

Output Markdown and a JSON block named 'catalog' with fields described above.
"""
        content = chat_llm([{"role": "user", "content": prompt}])
        return {"catalog_artifact": content}