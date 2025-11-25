# core/base_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    name: str = "base_agent"
    description: str = "Abstract agent"

    @abstractmethod
    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process the task and return a structured artifact."""
        ...