# core/llm.py
from typing import List, Dict
import ollama
from config import LLM_MODEL, TEMPERATURE

def chat_llm(messages: List[Dict[str, str]]) -> str:
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": TEMPERATURE}
    )
    return resp["message"]["content"]