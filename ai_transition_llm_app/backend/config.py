"""
Configuration file for AI Transition LLM App.

Contains:
- Allowed LLM model mapping
- Base URL for TCS GenAI Lab
- API key (read from env variable or hardcoded temporarily for dev)
- Default model selections for chatbot & workflow
- Synthetic data loading helper
"""

import os
import json
from pathlib import Path


# ============================================================
# API BASE URL (TCS GenAI Lab Endpoint)
# ============================================================

BASE_URL = "https://genailab.tcs.in"


# ============================================================
# API KEY HANDLING
# ============================================================

# You can set an environment variable:  export GENAI_KEY="sk-xxxx"
GENAI_API_KEY = os.getenv("GENAI_KEY", "REPLACE_ME_WITH_YOUR_KEY_HERE")

if GENAI_API_KEY == "REPLACE_ME_WITH_YOUR_KEY_HERE":
    print(
        "[CONFIG WARNING] GENAI_API_KEY not set. Please update config.py or "
        "export environment variable GENAI_KEY=your_key"
    )


# ============================================================
# ALLOWED MODELS FOR THE HACKATHON (Mandatory List)
# ============================================================

ALLOWED_MODELS = {
    "GPT-3.5 Turbo": "azure/genailab-maas-gpt-35-turbo",
    "GPT-4o": "azure/genailab-maas-gpt-4o",
    "GPT-4o Mini": "azure/genailab-maas-gpt-4o-mini",
    "DeepSeek R1 (Reasoning)": "azure_ai/genailab-maas-DeepSeek-R1",
    "DeepSeek V3": "azure_ai/genailab-maas-DeepSeek-V3-0324",
    "Llama 3.2 90B Vision": "azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct",
    "Llama 3.3 70B": "azure_ai/genailab-maas-Llama-3.3-70B-Instruct",
    "Llama 4 Maverick 17B": "azure_ai/genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Phi 3.5 Vision": "azure_ai/genailab-maas-Phi-3.5-vision-instruct",
    "Phi 4 Reasoning": "azure_ai/genailab-maas-Phi-4-reasoning",
}


# ============================================================
# DEFAULT MODEL SELECTIONS
# ============================================================

DEFAULT_CHAT_MODEL = ALLOWED_MODELS["DeepSeek V3"]
DEFAULT_COMPARE_MODEL = ALLOWED_MODELS["GPT-4o"]
DEFAULT_JUDGE_MODEL = ALLOWED_MODELS["Phi 4 Reasoning"]

# Agents in LangGraph will use this unless overridden
DEFAULT_AGENT_MODEL = ALLOWED_MODELS["DeepSeek V3"]


# ============================================================
# SYNTHETIC DATA FOLDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SYNTHETIC_DATA_DIR = BASE_DIR / "synthetic_data"


def load_json(filename: str):
    """
    Utility to load JSON files from backend/synthetic_data folder.
    Agents and MCP tools will use this to read structured context data.
    """
    filepath = SYNTHETIC_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Synthetic data file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# Preload important sets if needed
def load_all_synthetic_data():
    """
    Loads all synthetic data files at once.
    Return as dictionary. Helps when LangGraph workflow needs global context.
    """
    data_cache = {}

    for file in ["project_data.json", "risk_logs.json", "comms_logs.json", "transition_examples.json"]:
        try:
            data_cache[file] = load_json(file)
        except Exception as e:
            print(f"[Synthetic Data Warning] {file}: {e}")

    return data_cache


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n[CONFIG] AI Transition App Configuration Loaded:")
print(f"- Base URL: {BASE_URL}")
print(f"- API Key Loaded: {'YES' if GENAI_API_KEY != 'REPLACE_ME_WITH_YOUR_KEY_HERE' else 'NO'}")
print(f"- Synthetic Data Directory: {SYNTHETIC_DATA_DIR}")
print(f"- Default Chat Model: {DEFAULT_CHAT_MODEL}")
print(f"- Default Agent Model: {DEFAULT_AGENT_MODEL}")
print("===========================================================\n")
