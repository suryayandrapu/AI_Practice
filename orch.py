# orchestrator/team.py
from typing import Dict, Any
from pref import PreferencesAgent
from data import DataIngestAgent
from const import ConstraintsAgent
from itin import ItineraryPlannerAgent
from alter import AlternativesAgent
from synth import SynthesizerAgent

def run_travel_team(task: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Normalize preferences
    pref = PreferencesAgent().run(task)
    task.update(pref)

    # 2) Summarize destination data
    cat = DataIngestAgent().run(task)
    task.update(cat)

    # 3) Consolidate constraints
    cons = ConstraintsAgent().run(task)
    task.update(cons)

    # 4) Plan itinerary
    plan = ItineraryPlannerAgent().run(task)
    task.update(plan)

    # 5) Add alternates
    alts = AlternativesAgent().run(task)
    task.update(alts)

    # 6) Final synthesis
    final = SynthesizerAgent().run(task)
    task.update(final)

    return task