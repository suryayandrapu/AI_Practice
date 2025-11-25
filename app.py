# app.py
import streamlit as st
from datetime import date
from orch import run_travel_team

st.set_page_config(page_title="Travel Itinerary – Multi-Agent (LLaMA 2)", layout="wide")

st.title("Travel Itinerary Generator (Multi‑Agent, LLaMA 2 via Ollama)")

with st.sidebar:
    st.header("Trip details")
    destination = st.text_input("Destination", value="Singapore")
    start_date = st.date_input("Start date", value=date.today())
    end_date = st.date_input("End date", value=date.today())
    pace = st.selectbox("Pace", ["relaxed", "medium", "fast"], index=1)
    budget = st.text_input("Budget per day (e.g., INR 3000)", value="INR 3000")
    activities = st.multiselect(
        "Activity types",
        ["museums", "food", "nature", "shopping", "family", "adventure"],
        default=["food","museums","nature"]
    )
    must_see = st.text_area("Must-see items (comma-separated)", value="Gardens by the Bay, Marina Bay Sands")
    notes = st.text_area("Additional notes (accessibility, time windows, etc.)", value="")
    run_btn = st.button("Generate itinerary")

st.markdown("---")

sample_destination_data = {
    "attractions": [
        {"name": "Gardens by the Bay", "category": "nature", "hours": "09:00-21:00", "fee": "SGD 20", "coords": "1.2816,103.8636"},
        {"name": "Marina Bay Sands SkyPark", "category": "viewpoint", "hours": "10:00-22:00", "fee": "SGD 30", "coords": "1.2834,103.8607"},
        {"name": "National Museum of Singapore", "category": "museum", "hours": "10:00-19:00", "fee": "SGD 15", "coords": "1.2966,103.8485"},
    ],
    "transit": {"mrt": "dense network; typical travel 10-25 minutes", "cabs": "available widely"},
    "seasonal": {"november": "humid; occasional showers; indoor alternates recommended"},
}

if run_btn:
    with st.spinner("Planning your itinerary with multi‑agent reasoning..."):
        task = {
            "destination": destination,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "preferences": {
                "pace": pace,
                "budget_per_day": budget,
                "activity_types": activities,
                "must_see": must_see,
                "notes": notes,
            },
            "destination_data": sample_destination_data,
        }

        result = run_travel_team(task)

        st.success("Itinerary generated!")
        st.subheader("Final deliverable")
        st.markdown(result.get("final_markdown", "No output"))

        # Optional: show intermediate artifacts in tabs
        tabs = st.tabs(["Preferences", "Catalog", "Constraints", "Itinerary", "Alternates"])
        with tabs[0]:
            st.markdown(result.get("preferences_artifact", ""))
        with tabs[1]:
            st.markdown(result.get("catalog_artifact", ""))
        with tabs[2]:
            st.markdown(result.get("constraints_artifact", ""))
        with tabs[3]:
            st.markdown(result.get("itinerary_artifact", ""))
        with tabs[4]:
            st.markdown(result.get("alternates_artifact", ""))

else:
    st.info("Fill trip details in the sidebar and click Generate itinerary.")