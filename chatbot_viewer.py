import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.cloud import firestore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Streamlit page appearance.
st.set_page_config(page_title="Project Aria Chatbot", page_icon="💬", layout="centered")


def load_environment() -> tuple[str, str]:
    """Load project credentials from the local .env file if it exists."""
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    return os.environ.get("GCP_PROJECT", ""), os.environ.get("GEMINI_API_KEY", "")


@st.cache_data(ttl=60)
def get_firestore_events(firestore_client, limit: int = 20):
    """Fetch recent gaze event documents from Firestore for chat context."""
    if not firestore_client:
        return []

    # Query the main collection that stores the gaze-event metadata.
    docs = firestore_client.collection("gaze_events").limit(limit).stream()
    events = []
    for doc in docs:
        data = doc.to_dict() or {}
        data["id"] = doc.id
        events.append(data)

    events.sort(key=lambda item: item.get("timestamp", 0) or 0, reverse=True)
    return events


def ingest_object_index(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a lightweight lookup index from object names to their Firestore payloads."""
    index = []
    for event in events:
        # Each Firestore document contains a llm_analysis payload with scene metadata and object lists.
        llm_analysis = event.get("llm_analysis") or {}
        objects = llm_analysis.get("objects") or []
        for obj in objects:
            if not obj.get("object_name"):
                continue
            index.append(
                {
                    "doc_id": event.get("id"),
                    "scene_meta": llm_analysis.get("scene_meta", {}),
                    "object": obj,
                    "event": event,
                }
            )
    return index


def extract_object_name(prompt: str) -> str:
    """Extract the object name from the user's question so it can be matched in Firestore."""
    cleaned = prompt.strip()
    if not cleaned:
        return ""

    for keyword in ["looking for", "find", "search for", "show me", "i need", "object"]:
        if keyword in cleaned.lower():
            match = re.search(rf"{re.escape(keyword)}\s+(.+)", cleaned, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip("? .,:;!")

    return cleaned.strip("? .,:;!")


def find_matching_objects(index: list[dict[str, Any]], object_name: str) -> list[dict[str, Any]]:
    """Find Firestore entries whose object_name best matches the user request."""
    if not object_name:
        return []

    # Compare the user’s requested object name against every indexed object name.
    # A simple substring match is enough for the first version of this chatbot.
    target = object_name.lower().strip()
    matches = []
    for entry in index:
        obj = entry.get("object", {})
        name = str(obj.get("object_name", "")).lower()
        if target in name or name in target:
            matches.append(entry)
    return matches


def get_firestore_reply(firestore_client, prompt: str) -> str:
    """Search Firestore for a matching object and return the relevant JSON payload."""
    if not firestore_client:
        return "Firestore is not configured yet. Add GCP_PROJECT to your .env file and restart the app."

    # Pull the latest event records and build an object-to-payload lookup index.
    events = get_firestore_events(firestore_client, limit=50)
    object_index = ingest_object_index(events)
    object_name = extract_object_name(prompt)

    if not object_name:
        return "Please tell me the object name you want to look up."

    matches = find_matching_objects(object_index, object_name)
    if not matches:
        return f"I could not find an object matching '{object_name}' in the Firestore data."

    # Use the first strong match as the main result for now.
    best_match = matches[0]
    object_data = best_match.get("object", {})
    scene_meta = best_match.get("scene_meta", {})
    event_data = best_match.get("event", {})

    # Package the matched object and the surrounding scene context into a JSON response.
    payload = {
        "scene_meta": scene_meta,
        "object": object_data,
        "source_event_id": best_match.get("doc_id"),
        "timestamp": event_data.get("timestamp"),
        "run_id": event_data.get("run_id"),
    }

    return (
        f"I found a match for '{object_data.get('object_name')}'.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )


def main() -> None:
    """Run the main chat UI flow for Project Aria object lookup."""
    st.title("💬 Project Aria Chatbox")
    st.caption("Tell me what you are looking for and I will search the Firestore object data for the matching JSON payload.")

    GCP_PROJECT = os.environ.get("GCP_PROJECT", "project-aria-501223")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    firestore_client = firestore.Client(project=GCP_PROJECT)
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "What are you looking for?",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not GCP_PROJECT:
        st.sidebar.error("Set GCP_PROJECT in your .env file before using Firestore.")
    if not GEMINI_API_KEY:
        st.sidebar.info("Gemini is optional here; Firestore object lookup is enabled without it.")

    # Wait for the user to enter a question and then search Firestore for the object.
    if prompt := st.chat_input("What are you looking for?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # First do the structured Firestore lookup, then optionally enhance the result with Gemini.
        reply = get_firestore_reply(firestore_client, prompt)
        if gemini_client and GEMINI_API_KEY:
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=[
                        f"Use the Firestore payload below to answer the user clearly.\n\n{reply}",
                        f"User question: {prompt}",
                    ],
                )
                reply = response.text.strip()
            except Exception as exc:
                logger.error("Gemini request failed: %s", exc)
                reply = f"Firestore lookup completed, but Gemini failed: {exc}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    st.sidebar.header("Status")
    st.sidebar.caption(f"Firestore project: {GCP_PROJECT or 'not configured'}")
    st.sidebar.caption(f"Gemini configured: {'yes' if GEMINI_API_KEY else 'no'}")
    st.sidebar.caption(f"Recent Firestore events: {len(get_firestore_events(firestore_client, limit=20))}")

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "What are you looking for?",
            }
        ]
        st.rerun()


if __name__ == "__main__":
    main()
