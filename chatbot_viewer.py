import json
import logging
import os
import re
from typing import Any

import streamlit as st
from google import genai
from google.cloud import firestore
from RAG_delivery.object_search import search_object, get_latest_distinct_objects, get_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Streamlit page appearance.
st.set_page_config(page_title="Project Aria Chatbot", page_icon="💬", layout="centered")

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val.strip().strip('"').strip("'")

GCP_PROJECT = os.environ.get("GCP_PROJECT", "project-aria-501223")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


@st.cache_resource
def get_clients():
    """Initialize and cache Firestore and Gemini clients."""
    fs = firestore.Client(project=GCP_PROJECT)
    gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
    return fs, gemini


@st.cache_data(ttl=60)
def get_firestore_events(_firestore_client, limit: int = 20):
    """Fetch recent object documents from Firestore for chat context."""
    if not _firestore_client:
        return []

    events = []
    for collection_name in ["rag_object_collection", "gaze_events"]:
        docs = _firestore_client.collection(collection_name).limit(limit).stream()
        for doc in docs:
            data = doc.to_dict() or {}
            data["id"] = doc.id
            data["__collection__"] = collection_name
            events.append(data)
        if events:
            break

    events.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
    return events


def ingest_object_index(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a lightweight lookup index from Firestore object payloads."""
    index = []
    for event in events:
        if event.get("object_name"):
            index.append(
                {
                    "doc_id": event.get("id"),
                    "scene_meta": event.get("scene_meta", {}),
                    "object": event,
                    "event": event,
                    "source_collection": event.get("__collection__"),
                }
            )
            continue

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
                    "source_collection": event.get("__collection__"),
                }
            )
    return index


def extract_object_name(prompt: str) -> str:
    """Extract the object name from the user's question so it can be matched in Firestore."""
    cleaned = prompt.strip()
    if not cleaned:
        return ""

    for keyword in ["looking for", "find", "search for", "show me", "i need", "object", "I forget", "Lost my"]:
        if keyword in cleaned.lower():
            match = re.search(rf"{re.escape(keyword)}\s+(.+)", cleaned, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip("? .,:;!")

    return cleaned.strip("? .,:;!")

def build_gemini_prompt(query: str, search_payload: dict[str, Any]) -> str:
    """Build the prompt used to ask Gemini for the final location reply."""
    search_json = json.dumps(search_payload, indent=2, default=str)
    return (
        "You are getting the results of a sematic search on objects. Based on the json given, describe the location of the object."
        "\n\nSearch payload:\n"
        f"{search_json}\n\n"
        "User question:\n"
        f"{query}\n"
    )


def normalize_crop_path(crop_path: str) -> str:
    """Normalize a crop path into bucket_name/run_id/gaze_trigger."""
    if not isinstance(crop_path, str):
        return crop_path

    path = crop_path.strip()
    if not path:
        return crop_path

    bucket_name = os.environ.get("GCS_BUCKET") or os.environ.get("BUCKET_NAME")

    if path.startswith("gs://"):
        path = path[len("gs://") :]
        if "/" in path:
            bucket_name, path = path.split("/", 1)
        else:
            path = ""

    segments = [segment for segment in path.split("/") if segment]
    run_id = next((segment for segment in segments if segment.startswith("run_")), None)
    gaze_trigger = next((segment for segment in segments if segment.startswith("gaze_trigger")), None)

    if run_id and gaze_trigger:
        if bucket_name:
            return f"{bucket_name}/{run_id}/{gaze_trigger}"
        return f"{run_id}/{gaze_trigger}"

    if run_id and segments:
        fallback_trigger = segments[-1]
        if bucket_name:
            return f"{bucket_name}/{run_id}/{fallback_trigger}"
        return f"{run_id}/{fallback_trigger}"

    return crop_path


def get_normalized_crop_paths(payload: dict[str, Any]) -> list[str]:
    """Collect normalized crop paths from the get_object JSON response without modifying it."""
    normalized_paths: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "crop_path" and isinstance(child, str):
                    normalized_paths.append(normalize_crop_path(child))
                else:
                    walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return normalized_paths


def ask_gemini_for_reply(gemini_client, prompt: str) -> str:
    """Ask Gemini for the final chat reply using structured search context."""
    if not gemini_client:
        return ""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[prompt],
        )
        return extract_gemini_text(response) or ""
    except Exception as exc:
        logger.error("Gemini location reply failed: %s", exc)
        return ""


def get_firestore_object(_firestore_client, prompt: str) -> dict[str, Any]:
    """Search Firestore for a matching object and return a structured payload for model context using object_search helpers."""
    if not _firestore_client:
        return {"error": "Firestore is not configured yet. Add GCP_PROJECT to your .env file and restart the app."}

    # Extract a concise object name from the user's prompt
    requested_name = extract_object_name(prompt)
    if not requested_name:
        return {"error": "No object name could be extracted from the query."}

    # Use the RAG_delivery.object_search functions to perform search and format results
    matches = search_object(requested_name)
    latest = get_latest_distinct_objects(matches)
    payload = get_results(requested_name, latest)
    return payload


def get_plain_english_location_reply(payload: dict[str, Any], prompt: str) -> str:
    """In a couple of sentencese, describe the object's location."""
    if not payload:
        return "I couldn't find a matching object in the Firestore data."

    if payload.get("error"):
        return payload["error"]

    if payload.get("answer"):
        return payload["answer"]

    matches = payload.get("matches") or []
    if matches:
        descriptions = []
        for obj in matches:
            name = obj.get("object_name") or "object"
            location = obj.get("object_location") or obj.get("scene_meta", {}).get("location") or "the scene"
            descriptions.append(f"{name} in {location}")
        return "I found the following objects: " + ", ".join(descriptions)

    object_data = payload.get("object") or {}
    object_name = object_data.get("object_name") or "the object"
    scene_meta = payload.get("scene_meta") or {}

    location_candidates = []
    for key in ["location", "room", "scene", "scene_name", "environment", "place", "area"]:
        value = scene_meta.get(key)
        if value:
            location_candidates.append(str(value))

    if not location_candidates:
        for key in ["description", "scene_description", "summary"]:
            value = scene_meta.get(key)
            if value:
                location_candidates.append(str(value))
                break

    location = location_candidates[0] if location_candidates else "the detected scene"
    return f"I found {object_name} in {location}."


def extract_gemini_text(response: Any) -> str:
    """Extract text from the Gemini SDK response in a way that works across response shapes."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                parts.append(part_text.strip())

    if parts:
        return "\n".join(parts).strip()

    return ""


def main() -> None:
    """Run the main chat UI flow for Project Aria object lookup."""
    st.title("💬 Project Aria Chatbox")
    st.caption("Ask questions about the detected objects and I will use Gemini with the Firestore context to respond.")

    if not GCP_PROJECT:
        st.error("GCP_PROJECT not found in .env")
        st.stop()

    firestore_client, gemini_client = get_clients()

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

    if not GEMINI_API_KEY:
        st.sidebar.info("Gemini is optional here; Firestore object lookup is enabled without it.")

    if prompt := st.chat_input("What are you looking for?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Use object_search to produce a structured search payload, then give that JSON to Gemini
        payload = get_firestore_object(firestore_client, prompt)
        image_paths = get_normalized_crop_paths(payload)
        print(f"Normalized crop paths: {image_paths}")

        gemini_prompt_text = ""
        gemini_reply = ""
        if gemini_client:
            gemini_prompt_text = build_gemini_prompt(prompt, payload)
            gemini_reply = ask_gemini_for_reply(gemini_client, gemini_prompt_text)
            reply = gemini_reply or payload.get("answer") or get_plain_english_location_reply(payload, prompt)
        else:
            reply = payload.get("answer") or get_plain_english_location_reply(payload, prompt)

        with st.expander("🔎 Debug view", expanded=True):
            st.markdown("**Firestore payload**")
            st.code(json.dumps(payload, indent=2, default=str), language="json")
            if gemini_client:
                st.markdown("**Gemini prompt**")
                st.code(gemini_prompt_text, language="text")
                st.markdown("**Gemini reply**")
                st.code(gemini_reply, language="text")
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
