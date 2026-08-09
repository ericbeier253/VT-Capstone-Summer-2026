import json
import logging
import os
import re
from typing import Any

import streamlit as st
from google import genai
from google.cloud import firestore

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
def parse_gemini_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from Gemini response text."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return None

    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def search_with_gemini(gemini_client, prompt: str, object_index: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Use Gemini to semantically match the user's prompt against Firestore object metadata."""
    if not gemini_client or not object_index:
        return None

    candidates = []
    for entry in object_index[:20]:
        obj = entry["object"]
        candidates.append(
            {
                "doc_id": entry["doc_id"],
                "collection": entry.get("source_collection"),
                "object_name": obj.get("object_name"),
                "object_description": obj.get("object_description"),
                "object_location": obj.get("object_location"),
                "scene_meta": entry.get("scene_meta", {}),
                "run_id": entry.get("event", {}).get("run_id"),
                "timestamp": entry.get("event", {}).get("timestamp"),
            }
        )

    search_prompt = (
        "You are a search assistant. The user asked a question and you have a set of Firestore object metadata records. "
        "Use the records only to determine which object or objects best match the user question and answer clearly. "
        "If multiple objects are requested, return all matching objects. If no object matches, say so. "
        "Your answer must be valid JSON only, with these fields:\n"
        "  - answer: a short plain-English reply\n"
        "  - matched_objects: an array of selected object metadata records\n"
        "  - source_doc_ids: an array of Firestore document ids or an empty array\n"
        "  - run_ids: an array of associated run ids or an empty array\n"
        "  - timestamps: an array of matching timestamps or an empty array\n"
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[search_prompt, json.dumps({"query": prompt, "objects": candidates}, indent=2, default=str)],
        )
        result = parse_gemini_json(response.text)
        return result
    except Exception as exc:
        logger.error("Gemini semantic search failed: %s", exc)
        return None


def get_firestore_object(_firestore_client, prompt: str, gemini_client) -> dict[str, Any]:
    """Search Firestore for a matching object and return a structured payload for model context."""
    if not _firestore_client:
        return {"error": "Firestore is not configured yet. Add GCP_PROJECT to your .env file and restart the app."}

    events = get_firestore_events(_firestore_client, limit=50)
    object_index = ingest_object_index(events)
    if not object_index:
        return {"error": "No object metadata was found in Firestore."}

    gemini_result = search_with_gemini(gemini_client, prompt, object_index)
    if gemini_result and gemini_result.get("answer"):
        return {
            "answer": gemini_result["answer"],
            "matches": gemini_result.get("matched_objects", []),
            "source_doc_ids": gemini_result.get("source_doc_ids", []),
            "run_ids": gemini_result.get("run_ids", []),
            "timestamps": gemini_result.get("timestamps", []),
        }

    # Fallback text matching for requests with multiple known objects.
    prompt_lower = prompt.lower()
    fallback_matches = []
    seen_names = set()
    for entry in object_index:
        obj = entry.get("object", {})
        name = str(obj.get("object_name", "")).lower()
        if not name or name in seen_names:
            continue

        if name in prompt_lower or any(word in prompt_lower for word in re.findall(r"\w+", name)):
            fallback_matches.append(entry)
            seen_names.add(name)

    if fallback_matches:
        matched_objects = []
        source_doc_ids = []
        run_ids = []
        timestamps = []
        for entry in fallback_matches:
            obj = entry.get("object", {})
            matched_objects.append(obj)
            source_doc_ids.append(entry.get("doc_id"))
            run_ids.append(entry.get("event", {}).get("run_id"))
            timestamps.append(entry.get("event", {}).get("timestamp"))

        answer = "I found the following objects: " + ", ".join(
            f'{obj.get("object_name", "object")} in {obj.get("object_location", "the scene")}' for obj in matched_objects
        )
        return {
            "answer": answer,
            "matches": matched_objects,
            "source_doc_ids": source_doc_ids,
            "run_ids": run_ids,
            "timestamps": timestamps,
        }

    return {"error": "I could not find any objects matching that request in the Firestore data."}


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

        # First do the structured Firestore lookup, using Gemini for semantic search and response handling.
        payload = get_firestore_object(firestore_client, prompt, gemini_client)
        reply = payload.get("answer")
        if not reply:
            reply = get_plain_english_location_reply(payload, prompt)

        with st.expander("🔎 Gemini debug view", expanded=True):
            st.markdown("**Firestore payload**")
            st.code(json.dumps(payload, indent=2, default=str), language="json")
            st.markdown("**Gemini extraction result**")
            st.code(extracted_result, language="text")
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
