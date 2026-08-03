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
    """Fetch recent gaze event documents from Firestore for chat context."""
    if not _firestore_client:
        return []

    # Query the main collection that stores the gaze-event metadata.
    docs = _firestore_client.collection("gaze_events").limit(limit).stream()
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

    for keyword in ["looking for", "find", "search for", "show me", "i need", "object", "I forget", "Lost my"]:
        if keyword in cleaned.lower():
            match = re.search(rf"{re.escape(keyword)}\s+(.+)", cleaned, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip("? .,:;!")

    return cleaned.strip("? .,:;!")


def parse_gemini_match_response(response_text: str) -> dict[str, Any]:
    """Parse a Gemini response that includes a JSON match block for the object search."""
    if not response_text:
        return {}

    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return {}

    try:
        payload = json.loads(match.group(1))
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        logger.warning("Gemini match response was not valid JSON: %s", response_text)

    return {}


def get_firestore_object(_firestore_client, prompt: str, gemini_client: Any | None = None) -> dict[str, Any]:
    """Search Firestore for a matching object and return a structured payload for model context."""
    if not _firestore_client:
        return {"error": "Firestore is not configured yet. Add GCP_PROJECT to your .env file and restart the app."}

    # Pull the latest event records and build a simple object lookup index.
    events = get_firestore_events(_firestore_client, limit=50)
    object_index = ingest_object_index(events)
    object_name = extract_object_name(prompt)

    if not object_name:
        return {"error": "Please tell me the object name you want to look up."}

    if gemini_client and GEMINI_API_KEY:
        try:
            serialized_index = []
            for idx, entry in enumerate(object_index[:20]):
                obj = entry.get("object", {})
                serialized_index.append(
                    {
                        "index": idx,
                        "object_name": obj.get("object_name"),
                        "description": obj.get("object_description") or obj.get("description"),
                        "scene": entry.get("scene_meta", {}),
                    }
                )

            gemini_prompt = "\n\n".join(
                [
                    "Search the following object catalog and find the best semantic match for the user's request.",
                    "Return a JSON object with keys: match_index, matched_object_name, reason.",
                    "Use the whole catalog, not just the object_name field. Consider descriptions and scene context.",
                    json.dumps(serialized_index, indent=2, default=str),
                    f"User request: {prompt}",
                ]
            )
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[{"role": "user", "parts": [{"text": gemini_prompt}]}],
            )
            response_text = extract_gemini_text(response)
            parsed = parse_gemini_match_response(response_text)
            if parsed and parsed.get("match_index") is not None:
                match_index = int(parsed["match_index"])
                if 0 <= match_index < len(object_index):
                    entry = object_index[match_index]
                    obj = entry.get("object", {})
                    scene_meta = entry.get("scene_meta", {})
                    event_data = entry.get("event", {})
                    return {
                        "scene_meta": scene_meta,
                        "object": obj,
                        "source_event_id": entry.get("doc_id"),
                        "timestamp": event_data.get("timestamp"),
                        "run_id": event_data.get("run_id"),
                        "match_reason": parsed.get("reason"),
                        "matched_object_name": parsed.get("matched_object_name"),
                    }
        except Exception as exc:
            logger.exception("Gemini semantic object search failed: %s", exc)

    # Compare the requested object name against every indexed object name.
    target = object_name.lower().strip()
    for entry in object_index:
        obj = entry.get("object", {})
        name = str(obj.get("object_name", "")).lower()
        if target in name or name in target:
            object_data = obj
            scene_meta = entry.get("scene_meta", {})
            event_data = entry.get("event", {})

            return {
                "scene_meta": scene_meta,
                "object": object_data,
                "source_event_id": entry.get("doc_id"),
                "timestamp": event_data.get("timestamp"),
                "run_id": event_data.get("run_id"),
            }

    return {"error": f"I could not find an object matching '{object_name}' in the Firestore data."}


def get_plain_english_location_reply(payload: dict[str, Any], prompt: str) -> str:
    """In a couple of sentencese, describe the object's location."""
    if not payload:
        return "I couldn't find a matching object in the Firestore data."

    if payload.get("error"):
        return payload["error"]

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

        payload = get_firestore_object(firestore_client, prompt, gemini_client=gemini_client)
        extracted_result = payload.get("matched_object_name") or payload.get("object", {}).get("object_name") or ""
        logging_prompt = None
        if gemini_client and GEMINI_API_KEY:
            try:
                gemini_prompt = "\n\n".join(
                    [
                        "Use the Firestore payload below as hidden context to answer the user's question.",
                        "Write a rich response in 3 to 4 complete sentences.",
                        "Be natural and conversational, mention the object name, location, and relevant context from the payload.",
                        json.dumps(payload, indent=2, default=str),
                        f"User question: {prompt}",
                    ]
                )
                logging_prompt = gemini_prompt
                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[{"role": "user", "parts": [{"text": gemini_prompt}]}],
                )
                reply = extract_gemini_text(response)
                if not reply:
                    raise RuntimeError("Gemini returned no usable text")
            except Exception as exc:
                logger.exception("Gemini request failed: %s", exc)
                reply = get_plain_english_location_reply(payload, prompt)
        else:
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
