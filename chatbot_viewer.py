import json
import logging
import os
import re
from typing import Any
from datetime import datetime

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
GCS_BUCKET = os.environ.get("GCS_BUCKET") or GCP_PROJECT
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
            if not obj.get("img_path") and event.get("img_path"):
                obj["img_path"] = event.get("img_path")
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


def extract_requested_items(prompt: str) -> list[str]:
    """Extract a list of requested items from a user query."""
    cleaned = prompt.strip().lower()
    cleaned = re.sub(r"^(i\s+am\s+)?(looking|searching)\s+for\s+", "", cleaned)
    cleaned = re.sub(r"^(find|find me|search for)\s+", "", cleaned)
    parts = re.split(r",| and |&", cleaned)
    items = [part.strip(" ?.!\n") for part in parts if part.strip()]
    return [item for item in items if item]


def score_object_match(item: str, obj: dict[str, Any]) -> int:
    """Score how well an object matches a requested item phrase."""
    item_lower = item.lower()
    name = str(obj.get("object_name", "")).lower()
    desc = str(obj.get("object_description", "")).lower()
    location = str(obj.get("object_location", "")).lower()

    score = 0
    if item_lower == name:
        score += 120
    if item_lower in name:
        score += 70
    words = [w for w in re.findall(r"\w+", item_lower) if len(w) > 1]
    if words:
        if all(word in name for word in words):
            score += 40
        if all(word in desc for word in words):
            score += 20
        if all(word in location for word in words):
            score += 10
        score += sum(8 for word in words if word in name)
        score += sum(3 for word in words if word in desc)
        score += sum(2 for word in words if word in location)

    # Prefer a more concise name when the query is generic and multiple monitors exist.
    if item_lower in ["monitor", "laptop", "keys", "keyboard"] and item_lower in name:
        name_len = len(name.split())
        score += max(0, 10 - name_len)

    # Prefer exact qualifiers if provided in the user request.
    if "curved" in item_lower and "curved" in name:
        score += 30
    if "wall" in item_lower and "wall" in name:
        score += 30
    if "small" in item_lower and "small" in name:
        score += 20
    if "large" in item_lower and "large" in name:
        score += 20

    if item_lower in desc:
        score += 25
    if item_lower in location:
        score += 10

    # Give a small boost to object names that include the requested keyword at the start.
    if name.startswith(item_lower):
        score += 15

    return score


def find_best_match_for_item(item: str, matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the best matching object for a requested item from a match list."""
    best = None
    best_score = -1
    for obj in matches:
        score = score_object_match(item, obj)
        if score > best_score:
            best_score = score
            best = obj
        elif score == best_score and best is not None:
            best_name = str(best.get("object_name", "")).lower()
            obj_name = str(obj.get("object_name", "")).lower()
            if len(obj_name.split()) < len(best_name.split()):
                best = obj
    return best

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
    sorted_index = sorted(
        object_index,
        key=lambda entry: entry.get("event", {}).get("timestamp", ""),
        reverse=True,
    )
    for entry in sorted_index[:20]:
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
                "img_path": entry.get("event", {}).get("img_path"),
            }
        )

    search_prompt = (
        "You are a search assistant. The user asked a question and you have a set of Firestore object metadata records. "
        "Use the records only to determine which object or objects best match the user question and answer clearly. "
        "Prefer more recent images when returning matches, based on the timestamp field. Use the newest matching image for each object. "
        "If the user asks for multiple items, return results for each distinct requested object rather than only the single newest object. "
        "If no object matches, say so. "
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
        run_ids = gemini_result.get("run_ids", []) or []
        timestamps = gemini_result.get("timestamps", []) or []
        matched_objects = gemini_result.get("matched_objects", []) or []

        if timestamps and matched_objects and len(timestamps) == len(matched_objects):
            latest_by_name: dict[str, dict[str, Any]] = {}
            for obj, ts, rid in zip(matched_objects, timestamps, run_ids):
                name = str(obj.get("object_name", "")).lower()
                if not name:
                    continue
                if name not in latest_by_name or ts > latest_by_name[name]["timestamp"]:
                    latest_by_name[name] = {"object": obj, "timestamp": ts, "run_id": rid}

            if latest_by_name:
                matched_objects = [entry["object"] for entry in latest_by_name.values()]
                run_ids = [entry["run_id"] for entry in latest_by_name.values()]
                timestamps = [entry["timestamp"] for entry in latest_by_name.values()]

        return {
            "answer": gemini_result["answer"],
            "matches": matched_objects,
            "source_doc_ids": gemini_result.get("source_doc_ids", []),
            "run_ids": run_ids,
            "timestamps": timestamps,
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
        latest_by_name: dict[str, dict[str, Any]] = {}
        for entry in fallback_matches:
            obj = entry.get("object", {})
            name = str(obj.get("object_name", "")).lower()
            if not name:
                continue
            ts = entry.get("event", {}).get("timestamp", "")
            run_id = entry.get("event", {}).get("run_id")
            if name not in latest_by_name or ts > latest_by_name[name]["timestamp"]:
                latest_by_name[name] = {"entry": entry, "timestamp": ts, "run_id": run_id}

        matched_objects = []
        source_doc_ids = []
        run_ids = []
        timestamps = []
        for record in latest_by_name.values():
            entry = record["entry"]
            obj = entry.get("object", {})
            matched_objects.append(obj)
            source_doc_ids.append(entry.get("doc_id"))
            run_ids.append(record["run_id"])
            timestamps.append(record["timestamp"])

        answer = "Found the " + ", ".join(
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
    """Return a clearer response, with sections for multiple matching objects."""
    if not payload:
        return "I couldn't find a matching object in the Firestore data."

    if payload.get("error"):
        return payload["error"]

    matches = payload.get("matches") or []
    run_ids = payload.get("run_ids") or []

    if matches:
        requested_items = extract_requested_items(prompt)
        if requested_items:
            def parse_timestamp(ts: str) -> datetime | None:
                if not ts:
                    return None
                try:
                    return datetime.fromisoformat(str(ts))
                except Exception:
                    return None

            def get_latest_entry_for_name(name: str, match_list: list[dict[str, Any]]) -> dict[str, Any] | None:
                name_l = (name or "").lower()
                candidates = [m for m in match_list if str(m.get("object_name", "")).lower() == name_l]
                if not candidates:
                    return None
                best = None
                best_ts = None
                for c in candidates:
                    ts = parse_timestamp(c.get("timestamp") or c.get("event", {}).get("timestamp"))
                    if ts is None:
                        continue
                    if best_ts is None or ts > best_ts:
                        best_ts = ts
                        best = c
                return best or candidates[0]

            reply_lines = []
            for item in requested_items:
                best_match = find_best_match_for_item(item, matches)
                if best_match:
                    latest = get_latest_entry_for_name(best_match.get("object_name", ""), matches)
                    use = latest or best_match
                    name = use.get("object_name") or "Unknown object"
                    location = use.get("object_location") or use.get("scene_meta", {}).get("location") or "an unknown location"
                    reply_lines.append(f"I found {name} in {location}.")
            if reply_lines:
                return "\n".join(reply_lines)

        lines = [f"### Found {len(matches)} matching object{'s' if len(matches) != 1 else ''}", ""]
        for index, obj in enumerate(matches, start=1):
            name = obj.get("object_name") or "Unknown object"
            description = obj.get("object_description") or "No description available."
            location = obj.get("object_location") or obj.get("scene_meta", {}).get("location") or "unknown location"
            object_run_id = run_ids[index - 1] if index - 1 < len(run_ids) else None
            raw_img_path = obj.get("crop_path")
            gcs_uri = resolve_gcs_uri(raw_img_path)

            lines.append(f"**{index}. {name}**")
            lines.append(f"- Description: {description}")
            lines.append(f"- Location: {location}")
            if object_run_id:
                lines.append(f"- Run ID: `{object_run_id}`")
            if gcs_uri:
                lines.append(f"- Image GCS path: `{gcs_uri}`")
            lines.append("")

        source_docs = payload.get("source_doc_ids") or []
        if source_docs:
            lines.append("### Source Firestore document IDs")
            for doc_id in source_docs:
                lines.append(f"- `{doc_id}`")
            lines.append("")

        return "\n".join(lines)

    if payload.get("answer"):
        return payload["answer"]

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
