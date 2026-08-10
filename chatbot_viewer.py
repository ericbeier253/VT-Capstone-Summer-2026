import io
import json
import logging
import os
import re
from typing import Any

import streamlit as st
from datetime import timedelta

from google import genai
from google.cloud import firestore, storage
from RAG_delivery.object_search import search_object, get_latest_distinct_objects, get_results, parse_query

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



def build_gemini_prompt(query: str, search_payload: dict[str, Any]) -> str:
    """Build the prompt used to ask Gemini for the final location reply."""
    search_json = json.dumps(search_payload, indent=2, default=str)
    return (
        "You are getting the results of a semantic search for an object requested by the user. "
        "Based on the JSON data provided, describe the location of the object as accurately as possible. "
        "The search results array is already strictly sorted from newest to oldest. Rank 1 is deterministically the most recent sighting of the object. Focus your primary answer on Rank 1 as the current whereabouts, and optionally mention the older locations (Rank 2+) as places it was seen previously if relevant. "
        "Pay special attention to the 'environment' description and 'scene_meta' to provide clear, spatially aware directions on where to find the object in the physical space. "
        "IMPORTANT: DO NOT include the raw bounding box pixel coordinates in your reply, and DO NOT describe where the object is located within the image frame (e.g. avoid saying 'at the bottom center of the image'). The user can already see the image. Describe the location in the physical room using natural language only."
        "\n\nSearch payload:\n"
        f"{search_json}\n\n"
        "User question:\n"
        f"{query}\n"
    )


def generate_signed_gcs_url(bucket_name: str, object_path: str) -> str:
    """Generate a signed URL for a Google Cloud Storage object."""
    if not bucket_name or not object_path:
        return ""

    try:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            os.path.join(os.path.dirname(__file__), '.secrets', 'aria-uploader-key.json')
        )
        client = storage.Client(credentials=creds)
    except Exception:
        client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)

    try:
        return blob.generate_signed_url(
            expiration=timedelta(hours=1),
            version="v4",
        )
    except Exception:
        return f"https://storage.googleapis.com/{bucket_name}/{object_path}"


def build_public_url_from_crop_path(crop_path: str) -> str:
    """Build a complete public or signed Storage URL for the crop path."""
    if not isinstance(crop_path, str):
        return ""

    path = crop_path.strip()
    if not path:
        return ""

    if path.startswith("https://storage.googleapis.com/"):
        return path

    bucket_name = os.environ.get("GCS_BUCKET") or os.environ.get("BUCKET_NAME") or "project-aria-gaze-photos-eb-01"

    if path.startswith("gs://"):
        path = path[len("gs://") :]
        if "/" in path:
            bucket_name, path = path.split("/", 1)
        else:
            path = ""

    segments = [segment for segment in path.split("/") if segment]
    if segments and segments[0].startswith("project-aria-gaze-photos"):
        bucket_name = segments[0]
        segments = segments[1:]

    run_id = next((segment for segment in segments if segment.startswith("run_")), None)
    gaze_trigger = next((segment for segment in segments if segment.startswith("gaze_trigger")), None)

    if run_id and gaze_trigger:
        if not gaze_trigger.lower().endswith(".jpg"):
            gaze_trigger += ".jpg"
        return generate_signed_gcs_url(bucket_name, f"{run_id}/{gaze_trigger}")

    if run_id and segments:
        fallback_trigger = segments[-1]
        if not fallback_trigger.lower().endswith(".jpg"):
            fallback_trigger += ".jpg"
        return generate_signed_gcs_url(bucket_name, f"{run_id}/{fallback_trigger}")

    if segments:
        public_path = "/".join(segments)
        return generate_signed_gcs_url(bucket_name, public_path)

    return f"https://storage.googleapis.com/{bucket_name}/"


def build_public_url(payload: dict[str, Any]) -> list[str]:
    """Collect public or signed URLs for crop_path values in the get_object JSON response."""
    public_urls: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            img_path = value.get("parent_image") or value.get("crop_path")
            if img_path and isinstance(img_path, str):
                public_urls.append(build_public_url_from_crop_path(img_path))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return public_urls


def get_crop_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect crop entries with crop_path and bounding boxes data."""
    entries: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            img_path = value.get("parent_image") or value.get("crop_path")
            if img_path and isinstance(img_path, str):
                entries.append(
                    {
                        "crop_path": img_path,
                        "bounding_box": value.get("bounding_boxes") or value.get("boundary_box") or value.get("bounding_box"),
                    }
                )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return entries


def render_annotated_image(image_url: str, bounding_box: dict[str, Any] | None) -> None:
    """Render a crop image and draw a red boundary box if available."""
    if not bounding_box:
        st.image(image_url, use_container_width=True)
        return

    try:
        from PIL import Image, ImageDraw
        import requests

        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        draw = ImageDraw.Draw(image)

        print(f"Drawing bounding box: {bounding_box}, path {image_url}")
        
        # Handle cases where bounding_box is a list of dicts instead of a single dict
        if isinstance(bounding_box, list):
            bbox_dict = bounding_box[0] if bounding_box else {}
        else:
            bbox_dict = bounding_box
            
        x1 = int(bbox_dict.get("x1", 0))
        y1 = int(bbox_dict.get("y1", 0))
        x2 = int(bbox_dict.get("x2", 0))
        y2 = int(bbox_dict.get("y2", 0))

        # Ensure valid coordinates
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        
        # Scale coordinates from Gemini's 1000x1000 grid to actual image dimensions
        img_width, img_height = image.size
        left = int((left / 1000.0) * img_width)
        right = int((right / 1000.0) * img_width)
        top = int((top / 1000.0) * img_height)
        bottom = int((bottom / 1000.0) * img_height)

        # Draw a red rectangle outline using the provided pixel coordinates.
        draw.rectangle([left, top, right, bottom], outline="red", width=4)
        st.image(image, use_container_width=True)
    except Exception as e:
        print(f"Error processing image: {e}")
        st.image(image_url, use_container_width=True)


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


def get_firestore_object(_firestore_client, gemini_client, prompt: str) -> dict[str, Any]:
    """Search Firestore for a matching object and return a structured payload for model context using object_search helpers."""
    if not _firestore_client:
        return {"error": "Firestore is not configured yet. Add GCP_PROJECT to your .env file and restart the app."}
    
    if not gemini_client:
        return {"error": "Gemini API key is required to parse the object name from your question."}

    # Extract a concise object name from the user's prompt using the robust Gemini parser
    parsed = parse_query(prompt)
    objects = parsed.get("objects", [])
    
    if not objects:
        return {"error": "No object could be extracted from your query. Try asking something like 'where is my nasal spray?'"}
        
    requested_name = objects[0]

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
        payload = get_firestore_object(firestore_client, gemini_client, prompt)
        crop_entries = get_crop_entries(payload)
        image_paths = [build_public_url_from_crop_path(entry["crop_path"]) for entry in crop_entries if entry.get("crop_path")]
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
            if crop_entries:
                with st.expander("🖼️ Image view (top 3 ranked images)", expanded=True):
                    for idx, entry in enumerate(crop_entries[:3]):
                        image_url = build_public_url_from_crop_path(entry["crop_path"])
                        st.markdown(f"**Image {idx + 1}: {entry['crop_path']}**")
                        render_annotated_image(image_url, entry.get("bounding_box"))


if __name__ == "__main__":
    main()
