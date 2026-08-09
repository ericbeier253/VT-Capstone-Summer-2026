import streamlit as st
from google.cloud import firestore
import os
import io
import urllib.request
from urllib.error import HTTPError, URLError
import pandas as pd
from PIL import Image, ImageDraw

st.set_page_config(page_title="Project Aria Gaze Viewer", layout="wide")

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val.strip().strip('"').strip("'")

st.title("👁️ Project Aria Gaze Events Viewer")

gcp_project = os.environ.get("GCP_PROJECT")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "project-aria-gaze-photos-eb-01")
if not gcp_project:
    st.error("GCP_PROJECT not found in .env")
    st.stop()

@st.cache_resource
def get_clients():
    fs = firestore.Client(project=gcp_project)
    return fs


def resolve_gcs_uri(raw_path: str | None) -> str | None:
    if not raw_path or not isinstance(raw_path, str):
        return None

    raw_path = raw_path.strip()
    if raw_path.startswith("gs://"):
        return raw_path

    if raw_path.startswith("/"):
        raw_path = raw_path.lstrip("/")

    if raw_path.startswith("cropped_objects/"):
        stripped = raw_path[len("cropped_objects/"):]
        parts = stripped.split("/")
        if len(parts) >= 2:
            run = parts[0]
            frame_dir = parts[1]
            frame_name = frame_dir if frame_dir.endswith(".jpg") else f"{frame_dir}.jpg"
            return f"gs://{GCS_BUCKET}/{run}/{frame_name}"

    if GCS_BUCKET:
        return f"gs://{GCS_BUCKET}/{raw_path}"

    return None


def object_image_key(obj_doc: dict) -> str | None:
    if parent_image := obj_doc.get("parent_image"):
        if isinstance(parent_image, str):
            return os.path.basename(parent_image)

    for field in ["path", "crop_path", "img_path"]:
        uri = obj_doc.get(field)
        if isinstance(uri, str) and uri:
            gcs_uri = resolve_gcs_uri(uri)
            if gcs_uri:
                return gcs_uri
            return os.path.basename(uri)

    return None


def download_public_gcs_image(gs_uri: str) -> bytes | None:
    if not gs_uri.startswith("gs://"):
        return None

    bucket_blob = gs_uri[len("gs://"):]
    if "/" not in bucket_blob:
        return None

    bucket, blob_name = bucket_blob.split("/", 1)
    urls = [
        f"https://storage.googleapis.com/{bucket}/{blob_name}",
        f"https://storage.cloud.google.com/{bucket}/{blob_name}",
    ]

    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                return response.read()
        except (HTTPError, URLError, ValueError):
            continue
    return None


fs_client = get_clients()

@st.cache_data(ttl=60)
def get_runs():
    runs = set()
    for collection_name in ["gaze_events", "rag_object_collection"]:
        docs = fs_client.collection(collection_name).select(["run_id"]).stream()
        for doc in docs:
            data = doc.to_dict()
            if "run_id" in data:
                runs.add(data["run_id"])
    return sorted(list(runs), reverse=True)

runs = get_runs()

if not runs:
    st.info("No runs found in Firestore. Try recording some events first!")
    st.stop()

selected_run = st.selectbox("Select Run", runs)

if selected_run:
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.subheader(f"Events for {selected_run}")
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("🗑️ Delete Run", type="primary", use_container_width=True):
            with st.spinner("Deleting run from Firestore and GCS..."):
                # 1. Fetch and delete all Firestore docs
                docs = fs_client.collection("gaze_events").where("run_id", "==", selected_run).stream()
                batch = fs_client.batch()
                count = 0
                
                # Also collect GCS bucket paths to delete
                blobs_to_delete = []
                for doc in docs:
                    data = doc.to_dict()
                    img_uri = data.get("img_path", "")
                    if img_uri.startswith("gs://"):
                        parts = img_uri.replace("gs://", "").split("/", 1)
                        if len(parts) == 2:
                            blobs_to_delete.append((parts[0], parts[1]))
                    batch.delete(doc.reference)
                    count += 1
                    if count >= 490: # Firestore batch limit
                        batch.commit()
                        batch = fs_client.batch()
                        count = 0
                if count > 0:
                    batch.commit()
                    
                # 2. Delete GCS blobs (skipped when using public access only)
                for bucket_name, blob_name in blobs_to_delete:
                    logger.info("Skipping deletion of gs://%s/%s because public GCS access is used.", bucket_name, blob_name)
                
                # 3. Clear cache and reload
                get_runs.clear()
            st.rerun()
            
    def display_events():
        # Fetch events for this run
        docs = fs_client.collection("gaze_events").where("run_id", "==", selected_run).stream()
        
        # Fetch tracked objects for this run from both collections
        objects_by_image = {}
        for collection_name in ["object_collection", "rag_object_collection"]:
            obj_docs = fs_client.collection(collection_name).where("run_id", "==", selected_run).stream()
            for doc in obj_docs:
                data = doc.to_dict()
                key = object_image_key(data)
                if not key:
                    continue
                objects_by_image.setdefault(key, []).append(data)
                if key.startswith("gs://"):
                    basename = os.path.basename(key)
                    objects_by_image.setdefault(basename, []).append(data)
        
        events = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            events.append(data)
            
        if not events:
            st.write("No events found.")
        else:
            # Sort locally to avoid needing Firestore composite indexes
            df = pd.DataFrame(events)
            df = df.sort_values("timestamp")
            events = df.to_dict('records')
            
            for event in events:
                with st.container(border=True):
                    col1, col2 = st.columns([1, 4])
                    img_uri = event.get("img_path", "")
                    frame_objects = []
                    resolved_img_uri = resolve_gcs_uri(img_uri) or img_uri
                    
                    with col1:
                        if resolved_img_uri.startswith("gs://"):
                            img_bytes = download_public_gcs_image(resolved_img_uri)
                            if img_bytes is not None:
                                try:
                                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    draw = ImageDraw.Draw(img)
                                except Exception as e:
                                    st.error(f"Failed to open image bytes: {e}")
                                    img = None
                                    draw = None
                            else:
                                img = None
                                draw = None

                            if img is not None and draw is not None:

                                    frame_objects = objects_by_image.get(resolved_img_uri, [])
                                    if not frame_objects:
                                        basename = os.path.basename(resolved_img_uri)
                                        frame_objects = objects_by_image.get(basename, [])

                                    for obj in frame_objects:
                                        bboxes = obj.get("bounding_boxes", [])
                                        uid = obj.get("object_id", "Unknown")
                                        name = obj.get("object_name", "")
                                        is_gaze_target = obj.get("is_gaze_target", False)
                                        color = "red" if is_gaze_target else "lime"

                                        for bbox in bboxes:
                                            raw_x1 = bbox.get("x1", 0)
                                            raw_y1 = bbox.get("y1", 0)
                                            raw_x2 = bbox.get("x2", 0)
                                            raw_y2 = bbox.get("y2", 0)

                                            if all(isinstance(v, (int, float)) for v in (raw_x1, raw_y1, raw_x2, raw_y2)):
                                                if 0 <= raw_x1 <= 1 and 0 <= raw_x2 <= 1:
                                                    x1 = int(raw_x1 * img.width)
                                                    x2 = int(raw_x2 * img.width)
                                                elif 0 <= raw_x1 <= 1000 and 0 <= raw_x2 <= 1000:
                                                    x1 = int(raw_x1 * img.width / 1000)
                                                    x2 = int(raw_x2 * img.width / 1000)
                                                else:
                                                    x1 = int(raw_x1)
                                                    x2 = int(raw_x2)

                                                if 0 <= raw_y1 <= 1 and 0 <= raw_y2 <= 1:
                                                    y1 = int(raw_y1 * img.height)
                                                    y2 = int(raw_y2 * img.height)
                                                elif 0 <= raw_y1 <= 1000 and 0 <= raw_y2 <= 1000:
                                                    y1 = int(raw_y1 * img.height / 1000)
                                                    y2 = int(raw_y2 * img.height / 1000)
                                                else:
                                                    y1 = int(raw_y1)
                                                    y2 = int(raw_y2)
                                            else:
                                                continue

                                            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                                            draw.text((x1, max(0, y1 - 15)), f"{uid} | {name}", fill=color)

                                    st.image(img, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Failed to load image: {e}")
                    
                    with col2:
                        st.subheader(f"Time: {event.get('timestamp', 0):.2f}s")
                        st.write(f"**Depth:** {event.get('depth', 0):.2f}")
                        st.write(f"**GCS Path:** `{img_uri}`")
                        
                        llm_analysis = event.get("llm_analysis")
                        if llm_analysis:
                            if "error" in llm_analysis:
                                st.error(f"Analysis Error: {llm_analysis['error']}")
                            else:
                                scene = llm_analysis.get("scene_meta", {})
                                objects = llm_analysis.get("objects", [])
                                
                                st.markdown("---")
                                st.markdown(f"**Scene:** {scene.get('description', 'N/A')}")
                                st.caption(f"Environment: {scene.get('environment', 'N/A')} | Lighting: {scene.get('lighting', 'N/A')}")
                                
                                if objects:
                                    gaze_target = next((obj for obj in objects if obj.get("is_gaze_target")), None)
                                    if gaze_target:
                                        st.success(f"🎯 **Gaze Target:** {gaze_target.get('object_name')} - {gaze_target.get('object_description')}")
                                    
                                    with st.expander(f"View {len(frame_objects)} Tracked Objects"):
                                        if frame_objects:
                                            display_objs = []
                                            for obj in frame_objects:
                                                display_objs.append({
                                                    "UID": obj.get("object_id"),
                                                    "Name": obj.get("object_name"),
                                                    "Target?": obj.get("is_gaze_target"),
                                                    "Description": obj.get("object_description")
                                                })
                                            st.dataframe(pd.DataFrame(display_objs), use_container_width=True)
                                        
                                with st.expander("🐞 Debugger: Raw LLM Output"):
                                    st.json(llm_analysis)
                        
    display_events()
