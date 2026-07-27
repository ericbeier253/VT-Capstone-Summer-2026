import streamlit as st
from google.cloud import storage, firestore
import os
import io
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
if not gcp_project:
    st.error("GCP_PROJECT not found in .env")
    st.stop()

@st.cache_resource
def get_clients():
    fs = firestore.Client(project=gcp_project)
    cs = storage.Client(project=gcp_project)
    return fs, cs

fs_client, storage_client = get_clients()

@st.cache_data(ttl=60)
def get_runs():
    # Fetch unique runs by scanning existing events
    docs = fs_client.collection("gaze_events").select(["run_id"]).stream()
    runs = set()
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
    col1, col2 = st.columns([8, 1])
    with col1:
        st.subheader(f"Events for {selected_run}")
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
            
    def display_events():
        # Fetch events for this run
        docs = fs_client.collection("gaze_events").where("run_id", "==", selected_run).stream()
        
        # Fetch tracked objects for this run
        obj_docs = fs_client.collection("object_collection").where("run_id", "==", selected_run).stream()
        objects_by_image = {}
        for doc in obj_docs:
            data = doc.to_dict()
            basename = os.path.basename(data.get("parent_image", ""))
            if basename not in objects_by_image:
                objects_by_image[basename] = []
            objects_by_image[basename].append(data)
        
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
                    
                    with col1:
                        if img_uri.startswith("gs://"):
                            parts = img_uri.replace("gs://", "").split("/", 1)
                            if len(parts) == 2:
                                bucket_name, blob_name = parts
                                try:
                                    bucket = storage_client.bucket(bucket_name)
                                    blob = bucket.blob(blob_name)
                                    img_bytes = blob.download_as_bytes()
                                    
                                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    draw = ImageDraw.Draw(img)
                                    
                                    basename = os.path.basename(img_uri)
                                    frame_objects = objects_by_image.get(basename, [])
                                    
                                    for obj in frame_objects:
                                        bboxes = obj.get("bounding_boxes", [])
                                        uid = obj.get("object_id", "Unknown")
                                        name = obj.get("object_name", "")
                                        is_gaze_target = obj.get("is_gaze_target", False)
                                        color = "red" if is_gaze_target else "lime"
                                        
                                        for bbox in bboxes:
                                            # Gemini outputs normalized coords in [0, 1000] range
                                            raw_x1 = bbox.get("x1", 0)
                                            raw_y1 = bbox.get("y1", 0)
                                            raw_x2 = bbox.get("x2", 0)
                                            raw_y2 = bbox.get("y2", 0)
                                            
                                            # Scale to actual image pixel dimensions
                                            x1 = int(img.width * raw_x1 / 1000)
                                            y1 = int(img.height * raw_y1 / 1000)
                                            x2 = int(img.width * raw_x2 / 1000)
                                            y2 = int(img.height * raw_y2 / 1000)
                                            
                                            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                                            draw.text((x1, max(0, y1-15)), f"{uid} | {name}", fill=color)
                                    
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
