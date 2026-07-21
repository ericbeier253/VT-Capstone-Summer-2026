import streamlit as st
from google.cloud import storage, firestore
import os
import pandas as pd

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
                                    st.image(img_bytes, use_container_width=True)
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
                                    
                                    with st.expander(f"View all {len(objects)} detected objects"):
                                        st.dataframe(pd.DataFrame(objects), use_container_width=True)
                                        
                                with st.expander("🐞 Debugger: Raw LLM Output"):
                                    st.json(llm_analysis)
                        
    display_events()
