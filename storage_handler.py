import os
import requests
from google.cloud import storage, firestore

class StorageHandler:
    def __init__(self, mode="local", gcp_project=None, gcs_bucket=None):
        self.mode = mode
        self.gcp_project = gcp_project
        self.gcs_bucket = gcs_bucket
        
        self.fs_client = None
        self.storage_client = None
        
        if self.mode == "cloud":
            if not self.gcp_project or not self.gcs_bucket:
                raise ValueError("GCP Project ID and GCS Bucket must be provided for cloud storage")
            
            print(f"Initializing GCP clients for project {self.gcp_project}...")
            self.fs_client = firestore.Client(project=self.gcp_project)
            self.storage_client = storage.Client(project=self.gcp_project)
            
    def save_event(self, timestamp: float, depth: float, img_path: str, run_id: str = None, llm_analysis: dict = None) -> str:
        log_str = ""
        
        if self.mode == "cloud":
            try:
                # 1. Upload to GCS
                bucket = self.storage_client.bucket(self.gcs_bucket)
                blob_name = f"{run_id}/{os.path.basename(img_path)}" if run_id else os.path.basename(img_path)
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(img_path)
                img_uri = f"gs://{self.gcs_bucket}/{blob_name}"
                log_str += f"   ☁️  Uploaded to GCS: {img_uri}\n"
                
                # 2. Save to Firestore
                doc_ref = self.fs_client.collection("gaze_events").document()
                doc_data = {
                    "timestamp": timestamp,
                    "depth": depth,
                    "img_path": img_uri
                }
                if run_id:
                    doc_data["run_id"] = run_id
                if llm_analysis:
                    doc_data["llm_analysis"] = llm_analysis
                    
                doc_ref.set(doc_data)
                log_str += "   ✅ Successfully saved to Firestore\n"
            except Exception as e:
                log_str += f"   ❌ Cloud Error: {e}\n"
                
        elif self.mode == "local":
            try:
                payload = {
                    "timestamp": timestamp,
                    "depth": depth,
                    "img_path": img_path
                }
                if llm_analysis:
                    payload["llm_analysis"] = llm_analysis
                    
                response = requests.post(
                    "http://127.0.0.1:8000/insert",
                    json=payload,
                    timeout=2.0
                )
                if response.status_code == 200:
                    log_str += "   ✅ Successfully saved to local database\n"
                else:
                    log_str += f"   ❌ DB Error: {response.text}\n"
            except requests.exceptions.RequestException as e:
                log_str += f"   ❌ Failed to connect to local DB: {e}\n"
                
        return log_str
