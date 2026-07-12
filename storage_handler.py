import os
from google.cloud import storage, firestore

class CloudStorageHandler:
    def __init__(self, gcp_project=None, gcs_bucket=None):
        self.gcp_project = gcp_project
        self.gcs_bucket = gcs_bucket
        
        if not self.gcp_project or not self.gcs_bucket:
            raise ValueError("GCP Project ID and GCS Bucket must be provided for cloud storage")
        
        print(f"Initializing GCP clients for project {self.gcp_project}...")
        self.fs_client = firestore.Client(project=self.gcp_project)
        self.storage_client = storage.Client(project=self.gcp_project)
            
    def save_event(self, timestamp: float, depth: float, img_path: str, run_id: str = None) -> str:
        log_str = ""
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
            doc_ref.set(doc_data)
            log_str += "   ✅ Successfully saved to Firestore\n"
        except Exception as e:
            log_str += f"   ❌ Cloud Error: {e}\n"
        
        return log_str
