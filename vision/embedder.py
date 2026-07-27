import os
import requests
from pathlib import Path


class DinoEmbedder:

    def __init__(
        self,
        base_url: str = "https://projectaria.hubscope.systems",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("DINOV2_API_KEY", "")
        
        if not self.api_key:
            print("Warning: DINOV2_API_KEY not found in .env. Microservice calls may fail.")

    def embed_batch(
        self,
        image_paths: list[str | Path],
    ) -> list[list[float]]:
        if not image_paths:
            return []
            
        endpoint = f"{self.base_url}/embed"
        headers = {
            "X-API-Key": self.api_key
        }
        
        # Prepare multipart/form-data for files
        files_data = []
        for path in image_paths:
            # We open the files and let requests close them after sending
            files_data.append(
                ("files", (os.path.basename(path), open(path, "rb"), "image/jpeg"))
            )
            
        try:
            response = requests.post(endpoint, headers=headers, files=files_data)
            response.raise_for_status()
            
            data = response.json()
            # The API returns {"embeddings": [{"filename": "...", "embedding": [...]}]}
            # We need to extract just the embeddings in the same order
            embeddings = [item["embedding"] for item in data.get("embeddings", [])]
            return embeddings
            
        except Exception as e:
            print(f"Error calling DinoV2 microservice: {e}")
            # Return empty embeddings for fault tolerance (or raise, depending on design)
            raise
        finally:
            # Clean up open file handles
            for _, file_tuple in files_data:
                file_tuple[1].close()