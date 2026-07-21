import os
import json
import logging
import base64
from io import BytesIO
import time
from PIL import Image

from google import genai
from google.cloud import storage, firestore
from dotenv import load_dotenv

# Load environment variables if present (e.g. for GCP_PROJECT)
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

GCP_PROJECT = os.environ.get("GCP_PROJECT", "project-aria-501223")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Clients
db = firestore.Client(project=GCP_PROJECT)
storage_client = storage.Client(project=GCP_PROJECT)
gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def analyze_image(image: Image.Image) -> dict:
    """Sends the image to Gemini and returns a JSON dictionary of the analysis."""
    prompt = """
    Analyze the image. The red crosshair indicates the user's gaze vector.
    
    You MUST return exactly one valid JSON object with meta information about the overall scene, and an itemized list of objects found in the image. Format it exactly like this:
    {
      "scene_meta": {
        "description": "General description of the overall scene",
        "environment": "Indoors, outdoors, office, kitchen, etc.",
        "lighting": "Bright, dim, natural, artificial, etc."
      },
      "objects": [
        {
          "object_name": "Name of the object",
          "object_description": "Detailed description of the object",
          "object_location": "Contextual location of the object in the image",
          "is_gaze_target": true // true if this object is under the red crosshair, false otherwise
        }
      ]
    }
    """
    try:
        response = gemini.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[prompt, image]
        )
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        logger.error("Error analyzing image with Gemini: %s", e)
        return {"error": str(e)}

def main():
    logger.info("Starting one-time backfill of older Firestore records...")
    
    # Get all documents in the gaze_events collection
    events_ref = db.collection("gaze_events")
    docs = events_ref.stream()
    
    processed_count = 0
    skipped_count = 0
    
    for doc in docs:
        data = doc.to_dict()
        
        # Check if already processed
        if "llm_analysis" in data:
            logger.info("Skipping document %s (already has llm_analysis)", doc.id)
            skipped_count += 1
            continue
            
        img_uri = data.get("img_path")
        if not img_uri or not img_uri.startswith("gs://"):
            logger.warning("Document %s has invalid or missing img_path: %s", doc.id, img_uri)
            skipped_count += 1
            continue
            
        logger.info("Processing document %s with img_path: %s", doc.id, img_uri)
        
        # Parse bucket and blob name from gs://<bucket>/<blob>
        path_parts = img_uri.replace("gs://", "").split("/", 1)
        if len(path_parts) != 2:
            logger.error("Could not parse gs URI: %s", img_uri)
            continue
            
        bucket_name, blob_name = path_parts
        
        # Download image
        try:
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if not blob.exists():
                logger.error("Blob %s does not exist in bucket %s", blob_name, bucket_name)
                skipped_count += 1
                continue
                
            image_bytes = blob.download_as_bytes()
            image = Image.open(BytesIO(image_bytes))
        except Exception as e:
            logger.error("Failed to download image %s: %s", img_uri, e)
            continue
            
        # Analyze image
        analysis_result = analyze_image(image)
        if "error" in analysis_result:
            logger.error("Skipping update for %s due to analysis error.", doc.id)
            continue
            
        # Update document
        try:
            doc.reference.update({
                "llm_analysis": analysis_result
            })
            logger.info("Successfully updated %s", doc.id)
            processed_count += 1
        except Exception as e:
            logger.error("Failed to update document %s: %s", doc.id, e)
            
        # Give Gemini a breather to avoid 503 Overloaded errors
        time.sleep(2)
        
    logger.info("Backfill complete! Processed: %d, Skipped: %d", processed_count, skipped_count)

if __name__ == "__main__":
    main()
