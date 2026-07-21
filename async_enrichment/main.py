import base64
import json
import logging
import os
import functions_framework
from io import BytesIO
from PIL import Image

from google import genai
from google.cloud import storage, firestore

# Configuration (from environment variables or hardcoded for now)
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
        
        # Clean up response text to ensure we extract just the JSON
        text = response.text.replace("```json", "").replace("```", "").strip()
        logger.info("Gemini raw response: %s", text)
        return json.loads(text)
    except Exception as e:
        logger.error("Error analyzing image: %s", e)
        return {"error": str(e)}

@functions_framework.cloud_event
def initial_enrichment(cloud_event):
    """Triggered by a Pub/Sub message from a GCS object creation event."""
    
    try:
        # Decode the Pub/Sub message payload
        payload = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
        event_data = json.loads(payload)
    except Exception as e:
        logger.error("Failed to decode Pub/Sub message: %s", e)
        return

    # Extract bucket and filename
    bucket_name = event_data.get("bucket")
    blob_name = event_data.get("name")
    
    if not bucket_name or not blob_name:
        logger.error("Missing bucket or blob name in event data: %s", event_data)
        return
        
    img_uri = f"gs://{bucket_name}/{blob_name}"
    logger.info("Processing new image upload: %s", img_uri)
    
    # 1. Query Firestore for the document created by the headset
    # The headset script saves the image path as 'img_path'
    docs = db.collection("gaze_events").where("img_path", "==", img_uri).limit(1).stream()
    
    doc_ref = None
    for doc in docs:
        doc_ref = doc.reference
        break
        
    if not doc_ref:
        logger.warning("Could not find a Firestore document in 'gaze_events' matching img_path: %s", img_uri)
        # Even if we don't find it immediately, we could process and save it, 
        # but since we want to join, we'll abort.
        return
        
    logger.info("Found matching Firestore document: %s", doc_ref.id)
    
    # 2. Download the image from GCS
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        logger.error("Failed to download image %s: %s", img_uri, e)
        return

    # 3. Analyze the image with Gemini
    analysis_result = analyze_image(image)
    
    if "error" in analysis_result:
        logger.error("Skipping update due to analysis error.")
        return

    # 4. Append the analysis to the existing Firestore document
    try:
        doc_ref.update({
            "llm_analysis": analysis_result
        })
        logger.info("Successfully appended LLM analysis to document %s", doc_ref.id)
    except Exception as e:
        logger.error("Failed to update Firestore document %s: %s", doc_ref.id, e)

