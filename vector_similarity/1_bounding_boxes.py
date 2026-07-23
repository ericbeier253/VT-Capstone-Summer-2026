import os
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import storage
from google import genai
from google.genai import types
from google.genai.types import Part

import time

from pydantic import BaseModel, Field, model_validator
from typing import List

# ==========================================================
# SCHEMA
# ==========================================================

class BoundingBox(BaseModel):
    x1: int = Field(description="Left coordinate")
    y1: int = Field(description="Top coordinate")
    x2: int = Field(description="Right coordinate")
    y2: int = Field(description="Bottom coordinate")

    @model_validator(mode="after")
    def validate_box(self):
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError("Bounding box coordinates must be non-negative.")

        if self.x1 >= self.x2:
            raise ValueError("Bounding box must satisfy x1 < x2.")

        if self.y1 >= self.y2:
            raise ValueError("Bounding box must satisfy y1 < y2.")

        return self

class SceneMeta(BaseModel):
    description: str
    environment: str
    lighting: str


class DetectedObject(BaseModel):
    object_name: str
    object_description: str
    object_location: str

    # List of bounding boxes, each is [x1, y1, x2, y2]
    bounding_boxes: List[BoundingBox] #List[List[int]] = Field(
    #    description="Bounding boxes in [x1, y1, x2, y2] format."
    #)

    is_gaze_target: bool


class ImageAnalysis(BaseModel):
    scene_meta: SceneMeta
    objects: List[DetectedObject]

# ==========================================================
# CONFIGURATION
# ==========================================================

# Load .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val.strip().strip('"').strip("'")
                #print(key, ":", val.strip().strip('"').strip("'"))

PROJECT_ID = "Project Aria"

LOCATION = "us-central1"

BUCKET_NAME = "project-aria-gaze-photos-eb-01"

OUTPUT_DIR = "./results"

MAX_WORKERS = 1 # More than 1 leads to resouce exhaustion

MODEL_NAME = "gemini-3.5-flash-lite"

#PROMPT = """
#Analyze the image. The red crosshair indicates the user's gaze vector.
#
#You MUST return exactly one valid JSON object with meta information about the overall scene, and an itemized list of objects found in the image.
#
#Format exactly as:
#
#{
#  "scene_meta": {
#    "description": "General description of the overall scene",
#    "environment": "Indoors, outdoors, office, kitchen, etc.",
#    "lighting": "Bright, dim, natural, artificial, etc."
#  },
#  "objects": [
#    {
#      "object_name": "Name of the object",
#      "object_description": "Detailed description of the object",
#      "object_location": "Contextual location of the object in the image",
#      "bounding_boxes": "A list of integers denoting the top-left and bottom-right corner points of the bounding box for the object",
#      "is_gaze_target": true
#    }
#  ]
#}
#"""

PROMPT = """
Analyze the image.

The red crosshair indicates the user's gaze.

Detect all visible objects.

For each object provide:
- name
- description
- location
- bounding box
- whether it is the gaze target
"""

# ==========================================================
# CLIENTS
# ==========================================================

#storage_client = storage.Client()

from google.oauth2 import service_account
from google.cloud import storage

#credentials = service_account.Credentials.from_service_account_file(
#    ".secrets/aria-uploader-key.json"
#)

from google.oauth2 import service_account

credentials = (
    service_account.Credentials
    .from_service_account_file(
        ".secrets/aria-uploader-key.json",
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform"
        ]
    )
)

storage_client = storage.Client(
    project=PROJECT_ID,
    credentials=credentials
)

#genai_client = genai.Client(
#    vertexai=True,
#    project=PROJECT_ID,
#    location=LOCATION
#)

from google import genai

#genai_client = genai.Client(
#    vertexai=True,
#    project=PROJECT_ID,
#    location=LOCATION,
#    credentials=credentials
#)

api_key = os.environ.get("GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key) if api_key else None
if not api_key:
    print("Warning: GEMINI_API_KEY not found in .env. LLM enrichment will be disabled.")

# ==========================================================
# HELPERS
# ==========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def is_image(blob_name: str) -> bool:
    return Path(blob_name).suffix.lower() in IMAGE_EXTENSIONS


def analyze_gcs_image(blob: str) -> ImageAnalysis:#dict: #gcs_uri: str) -> dict:

    image_bytes = blob.download_as_bytes()

    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            PROMPT,
            Part.from_bytes(#from_uri(
                #file_uri=gcs_uri,
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=ImageAnalysis,
        )
    )

    #return json.loads(response.text)
    return response.parsed


def save_result(blob_name: str, result: dict):

    output_path = (
        Path(OUTPUT_DIR)
        / Path(blob_name).with_suffix(".json")
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            #result,
            result.model_dump(),
            f,
            indent=2,
            ensure_ascii=False
        )


def process_blob(blob):

    try:

        #gcs_uri = f"gs://{BUCKET_NAME}/{blob.name}"

        #result = analyze_gcs_image(blob)#gcs_uri)

        while True:
            try:
                time.sleep(1)
                result = analyze_gcs_image(blob)
                #print("Function succeeded!")
                break  # Exit the loop on success
            except Exception as e:
                print(f"Caught error: {e}. Retrying in 2 seconds...")
                time.sleep(1)  # Wait before trying again

        save_result(blob.name, result)

        logging.info(
            "Processed: %s",
            blob.name
        )

        return True

    except Exception as e:

        logging.exception(
            "Failed: %s",
            blob.name
        )

        return False


# ==========================================================
# MAIN
# ==========================================================

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    bucket = storage_client.bucket(BUCKET_NAME)

    image_blobs = [
        blob
        for blob in bucket.list_blobs()
        if is_image(blob.name)
    ]

    logging.info(
        "Found %d images",
        len(image_blobs)
    )

    success_count = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_blob,
                blob
            )
            for blob in image_blobs#[:1] # Only processing 1 image right now
        ]

        for future in as_completed(futures):

            if future.result():
                success_count += 1

    logging.info(
        "Completed %d/%d images",
        success_count,
        len(image_blobs)
    )


if __name__ == "__main__":
    main()

