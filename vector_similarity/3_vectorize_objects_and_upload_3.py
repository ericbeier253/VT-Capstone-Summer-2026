import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from google import genai
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.oauth2 import service_account

# Project-local imports
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from vision.schemas import ImageAnalysis
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_FOLDER = ROOT_DIR / "cropped_objects"
RESULTS_FOLDER = ROOT_DIR / "results"

SERVICE_ACCOUNT = ROOT_DIR / ".secrets" / "aria-uploader-key.json"

COLLECTION = "rag_object_collection_2"

IMAGE_MODEL_NAME = "facebook/dinov2-base"
TEXT_MODEL_NAME = "gemini-embedding-2" #"text-embedding-004"

LOCATION = "us-central1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CREDENTIALS
# ============================================================

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT,
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform"
    ],
)

PROJECT_ID = credentials.project_id

# Load .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val.strip().strip('"').strip("'")
                #print(key, ":", val.strip().strip('"').strip("'"))


# ============================================================
# FIRESTORE
# ============================================================

db = firestore.Client(
    project=PROJECT_ID,
    credentials=credentials,
)


# ============================================================
# GEMINI / VERTEX AI
# ============================================================

#genai_client = genai.Client(
#    vertexai=True,
#    project=PROJECT_ID,
#    location=LOCATION,
#    credentials=credentials,
#)

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


# ============================================================
# DINOv2
# ============================================================

print("Loading DINOv2...")

image_processor = AutoImageProcessor.from_pretrained(
    IMAGE_MODEL_NAME
)

image_model = AutoModel.from_pretrained(
    IMAGE_MODEL_NAME
).to(DEVICE)

image_model.eval()

print(f"DINOv2 device: {DEVICE}")


# ============================================================
# IMAGE EMBEDDING
# ============================================================

@torch.no_grad()
def embed_image(path: Path):
    """
    Generate a normalized 768-dimensional DINOv2
    image embedding.
    """

    image = Image.open(path).convert("RGB")

    inputs = image_processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = image_model(**inputs)

    embedding = outputs.last_hidden_state[:, 0]

    # L2 normalization
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    embedding = (
        embedding
        .squeeze()
        .cpu()
        .numpy()
        .tolist()
    )

    return embedding


# ============================================================
# TEXT EMBEDDING
# ============================================================

def embed_text(
    object_name: str,
    object_description: str,
):
    """
    Generate a Gemini text embedding from the object's
    name and description.

    Example input:

        object_name:
            "pen"

        object_description:
            "A black pen resting inside a desk organizer tray"

    The resulting embedding is stored separately from the
    DINOv2 image embedding.
    """

    text = (
        f"Object name: {object_name}. "
        f"Object description: {object_description}"
    )

    #response = genai_client.models.embed_content(
    #    model=TEXT_MODEL_NAME,
    #    contents=text,
    #    config={
    #        "task_type": "RETRIEVAL_DOCUMENT",
    #    },
    #)

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config={
            "output_dimensionality": 768
        }
    )

    return response.embeddings[0].values


# ============================================================
# LOAD JSON
# ============================================================

def read_json_to_dict(filepath: Path) -> dict:
    """
    Load and validate the corresponding object-analysis JSON.
    """

    raw_json_data = filepath.read_text(
        encoding="utf-8"
    )

    validated_model = ImageAnalysis.model_validate_json(
        raw_json_data
    )

    return validated_model.model_dump()


# ============================================================
# TIMESTAMP
# ============================================================

def extract_timestamp(
    run_id: Optional[str],
) -> Optional[datetime]:
    """
    Convert:

        run_20260715_102510

    into:

        2026-07-15 10:25:10 UTC
    """

    if not run_id:
        return None

    pattern = r"^run_(\d{8})_(\d{6})$"

    match = re.match(
        pattern,
        run_id,
    )

    if not match:
        print(
            f"WARNING: Could not extract timestamp "
            f"from run ID: {run_id}"
        )
        return None

    date_part = match.group(1)
    time_part = match.group(2)

    try:

        return datetime.strptime(
            date_part + time_part,
            "%Y%m%d%H%M%S",
        ).replace(
            tzinfo=timezone.utc
        )

    except ValueError:

        print(
            f"WARNING: Invalid timestamp "
            f"in run ID: {run_id}"
        )

        return None


# ============================================================
# CREATE DOCUMENT
# ============================================================

def create_document(
    image_path: Path,
):
    """
    Generate both image and text embeddings and construct
    the Firestore document.
    """

    print()
    print("=" * 70)
    print(f"Processing: {image_path}")
    print("=" * 70)

    # --------------------------------------------------------
    # Generate image embedding
    # --------------------------------------------------------

    print("Generating image embedding...")

    image_embedding = embed_image(
        image_path
    )

    print(
        f"Image embedding dimension: "
        f"{len(image_embedding)}"
    )

    # --------------------------------------------------------
    # Extract path metadata
    # --------------------------------------------------------

    run_name = image_path.parent.parent.name

    image_folder = image_path.parent.name

    object_filename = image_path.stem

    # --------------------------------------------------------
    # Load corresponding JSON
    # --------------------------------------------------------

    json_path = (
        RESULTS_FOLDER
        / run_name
        / f"{image_folder}.json"
    )

    if not json_path.exists():

        raise FileNotFoundError(
            f"JSON file not found: {json_path}"
        )

    json_dict = read_json_to_dict(
        json_path
    )

    # --------------------------------------------------------
    # Determine object index
    #
    # Example:
    #
    #   tray_0
    #   tray_1
    #
    # becomes:
    #
    #   object_name_2 = "tray"
    #   num = 0 / 1
    # --------------------------------------------------------

    parts = object_filename.rsplit(
        "_",
        1,
    )

    if len(parts) != 2:

        raise ValueError(
            f"Unexpected object filename: "
            f"{object_filename}"
        )

    object_name_2 = (
        parts[0]
        .replace("_", " ")
    )

    try:

        num = int(parts[1])

    except ValueError:

        raise ValueError(
            f"Could not determine object index "
            f"from filename: {object_filename}"
        )

    # --------------------------------------------------------
    # Find corresponding object in JSON
    # --------------------------------------------------------

    matched_object = None
    matched_bounding_box = None

    for obj in json_dict["objects"]:

        if obj["object_name"] != object_name_2:
            continue

        boxes = obj.get(
            "bounding_boxes",
            [],
        )

        if num >= len(boxes):
            continue

        matched_object = obj
        matched_bounding_box = boxes[num]

        break

    if matched_object is None:

        raise RuntimeError(
            f"Could not match object "
            f"'{object_filename}' "
            f"to JSON metadata in {json_path}"
        )

    # --------------------------------------------------------
    # Object metadata
    # --------------------------------------------------------

    object_name = matched_object[
        "object_name"
    ]

    object_description = matched_object[
        "object_description"
    ]

    object_location = matched_object[
        "object_location"
    ]

    # --------------------------------------------------------
    # Generate text embedding
    # --------------------------------------------------------

    print(
        f"Generating text embedding for: "
        f"{object_name}"
    )

    text_embedding = embed_text(
        object_name=object_name,
        object_description=object_description,
    )

    print(
        f"Text embedding dimension: "
        f"{len(text_embedding)}"
    )

    # --------------------------------------------------------
    # Construct Firestore document
    # --------------------------------------------------------

    time.sleep(1) # So as to not hit the time-limit based quota of 100 every 30 seconds

    doc = {

        # ----------------------------------------------------
        # Spatial/object metadata
        # ----------------------------------------------------

        "bounding_boxes":
            matched_bounding_box,

        "crop_path":
            str(image_path),

        "is_gaze_target":
            matched_object["is_gaze_target"],

        "object_description":
            object_description,

        "object_location":
            object_location,

        "object_name":
            object_name,

        "parent_image":
            image_path.name,

        "run_id":
            run_name,

        "timestamp":
            extract_timestamp(run_name),

        # ----------------------------------------------------
        # Scene metadata
        # ----------------------------------------------------

        "scene_meta":
            json_dict["scene_meta"],

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        # DINOv2 image embedding
        "embedding":
            Vector(image_embedding),

        # Gemini text embedding
        "text_embedding":
            Vector(text_embedding),
    }

    return doc


# ============================================================
# UPLOAD
# ============================================================

def upload_embedding(
    image_path: Path,
):
    """
    Create the document and upload it to Firestore.
    """

    try:

        doc = create_document(
            image_path
        )

        db.collection(
            COLLECTION
        ).document().set(doc)

        print(
            f"Uploaded: {image_path}"
        )

    except Exception as e:

        print(
            f"ERROR processing "
            f"{image_path}: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAG Object Embedding Pipeline")
    print("=" * 70)

    print(f"Project       : {PROJECT_ID}")
    print(f"Collection    : {COLLECTION}")
    print(f"Image model   : {IMAGE_MODEL_NAME}")
    print(f"Text model    : {TEXT_MODEL_NAME}")
    print(f"Device        : {DEVICE}")
    print(f"Root folder   : {ROOT_FOLDER}")

    print("=" * 70)

    image_paths = [
        path
        for path in ROOT_FOLDER.rglob("*")
        if path.suffix.lower() == ".jpg"
    ]

    print(
        f"Found {len(image_paths)} images."
    )

    success_count = 0
    failure_count = 0

    for image_path in image_paths:

        try:

            upload_embedding(
                image_path
            )

            success_count += 1

        except Exception as e:

            failure_count += 1

            print(
                f"Failed: {image_path}"
            )

            print(
                f"Reason: {e}"
            )

    print()
    print("=" * 70)
    print("Completed")
    print("=" * 70)

    print(
        f"Successful: {success_count}"
    )

    print(
        f"Failed:    {failure_count}"
    )


if __name__ == "__main__":
    main()

