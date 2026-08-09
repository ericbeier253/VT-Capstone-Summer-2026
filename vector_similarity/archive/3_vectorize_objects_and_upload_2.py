import os
from pathlib import Path

import torch
from PIL import Image

from transformers import AutoImageProcessor
from transformers import AutoModel

from google.cloud import firestore
from google.oauth2 import service_account
from google.cloud.firestore_v1.vector import Vector

import torch.nn.functional as F

from pathlib import Path
from pydantic import BaseModel, EmailStr
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import sys

root_dir = Path(__file__).resolve().parents[1]

# Insert the path into sys.path so Python can see it
sys.path.insert(0, str(root_dir))

from vision.schemas import ImageAnalysis

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

# Load .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val.strip().strip('"').strip("'")
                #print(key, ":", val.strip().strip('"').strip("'"))

ROOT_FOLDER = "cropped_objects"

PROJECT_ID = os.environ["GCP_PROJECT"] #"Project Aria"

SERVICE_ACCOUNT = ".secrets/aria-uploader-key.json"

COLLECTION = "rag_object_collection"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------
# Firestore
# -------------------------------------------------------

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT
)

db = firestore.Client(
    project=PROJECT_ID,
    credentials=credentials
)


# -------------------------------------------------------
# DINOv2
# -------------------------------------------------------

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
).to(DEVICE)

model.eval()


# -------------------------------------------------------
# Embedding
# -------------------------------------------------------

@torch.no_grad()
def embed_image(path: Path):

    image = Image.open(path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    #embedding = (
    #    outputs.last_hidden_state[:, 0]
    #    .squeeze()
    #    .cpu()
    #    .numpy()
    #)

    embedding = outputs.last_hidden_state[:, 0]

    # L2 normalize
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    embedding = (
        embedding.squeeze()
        .cpu()
        .numpy()
    )

    return embedding.tolist()


# -------------------------------------------------------
# Upload
# -------------------------------------------------------

def upload_embedding(image_path: Path):

    embedding = embed_image(image_path)

    run_name = image_path.parent.parent.name

    image_folder = image_path.parent.name

    object_name = image_path.stem

    json_dict = read_json_to_dict(r'results/'+run_name+'/'+image_folder+'.json')

    #doc = {

    #    "run": run_name,

    #    "folder": image_folder,

    #    "image_name": image_path.name,

    #    "object_name": object_name,

    #    "path": str(image_path),

    #    "embedding": embedding

    #}

    #db.collection(COLLECTION).document().set(doc)

    #doc = {
    #    "run": run_name,
    #    "folder": image_folder,
    #    "image_name": image_path.name,
    #    "object_name": object_name,
    #    "path": str(image_path),

    #    # Store as a Firestore Vector
    #    "embedding": Vector(embedding),
    #}

    num = int(object_name.split('_')[-1])

    object_name_2 = " ".join(object_name.split('_')[:-1])

    for obj in json_dict["objects"]:

        for bb in obj["bounding_boxes"]:

            if obj["object_name"]!=object_name_2:
                continue
            elif obj["object_name"]==object_name_2 and num!=0: 
                num-=1
                continue

            doc = {
                    "bounding_boxes": bb,
                    "crop_path": str(image_path),
                    "embedding": Vector(embedding),
                    "is_gaze_target": obj["is_gaze_target"],
                    "object_description": obj["object_description"],
                    #"object_id": obj["object_id"],
                    "object_location": obj["object_location"],
                    "object_name": obj["object_name"],
                    "parent_image": image_path.name,
                    "run_id": run_name,
                    "timestamp": extract_timestamp(run_name),
                    "scene_meta":json_dict["scene_meta"]
                }
            break

    print(run_name)
    print(image_path.name)
    print(object_name)
    print(num)
    print(doc)

    db.collection(COLLECTION).document().set(doc)

    #print(image_path)


# -------------------------------------------------------
# Load JSON
# -------------------------------------------------------

def read_json_to_dict(filepath: str) -> dict:
    json_file_path = Path(filepath)

    try:
        # Read raw JSON data from the file
        raw_json_data = json_file_path.read_text()
        
        # Parse and validate the JSON string directly using Pydantic
        validated_model = ImageAnalysis.model_validate_json(raw_json_data)
        
        # Export the validated data into a standard Python dictionary
        validated_dict = validated_model.model_dump()
        
        #print(validated_dict)
        #print(type(validated_dict))  # <class 'dict'>

    except Exception as e:
        print(f"Validation or File error: {e}")

    return validated_dict

# -------------------------------------------------------
# Support FUnctions: Timestamp
# -------------------------------------------------------

def extract_timestamp(run_id: Optional[str]) -> Optional[datetime]:
    """
    Extract a timestamp from an old run ID.

    Example:

        run_20260715_102510

    becomes:

        2026-07-15 10:25:10 UTC
    """

    if not run_id:
        return None

    pattern = r"^run_(\d{8})_(\d{6})$"

    match = re.match(pattern, run_id)

    if not match:
        print(
            f"WARNING: Could not extract timestamp from run ID: {run_id}"
        )
        return None

    date_part = match.group(1)
    time_part = match.group(2)

    try:
        return datetime.strptime(
            date_part + time_part,
            "%Y%m%d%H%M%S",
        ).replace(tzinfo=timezone.utc)

    except ValueError:
        print(
            f"WARNING: Invalid timestamp in run ID: {run_id}"
        )
        return None

# -------------------------------------------------------
# Main
# -------------------------------------------------------

for path in Path(ROOT_FOLDER).rglob("*"):

    if path.suffix.lower() in {

        ".jpg",
        #".jpeg",
        #".png",
        #".bmp",
        #".webp"

    }:

        #print(path)
        upload_embedding(path)