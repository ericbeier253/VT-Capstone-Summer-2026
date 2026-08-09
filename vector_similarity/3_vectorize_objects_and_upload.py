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

COLLECTION = "object_collection"

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

    #doc = {

    #    "run": run_name,

    #    "folder": image_folder,

    #    "image_name": image_path.name,

    #    "object_name": object_name,

    #    "path": str(image_path),

    #    "embedding": embedding

    #}

    #db.collection(COLLECTION).document().set(doc)

    doc = {
        "run": run_name,
        "folder": image_folder,
        "image_name": image_path.name,
        "object_name": object_name,
        "path": str(image_path),

        # Store as a Firestore Vector
        "embedding": Vector(embedding),
    }

    db.collection(COLLECTION).document().set(doc)

    print(image_path)


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