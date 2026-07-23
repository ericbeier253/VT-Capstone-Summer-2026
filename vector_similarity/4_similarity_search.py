# PRIOR TO RUNNING THIS, THE FIRESTORE COLLECTION CONTAINING THE EMBEDDINGS MUST BE INDEXED

import argparse
import os

import torch
from PIL import Image

from transformers import AutoImageProcessor, AutoModel

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.oauth2 import service_account


# ============================================================
# CONFIG
# ============================================================

# Load .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val.strip().strip('"').strip("'")
                #print(key, ":", val.strip().strip('"').strip("'"))

PROJECT_ID = os.environ["GCP_PROJECT"] #"Project Aria"

SERVICE_ACCOUNT = ".secrets/aria-uploader-key.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "facebook/dinov2-base"

TOP_K = 5

COLLECTION = "obj_embeddings"

VECTOR_FIELD = "embedding"

THRESHOLD = 0.25


# ============================================================
# FIRESTORE
# ============================================================

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT
)

db = firestore.Client(
    project=PROJECT_ID,
    credentials=credentials
)


# ============================================================
# LOAD DINOv2
# ============================================================

print("Loading DINOv2...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

print(f"Using device: {DEVICE}")


# ============================================================
# EMBEDDING
# ============================================================

@torch.no_grad()
def generate_embedding(image_path: str):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    embedding = (
        outputs.last_hidden_state[:, 0]
        .squeeze()
        .cpu()
        .numpy()
    )

    return embedding.tolist()


# ============================================================
# SEARCH
# ============================================================

def search_similar_images(query_vector):

    collection_ref = db.collection(COLLECTION)

    results = (
        collection_ref.find_nearest(
            vector_field=VECTOR_FIELD,
            query_vector=Vector(query_vector),
            distance_measure=DistanceMeasure.COSINE, 
            distance_threshold=THRESHOLD,
            limit=TOP_K,
        )
        .get()
    )

    print("\nTop Matches\n")

    for rank, doc in enumerate(results, start=1):

        data = doc.to_dict()

        print(f"{rank}. Document ID : {doc.id}")

        print(f"   Object       : {data.get('object_name')}")

        print(f"   Image        : {data.get('image_name')}")

        print(f"   Folder       : {data.get('folder')}")

        print(f"   Run          : {data.get('run_name')}")

        if "path" in data:
            print(f"   Path         : {data['path']}")

        # Firestore returns the distance as a special field
        if "__distance__" in data:
            print(f"   Distance     : {data['__distance__']:.6f}")

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Path to object image"
    )

    args = parser.parse_args()

    print("Generating embedding...")

    embedding = generate_embedding(args.image)

    print("Searching Firestore...")

    search_similar_images(embedding)


if __name__ == "__main__":
    main()