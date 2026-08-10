import uuid
import os
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.oauth2 import service_account


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

PROJECT_ID = os.environ["GCP_PROJECT"] #"Project Aria"

SERVICE_ACCOUNT = ".secrets/aria-uploader-key.json"

COLLECTION = "rag_object_collection" # "object_collection"

TOP_K = 30

# Maximum cosine distance to consider two objects identical.
# Tune this experimentally.

DISTANCE_THRESHOLD = 0.15


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
# Similarity Search
# -------------------------------------------------------

def nearest_neighbors(vector):

    return list(
        db.collection(COLLECTION)
        .find_nearest(
            vector_field="embedding",
            query_vector=vector,
            distance_measure=DistanceMeasure.COSINE,#DOT_PRODUCT,
            distance_result_field="distance",   # <-- REQUIRED
            limit=TOP_K,
        )
        .get()
    )

# -------------------------------------------------------
# Remove all existing object_ids
# -------------------------------------------------------

# Reload documents after cleanup

docs = list(db.collection(COLLECTION).stream())

print(f"Loaded {len(docs)} documents.")

batch = db.batch()

count = 0

for doc in docs:

    if "object_id" in doc.to_dict():

        batch.update(
            doc.reference,
            {
                "object_id": firestore.DELETE_FIELD
            }
        )

        count += 1

        # Firestore batches are limited to 500 operations
        if count % 500 == 0:
            batch.commit()
            batch = db.batch()

batch.commit()

print(f"Removed object_id from {count} documents.")

# -------------------------------------------------------
# Main
# -------------------------------------------------------

docs = list(db.collection(COLLECTION).stream())

print(f"Loaded {len(docs)} documents.")

for doc in docs:

    data = doc.to_dict()

    if "object_id" in data:
        continue

    print(f"\nProcessing {doc.id}")

    results = nearest_neighbors(data["embedding"])

    assigned_id = None

    for neighbor in results:

        if neighbor.id == doc.id:
            continue

        neighbor_data = neighbor.to_dict()

        #distance = neighbor_data.get("__distance__", 999)
        distance = neighbor.to_dict()["distance"]

        if distance > DISTANCE_THRESHOLD:
            continue

        if "object_id" in neighbor_data:

            assigned_id = neighbor_data["object_id"]

            print(
                f"Using existing object_id "
                f"{assigned_id} "
                f"(distance={distance:.4f})"
            )

            break

    #
    # No nearby object already has an ID.
    #
    if assigned_id is None:

        assigned_id = str(uuid.uuid4())

        print(
            f"Created new object_id "
            f"{assigned_id}"
        )

        #
        # Give the new ID to every close neighbor
        # that also lacks one.
        #
        batch = db.batch()

        batch.update(
            doc.reference,
            {"object_id": assigned_id}
        )

        for neighbor in results:

            #print(
            #    neighbor.id,
            #    n.get("__distance__"),
            #    n.get("object_id")
            #)

            if neighbor.id == doc.id:
                continue

            neighbor_data = neighbor.to_dict()

            #distance = neighbor_data.get("__distance__", 999)
            distance = neighbor.to_dict()["distance"]

            if distance > DISTANCE_THRESHOLD:
                continue

            if "object_id" not in neighbor_data:

                batch.update(
                    neighbor.reference,
                    {"object_id": assigned_id}
                )

        batch.commit()

    else:

        doc.reference.update(
            {
                "object_id": assigned_id
            }
        )

print("\nFinished.")