import os
from vision.embedder import DinoEmbedder
from vision.firestore_repo import FirestoreRepository
from vision.matcher import ObjectMatcher
from google.cloud import firestore
from google.oauth2 import service_account



# Load .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val.strip().strip('"').strip("'")
                #print(key, ":", val.strip().strip('"').strip("'"))

SERVICE_ACCOUNT = ".secrets/aria-uploader-key.json"

PROJECT_ID = os.environ["GCP_PROJECT"]

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT
)

db = firestore.Client(
    project=PROJECT_ID,
    credentials=credentials
)

repository = FirestoreRepository(db)

matcher = ObjectMatcher(

    repository,

    threshold=0.15,

)

embedder = DinoEmbedder()

embedding = embedder.embed(

        "cropped_objects/test/object_000_00.jpg"

    )

object_id = matcher.assign_object_id(

    embedding,

)

print(object_id)