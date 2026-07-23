import json
from pathlib import Path

from PIL import Image
from google.cloud import storage
from google.oauth2 import service_account


# =====================================================
# CONFIG
# =====================================================

PROJECT_ID = "Project Aria"

BUCKET_NAME = "project-aria-gaze-photos-eb-01"

#JSON_FILE = "results/run_20260711_192133/gaze_trigger_001_140.954.json"

OUTPUT_ROOT = "cropped_objects"

credentials = service_account.Credentials.from_service_account_file(
    ".secrets/aria-uploader-key.json",
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform"
    ]
)

storage_client = storage.Client(
    project=PROJECT_ID,
    credentials=credentials
)

# =====================================================
# HELPERS
# =====================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    #".jpeg",
    #".png",
    #".webp"
]


def find_image_blob(bucket, image_stem, folder):

    for ext in IMAGE_EXTENSIONS:

        candidate = f"{folder}/{image_stem}{ext}"

        blob = bucket.blob(candidate)

        if blob.exists():
            return blob

    return None


# =====================================================
# MAIN
# =====================================================

def crop_objects_from_json(json_path):

    with open(json_path, "r") as f:
        data = json.load(f)

    json_path = Path(json_path)

    image_stem = json_path.stem

    folder = str(json_path.parent).replace("results/", "")

    bucket = storage_client.bucket(BUCKET_NAME)

    image_blob = find_image_blob(
        bucket,
        image_stem,
        folder
    )

    if image_blob is None:
        raise RuntimeError(
            f"Image not found for {json_path}"
        )

    image_bytes = image_blob.download_as_bytes()

    from io import BytesIO

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    output_dir = (
        Path(OUTPUT_ROOT)
        / folder
        / image_stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for obj in data["objects"]:

        object_name = (
            obj["object_name"]
            .replace("/", "_")
            .replace(" ", "_")
        )

        boxes = obj.get(
            "bounding_boxes",
            []
        )

        for idx, box in enumerate(boxes):

            #x1, y1, x2, y2 = box
            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]

            crop = image.crop(
                (int(2560*x1/1000), int(1920*y1/1000), int(2560*x2/1000), int(1920*y2/1000))
            )

            output_file = (
                output_dir
                / f"{object_name}_{idx}.jpg"
            )

            crop.save(
                output_file,
                quality=95
            )

            print(
                f"Saved: {output_file}"
            )


if __name__ == "__main__":

    # Get only file names as strings
    #absolute_files = [str(f.resolve()) for f in Path("results").rglob("*") if f.is_file()]
    #print(absolute_files)
    relative_files = [str(f.resolve().relative_to(Path.cwd())) for f in Path("results").rglob("*") if f.is_file()]
    #print(relative_files)

    for file in relative_files:#[:1]:
        print(file)
        crop_objects_from_json(
            #absolute_files[0]
            file
            #JSON_FILE
        )
