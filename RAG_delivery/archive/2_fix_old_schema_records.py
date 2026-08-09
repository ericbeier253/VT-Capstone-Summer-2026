"""
fill_rag_object_metadata.py

Updates missing fields in rag_object_occurrences by looking up
corresponding gaze_events documents.

Mapping:

rag_object_occurrences.crop_path

    cropped_objects/
        run_x/
            gaze_trigger_x/
                object.jpg


becomes:


gaze_events.img_path

    gs://project-aria-gaze-photos-eb-01/
        run_x/
            gaze_trigger_x.jpg


Fields updated:

    object_name
    object_description
    object_location
"""

from google.cloud import firestore
from google.oauth2 import service_account
from pathlib import Path

from google import genai
from google.genai import types
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

RAG_COLLECTION = "rag_object_occurrences"
GAZE_COLLECTION = "gaze_events"

PROJECT_BUCKET = (
    "gs://project-aria-gaze-photos-eb-01/"
)


BATCH_SIZE = 500

DRY_RUN = False


# ============================================================================
# Firestore Client
# ============================================================================

def get_firestore_client():

    credentials_path = (
        Path(__file__).resolve().parent.parent
        / ".secrets"
        / "aria-uploader-key.json"
        #/ "service-account.json"
    )

    credentials = (
        service_account
        .Credentials
        .from_service_account_file(credentials_path)
    )

    return firestore.Client(
        project=credentials.project_id,
        credentials=credentials,
    )


# ============================================================================
# Path conversion
# ============================================================================

def crop_path_to_img_path(crop_path: str):
    """
    Convert:

    cropped_objects/run_20260715_102510/
        gaze_trigger_002_405.004/tray_0.jpg


    to:


    gs://project-aria-gaze-photos-eb-01/
        run_20260715_102510/
        gaze_trigger_002_405.004.jpg
    """

    if not crop_path:
        return None


    parts = crop_path.split("/")


    try:
        # cropped_objects
        # run_x
        # gaze_trigger_x
        # object.jpg

        run_id = parts[1]

        gaze_folder = parts[2]

    except IndexError:
        return None


    return (
        f"{PROJECT_BUCKET}"
        f"{run_id}/"
        f"{gaze_folder}.jpg"
    )


# ============================================================================
# Find matching object
# ============================================================================

def find_object_metadata(
    gaze_doc,
    crop_path
):
    """
    Match the object name from the crop filename.

    Example:

    tray_0.jpg

    matches:

    object_name:
        tray
    """

    if not gaze_doc.exists:
        return None


    data = gaze_doc.to_dict()


    objects = (
        data
        .get("llm_analysis", {})
        .get("objects", [])
    )


    crop_filename = (
        crop_path
        .split("/")[-1]
        .replace(".jpg", "")
    )


    # tray_0 -> tray
    base_name = (
        crop_filename
        .rsplit("_", 1)[0]
    )


    for obj in objects:

        obj_name = (
            obj
            .get("object_name", "")
            .lower()
        )


        if obj_name == base_name.lower():

            return {
                "object_name":
                    obj.get("object_name"),

                "object_description":
                    obj.get("object_description"),

                "object_location":
                    obj.get("object_location"),
            }


    return None



# ============================================================================
# Migration
# ============================================================================

def update_metadata():

    db = get_firestore_client()


    rag_ref = db.collection(RAG_COLLECTION)
    gaze_ref = db.collection(GAZE_COLLECTION)


    batch = db.batch()

    batch_count = 0

    updated = 0
    skipped = 0
    missing_events = 0
    missing_objects = 0


    for doc in rag_ref.stream():


        data = doc.to_dict()


        # Only update missing metadata

        if (
            data.get("parent_image")
            #data.get("object_name")
            #and data.get("object_location")
            #and data.get("object_description")
        ):
            continue


        crop_path = data.get("crop_path")


        if not crop_path:

            skipped += 1
            continue


        img_path = (
            crop_path_to_img_path(crop_path)
        )


        if not img_path:

            skipped += 1
            continue



        print()
        print("--------------------------------")
        print("Document:", doc.id)
        print("Image:", img_path)


        # Lookup gaze event

        gaze_query = (
            gaze_ref
            .where(
                "img_path",
                "==",
                img_path
            )
            .limit(1)
            .stream()
        )


        gaze_doc = next(
            gaze_query,
            None
        )


        if gaze_doc is None:

            print(
                "No gaze event found"
            )

            missing_events += 1
            continue



        metadata = find_object_metadata(
            gaze_doc,
            crop_path
        )


        if metadata is None:

            print(
                "No matching object found"
            )

            missing_objects += 1
            continue



        print(
            "Updating:",
            metadata
        )


        update_fields = {}


        if not data.get("object_name"):

            update_fields["object_name"] = (
                metadata["object_name"]
            )


        if not data.get("object_description"):

            update_fields["object_description"] = (
                metadata["object_description"]
            )


        if not data.get("object_location"):

            update_fields["object_location"] = (
                metadata["object_location"]
            )



        if update_fields:

            updated += 1


            if not DRY_RUN:

                batch.update(
                    doc.reference,
                    update_fields
                )


                batch_count += 1


                if batch_count >= BATCH_SIZE:

                    print(
                        "Committing batch..."
                    )

                    batch.commit()

                    batch = db.batch()

                    batch_count = 0



    if not DRY_RUN and batch_count:

        batch.commit()



    print()
    print("="*60)
    print("Completed")
    print("="*60)

    print(
        "Updated:",
        updated
    )

    print(
        "Missing gaze events:",
        missing_events
    )

    print(
        "Missing objects:",
        missing_objects
    )

    print(
        "Skipped:",
        skipped
    )



if __name__ == "__main__":
    update_metadata()