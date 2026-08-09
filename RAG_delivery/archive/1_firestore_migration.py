"""
migrate_object_collection.py

Converts all documents in `object_collection` into the NEW object schema
and writes them to `rag_object_occurrences`.

OLD schema:
    embedding
    folder
    image_name
    object_id
    object_name
    path
    run

NEW schema:
    bounding_boxes
    crop_path
    embedding
    is_gaze_target
    object_description
    object_id
    object_location
    parent_image
    run_id
    timestamp

For old records:
    run  -> run_id
    path -> crop_path

    run_YYYYMMDD_HHMMSS
        ->
    timestamp

Example:

    run_20260715_102510
        ->
    July 15, 2026 10:25:10 UTC

The original collection is never modified.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from google.cloud import firestore

from google.cloud import firestore
from google.oauth2 import service_account
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

SOURCE_COLLECTION = "object_collection"
DESTINATION_COLLECTION = "rag_object_occurrences"

# Firestore allows a maximum of 500 writes per batch.
BATCH_SIZE = 500

# Set to True first to verify the migration without writing.
DRY_RUN = False

# If False, existing destination documents are skipped.
OVERWRITE_EXISTING = True


# ============================================================================
# Firestore client
# ============================================================================

'''def get_firestore_client():
    """
    Use the existing Google Cloud / Firestore project configuration.

    This will use the same credentials your existing project uses, such as:

        GOOGLE_APPLICATION_CREDENTIALS

    or Application Default Credentials.
    """
    return firestore.Client()'''

def get_firestore_client():
    project_root = Path(__file__).resolve().parent.parent

    credentials_path = (
        project_root
        / ".secrets"
        / "aria-uploader-key.json"
        #/ "service-account.json"
    )

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path
    )

    return firestore.Client(
        project=credentials.project_id,
        credentials=credentials,
    )

# ============================================================================
# Timestamp extraction
# ============================================================================

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


# ============================================================================
# Schema detection
# ============================================================================

def is_new_schema(data: Dict[str, Any]) -> bool:
    """
    Determine whether a document is already using the new schema.

    The presence of `run_id`, `crop_path`, `timestamp`, or
    `object_description` is enough to identify the new schema.
    """

    new_fields = {
        "bounding_boxes",
        "crop_path",
        "is_gaze_target",
        "object_description",
        "object_location",
        "parent_image",
        "run_id",
        "timestamp",
    }

    return any(field in data for field in new_fields)


# ============================================================================
# Old -> New schema conversion
# ============================================================================

def convert_old_to_new(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an OLD schema document into the NEW schema.

    Old:

        embedding
        folder
        image_name
        object_id
        object_name
        path
        run

    New:

        bounding_boxes
        crop_path
        embedding
        is_gaze_target
        object_description
        object_id
        object_location
        parent_image
        run_id
        timestamp
    """

    run_id = data.get("run")

    timestamp = extract_timestamp(run_id)

    return {
        # ---------------------------------------------------------------
        # Existing fields
        # ---------------------------------------------------------------

        "embedding": data.get("embedding"),

        "object_id": data.get("object_id"),

        "object_name": data.get("object_name"),

        # ---------------------------------------------------------------
        # New-schema fields that cannot be recovered from old data
        # ---------------------------------------------------------------

        "bounding_boxes": [],

        "is_gaze_target": False,

        "object_description": None,

        "object_location": None,

        "parent_image": None,

        # ---------------------------------------------------------------
        # Renamed fields
        # ---------------------------------------------------------------

        "crop_path": data.get("path"),

        "run_id": run_id,

        # ---------------------------------------------------------------
        # Calculated timestamp
        # ---------------------------------------------------------------

        "timestamp": timestamp,
    }


# ============================================================================
# New schema normalization
# ============================================================================

def normalize_new_document(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct an existing NEW schema document using only the desired
    new-schema fields.

    This ensures that fields belonging to the old schema aren't carried
    forward accidentally.
    """

    return {
        "bounding_boxes": data.get("bounding_boxes", []),

        "crop_path": data.get("crop_path"),

        "embedding": data.get("embedding"),

        "is_gaze_target": data.get("is_gaze_target", False),

        "object_description": data.get("object_description"),

        "object_id": data.get("object_id"),

        "object_location": data.get("object_location"),

        "object_name": data.get("object_name"),

        "parent_image": data.get("parent_image"),

        "run_id": data.get("run_id"),

        "timestamp": data.get("timestamp"),
    }


# ============================================================================
# Document conversion
# ============================================================================

def convert_document(
    data: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:

    if is_new_schema(data):
        return "new", normalize_new_document(data)

    return "old", convert_old_to_new(data)


# ============================================================================
# Migration
# ============================================================================

def migrate():

    db = get_firestore_client()

    source_ref = db.collection(SOURCE_COLLECTION)
    destination_ref = db.collection(DESTINATION_COLLECTION)

    print("=" * 70)
    print("Firestore Object Collection Migration")
    print("=" * 70)

    print(f"Source      : {SOURCE_COLLECTION}")
    print(f"Destination : {DESTINATION_COLLECTION}")
    print(f"Dry run     : {DRY_RUN}")
    print(f"Overwrite   : {OVERWRITE_EXISTING}")

    print("=" * 70)

    batch = db.batch()
    batch_count = 0

    total_count = 0
    old_count = 0
    new_count = 0
    skipped_count = 0
    invalid_count = 0

    # ------------------------------------------------------------------------
    # Read source collection
    # ------------------------------------------------------------------------

    for source_doc in source_ref.stream():

        total_count += 1

        data = source_doc.to_dict()

        schema, normalized = convert_document(data)

        print()
        print("-" * 70)
        print(f"Document : {source_doc.id}")
        print(f"Schema   : {schema}")

        if schema == "old":
            old_count += 1
        else:
            new_count += 1

        # --------------------------------------------------------------------
        # Validate required fields
        # --------------------------------------------------------------------

        object_id = normalized.get("object_id")
        object_name = normalized.get("object_name")
        embedding = normalized.get("embedding")

        if not object_id:
            print("WARNING: Missing object_id. Skipping.")
            invalid_count += 1
            continue

        if not object_name:
            print("WARNING: Missing object_name.")

        if embedding is None:
            print("WARNING: Missing embedding.")

        # --------------------------------------------------------------------
        # Print conversion
        # --------------------------------------------------------------------

        print(f"object_id   : {object_id}")
        print(f"object_name : {object_name}")
        print(f"run_id      : {normalized.get('run_id')}")
        print(f"timestamp   : {normalized.get('timestamp')}")
        print(f"crop_path   : {normalized.get('crop_path')}")

        # --------------------------------------------------------------------
        # Destination reference
        #
        # Preserve the original Firestore document ID.
        # --------------------------------------------------------------------

        destination_doc = destination_ref.document(source_doc.id)

        # --------------------------------------------------------------------
        # Check for existing destination document
        # --------------------------------------------------------------------

        if not OVERWRITE_EXISTING and not DRY_RUN:

            existing = destination_doc.get()

            if existing.exists:
                print("Destination exists. Skipping.")
                skipped_count += 1
                continue

        # --------------------------------------------------------------------
        # Write
        # --------------------------------------------------------------------

        if not DRY_RUN:

            batch.set(
                destination_doc,
                normalized,
                merge=False,
            )

            batch_count += 1

            # Firestore maximum batch size = 500.
            if batch_count >= BATCH_SIZE:

                print(
                    f"\nCommitting batch of {batch_count} documents..."
                )

                batch.commit()

                print("Batch committed.")

                batch = db.batch()
                batch_count = 0

    # ------------------------------------------------------------------------
    # Commit remaining documents
    # ------------------------------------------------------------------------

    if not DRY_RUN and batch_count > 0:

        print(
            f"\nCommitting final batch of {batch_count} documents..."
        )

        batch.commit()

        print("Final batch committed.")

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Migration Complete")
    print("=" * 70)

    print(f"Total documents : {total_count}")
    print(f"Old documents   : {old_count}")
    print(f"New documents   : {new_count}")
    print(f"Skipped         : {skipped_count}")
    print(f"Invalid         : {invalid_count}")

    if DRY_RUN:
        print()
        print("DRY RUN: No documents were written.")

    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    migrate()