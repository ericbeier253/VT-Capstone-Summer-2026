"""
rag_query.py

Interactive RAG query against rag_object_collection.

Pipeline:

1. Accept a natural-language question from the command line.
2. Use Gemini to determine whether the question is a valid
   object-location query and extract the requested objects.
3. Print the extracted JSON.
4. Generate a Gemini text embedding for each object name.
5. Perform Firestore vector similarity search against
   rag_object_collection.text_embedding.
6. Keep matches whose cosine distance <= 0.15.
7. Print the top 5 matches for each requested object.

Example:

    python rag_query.py "Where are my car keys and smartphone?"

Expected parsed query:

    {
        "objects": [
            "car keys",
            "smartphone"
        ]
    }
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from google import genai
from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.oauth2 import service_account


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

ENV_FILE = ROOT_DIR / ".env"
SERVICE_ACCOUNT = (
    ROOT_DIR / ".secrets" / "aria-uploader-key.json"
)

COLLECTION = "rag_object_collection"

TEXT_EMBEDDING_FIELD = "text_embedding"

EMBEDDING_MODEL = "gemini-embedding-2"

# Must match the dimensionality used when creating
# text_embedding vectors in Firestore.
EMBEDDING_DIMENSION = 768

# Cosine distance threshold.
#
# 0.0 = identical
# 1.0 = maximally different
#
# Therefore 0.15 means only matches with:
#
#     cosine distance <= 0.15
#
DISTANCE_THRESHOLD = 0.15

TOP_K = 5

# Number of vector-search candidates requested from Firestore.
#
# We request more than TOP_K because some results may be
# removed by the distance threshold.
SEARCH_LIMIT = 20


# ============================================================
# LOAD .ENV
# ============================================================

def load_env():
    """
    Load the project's .env file without requiring python-dotenv.
    """

    if not ENV_FILE.exists():
        return

    with open(ENV_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()

            if key.startswith("export "):
                key = key[len("export "):].strip()

            if not key:
                continue

            os.environ[key] = (
                value.strip()
                .strip('"')
                .strip("'")
            )


load_env()


# ============================================================
# CREDENTIALS
# ============================================================

if not SERVICE_ACCOUNT.exists():

    raise FileNotFoundError(
        f"Service account not found:\n"
        f"{SERVICE_ACCOUNT}"
    )


credentials = (
    service_account.Credentials
    .from_service_account_file(
        SERVICE_ACCOUNT
    )
)

PROJECT_ID = credentials.project_id


# ============================================================
# FIRESTORE CLIENT
# ============================================================

db = firestore.Client(
    project=PROJECT_ID,
    credentials=credentials,
)


# ============================================================
# GEMINI CLIENT
# ============================================================

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY was not found in .env"
    )


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# QUERY PARSING
# ============================================================

QUERY_PARSER_PROMPT = """
You are a query parser for an object-location retrieval system.

The system stores physical objects that were observed by
a wearable camera.

Given a user's question, determine whether it mentions or is asking about one or more physical objects (even if they are asking about an event, action, state, or time involving that object).

If it contains a physical object:
- Extract ONLY the object names being requested.
- Do not include verbs.
- Do not include locations.
- Do not include dates.
- Do not include question words.
- Do not include explanatory text.

For example:

User:
"Where are my car keys and smartphone?"

Return:
{
  "objects": [
    "car keys",
    "smartphone"
  ]
}

User:
"What time did I take my nasal spray?"

Return:
{
  "objects": [
    "nasal spray"
  ]
}

User:
"Did I leave the oven on?"

Return:
{
  "objects": [
    "oven"
  ]
}

If the question does NOT mention any specific physical objects, return:

{}

Examples of invalid queries:

"What's the weather today?"
{}

"Tell me a joke."
{}

"How does a car engine work?"
{}

"How are you?"
{}

Return ONLY valid JSON.
"""


def parse_query(
    query: str,
) -> dict[str, list[str]]:
    """
    Parse a natural-language query into:

        {
            "objects": [
                "object 1",
                "object 2"
            ]
        }

    or:

        {}

    if the query is invalid.
    """

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",#"gemini-2.5-flash-lite",
        contents=[
            QUERY_PARSER_PROMPT,
            f"\nUser query:\n{query}",
        ],
        config={
            "response_mime_type": "application/json",
        },
    )

    try:

        parsed = json.loads(
            response.text
        )

    except json.JSONDecodeError:

        print(
            "ERROR: Gemini returned invalid JSON."
        )

        return {}


    if not isinstance(parsed, dict):
        return {}


    objects = parsed.get("objects")


    if not isinstance(objects, list):
        return {}


    # Clean the extracted object names.

    cleaned_objects = []

    for obj in objects:

        if not isinstance(obj, str):
            continue

        obj = obj.strip()

        if obj:
            cleaned_objects.append(obj)


    if not cleaned_objects:
        return {}


    return {
        "objects": cleaned_objects
    }


# ============================================================
# TEXT EMBEDDING
# ============================================================

def embed_object_name(
    object_name: str,
) -> list[float]:
    """
    Generate a Gemini embedding for an object name.

    Query embeddings use RETRIEVAL_QUERY because this text
    represents the user's search query.

    Stored object embeddings should have been generated with
    RETRIEVAL_DOCUMENT.
    """

    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=object_name,
        config={
            "task_type": "RETRIEVAL_QUERY",
            "output_dimensionality": EMBEDDING_DIMENSION,
        },
    )

    if not response.embeddings:

        raise RuntimeError(
            "Gemini returned no embedding."
        )


    embedding = response.embeddings[0].values


    if len(embedding) != EMBEDDING_DIMENSION:

        raise RuntimeError(
            f"Unexpected embedding dimension: "
            f"{len(embedding)}. "
            f"Expected {EMBEDDING_DIMENSION}."
        )


    return embedding


# ============================================================
# FIRESTORE VECTOR SEARCH
# ============================================================

'''def search_object(
    object_name: str,
) -> list[Any]:
    """
    Search rag_object_collection.text_embedding for the
    closest semantic matches to object_name.

    Firestore returns cosine distance values.

    Only documents with:

        distance <= DISTANCE_THRESHOLD

    are retained.

    Returns at most TOP_K documents.
    """

    print()
    print("-" * 70)
    print(f"Searching for: {object_name}")
    print("-" * 70)


    # --------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------

    query_embedding = embed_object_name(
        object_name
    )


    # --------------------------------------------------------
    # Firestore vector search
    # --------------------------------------------------------

    query = (
        db
        .collection(COLLECTION)
        .find_nearest(
            vector_field=TEXT_EMBEDDING_FIELD,
            query_vector=query_embedding,
            distance_measure=DistanceMeasure.COSINE,
            limit=SEARCH_LIMIT,
            distance_result_field="vector_distance",
        )
    )


    documents = list(
        query.get()
    )


    # --------------------------------------------------------
    # Apply distance threshold
    # --------------------------------------------------------

    matches = []


    for doc in documents:

        data = doc.to_dict()

        distance = data.get(
            "vector_distance"
        )


        if distance is None:

            print(
                f"WARNING: No distance returned "
                f"for document {doc.id}"
            )

            continue


        if distance <= DISTANCE_THRESHOLD:

            matches.append(
                (
                    distance,
                    doc,
                )
            )


    # --------------------------------------------------------
    # Sort by closest distance
    # --------------------------------------------------------

    matches.sort(
        key=lambda item: item[0]
    )


    return matches[:TOP_K]'''


def search_object(
    object_name: str,
) -> list[tuple[float, Any]]:
    """
    Search rag_object_collection for all objects within the
    cosine-distance threshold.

    Returns all matching documents whose cosine distance is
    <= DISTANCE_THRESHOLD.

    Results are NOT limited to the top 5.
    """

    print()
    print("-" * 70)
    print(f"Searching for: {object_name}")
    print("-" * 70)

    # --------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------

    query_embedding = embed_object_name(
        object_name
    )

    # --------------------------------------------------------
    # Vector search
    #
    # Set a sufficiently large limit so that all likely
    # threshold matches are returned.
    # --------------------------------------------------------

    query = (
        db
        .collection(COLLECTION)
        .find_nearest(
            vector_field=TEXT_EMBEDDING_FIELD,
            query_vector=query_embedding,
            distance_measure=DistanceMeasure.COSINE,
            limit=1000,
            distance_result_field="vector_distance",
        )
    )

    documents = list(query.get())

    # --------------------------------------------------------
    # Apply cosine distance threshold
    # --------------------------------------------------------

    matches = []

    for doc in documents:

        data = doc.to_dict()

        distance = data.get(
            "vector_distance"
        )

        if distance is None:
            continue

        if distance <= DISTANCE_THRESHOLD:

            matches.append(
                (
                    distance,
                    doc,
                )
            )

    return matches



def get_latest_distinct_objects(
    matches: list[tuple[float, Any]],
) -> list[tuple[float, Any]]:
    """
    From all matching documents, return only the latest
    document for each distinct object_id.

    The timestamp field is used to determine the latest
    occurrence.
    """

    latest_by_object_id = {}

    for distance, doc in matches:

        data = doc.to_dict()

        object_id = data.get(
            "object_id"
        )

        timestamp = data.get(
            "timestamp"
        )

        if not object_id:
            continue

        # First occurrence of this object_id
        if object_id not in latest_by_object_id:

            latest_by_object_id[object_id] = (
                distance,
                doc,
            )

            continue

        # Existing occurrence
        existing_distance, existing_doc = (
            latest_by_object_id[object_id]
        )

        existing_data = (
            existing_doc.to_dict()
        )

        existing_timestamp = (
            existing_data.get("timestamp")
        )

        # Keep the newer occurrence
        if (
            timestamp is not None
            and (
                existing_timestamp is None
                or timestamp > existing_timestamp
            )
        ):

            latest_by_object_id[object_id] = (
                distance,
                doc,
            )

    # Convert dictionary back to list
    results = list(
        latest_by_object_id.values()
    )

    # Sort latest occurrences by timestamp,
    # newest first.
    results.sort(
        key=lambda item: (
            item[1].to_dict().get(
                "timestamp"
            )
            is not None,
            item[1].to_dict().get(
                "timestamp"
            ),
        ),
        reverse=True,
    )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def get_results(
    object_name: str,
    matches: list[Any],
) -> dict[str, Any]:
    """
    Print Firestore documents and similarity distances and return
    structured JSON-friendly results.
    """

    payload: dict[str, Any] = {
        "object_name": object_name,
        "match_count": len(matches),
        "results": [],
    }

    print()
    print(f"RESULTS FOR: {object_name}")
    print(
        f"Matches within cosine distance "
        f"{DISTANCE_THRESHOLD}: "
        f"{len(matches)}"
    )
    print("=" * 70)

    if not matches:
        print("No matching objects found.")
        payload["message"] = "No matching objects found."
        print(json.dumps(payload, indent=2, default=str))
        return payload

    keep_fields = [
        "bounding_boxes",
        "crop_path",
        "is_gaze_target",
        "object_description",
        "object_id",
        "object_location",
        "object_name",
        "parent_image",
        "run_id",
        "scene_meta",
        "timestamp",
    ]

    def normalize_crop_path(crop_path: str) -> str:
        parts = crop_path.split("/")
        if not parts:
            return crop_path

        # Drop first and last items
        normalized_parts = parts[1:-1]
        if not normalized_parts:
            return crop_path

        # If the original path was a gs:// URL, preserve the bucket name
        if crop_path.startswith("gs://"):
            bucket = normalized_parts[0]
            remaining = normalized_parts[1:]
            if remaining:
                return "/".join([bucket, *remaining])
            return bucket

        return "/".join(normalized_parts)

    for rank, (distance, doc) in enumerate(matches, start=1):
        data = doc.to_dict() or {}
        data.pop("vector_distance", None)
        filtered_data = {key: data[key] for key in keep_fields if key in data}

        if "crop_path" in filtered_data:
            filtered_data["crop_path"] = normalize_crop_path(filtered_data["crop_path"])

        entry = {
            "rank": rank,
            "distance": distance,
            "document_id": doc.id,
            "fields": filtered_data,
        }
        payload["results"].append(entry)

        print()
        print(f"Rank #{rank}")
        print(f"Cosine distance: {distance:.6f}")
        print(f"Document ID: {doc.id}")
        print(json.dumps(filtered_data, indent=2, default=str))

    print(json.dumps(payload, indent=2, default=str))
    return payload


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            '  python rag_query.py '
            '"Where are my car keys and smartphone?"'
        )

        sys.exit(1)


    # --------------------------------------------------------
    # User query
    # --------------------------------------------------------

    user_query = " ".join(
        sys.argv[1:]
    )


    print()
    print("=" * 70)
    print("RAG OBJECT SEARCH")
    print("=" * 70)

    print(
        f"Query: {user_query}"
    )


    # --------------------------------------------------------
    # Parse query
    # --------------------------------------------------------

    parsed_query = parse_query(
        user_query
    )


    print()
    print("Parsed JSON:")

    print(
        json.dumps(
            parsed_query,
            indent=2,
        )
    )


    # --------------------------------------------------------
    # Invalid query
    # --------------------------------------------------------

    if not parsed_query:

        print()
        print(
            "Query is not a valid object-location query."
        )

        return


    # --------------------------------------------------------
    # Search each object
    # --------------------------------------------------------

    objects = parsed_query[
        "objects"
    ]


    for object_name in objects:

        try:

            #matches = search_object(
            #    object_name
            #)

            #get_results(
            #    object_name,
            #    matches,
            #)

            matches = search_object(
                object_name
            )

            print(
                f"Found {len(matches)} "
                f"matches within threshold."
            )

            latest_objects = get_latest_distinct_objects(
                matches
            )

            print(
                f"Found {len(latest_objects)} "
                f"distinct objects."
            )

            get_results(
                object_name,
                latest_objects,
            )

        except Exception as e:

            print()
            print(
                f"ERROR searching for "
                f"'{object_name}': {e}"
            )


    print()
    print("=" * 70)
    print("Search complete.")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
