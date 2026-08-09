from __future__ import annotations

from datetime import datetime

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure


class FirestoreRepository:

    def __init__(
        self,
        db: firestore.Client,
        collection="rag_object_collection", # HARDCODED
    ):

        self.db = db
        self.collection = db.collection(collection)

    def save_object(

        self,

        object_id: str,

        image_embedding: list[float],

        text_embedding: list[float],
        
        crop,

        run_id: str,

        parent_image: str,

    ):

        doc = {

            "object_id": object_id,

            "embedding": Vector(image_embedding),

            "text_embedding": Vector(text_embedding),

            "object_name":
                crop.object_data.object_name,

            "object_description":
                crop.object_data.object_description,

            "object_location":
                crop.object_data.object_location,

            "is_gaze_target":
                crop.object_data.is_gaze_target,

            "bounding_boxes": [

                b.model_dump()

                for b in crop.object_data.bounding_boxes

            ],

            "crop_path":
                str(crop.crop_path),

            "parent_image":
                parent_image,

            "run_id":
                run_id,

            "timestamp":
                firestore.SERVER_TIMESTAMP,

            "scene_meta": crop.object_data.scene_meta # Added by Sandeep

        }

        self.collection.document().set(doc)

    def nearest_neighbors(

        self,

        embedding,

        limit=10,

    ):

        return (

            self.collection

            .find_nearest(

                vector_field="embedding",

                query_vector=Vector(embedding),

                distance_measure=DistanceMeasure.COSINE,

                distance_result_field="distance",

                limit=limit,

            )

            .get()

        )