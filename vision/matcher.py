from __future__ import annotations

import uuid


class ObjectMatcher:

    def __init__(

        self,

        repository,

        threshold=0.15,

    ):

        self.repository = repository

        self.threshold = threshold

    def assign_object_id(

        self,

        embedding,

    ):

        neighbors = (

            self.repository

            .nearest_neighbors(

                embedding,

                limit=5,

            )

        )

        for neighbor in neighbors:

            data = neighbor.to_dict()

            distance = data.get("distance")

            if distance is None:

                continue

            if distance > self.threshold:

                continue

            object_id = data.get(

                "object_id"

            )

            if object_id:

                return object_id

        return str(uuid.uuid4())