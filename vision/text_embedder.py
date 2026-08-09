"""
text_embedder.py

Generates Gemini text embeddings for detected objects.

Input:
    Crop objects containing:

        crop.object_data.object_name
        crop.object_data.object_description

Output:
    A list of embedding vectors in the same order as the
    supplied crops.

The class intentionally does not create its own Gemini client.
The client is injected through the constructor so that the
application controls authentication/configuration.
"""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types


class TextEmbedder:
    """
    Generates semantic text embeddings for object metadata.

    Expected crop structure:

        crop.object_data.object_name
        crop.object_data.object_description
    """

    def __init__(
        self,
        client: genai.Client,
        model: str = "gemini-embedding-001",
        output_dimensionality: int = 768,
    ):
        self.client = client
        self.model = model
        self.output_dimensionality = output_dimensionality

    # =========================================================
    # TEXT CREATION
    # =========================================================

    @staticmethod
    def _object_to_text(crop: Any) -> str:
        """
        Convert the object's name and description into the text
        that will be embedded.

        Example:

            object_name:
                "pen"

            object_description:
                "A black pen resting inside a desk organizer tray"

        becomes:

            "Object name: pen. Object description: A black pen
             resting inside a desk organizer tray."
        """

        object_data = crop.object_data

        object_name = (
            object_data.object_name
            or ""
        ).strip()

        object_description = (
            object_data.object_description
            or ""
        ).strip()

        if not object_name and not object_description:
            raise ValueError(
                "Crop contains no object_name or "
                "object_description."
            )

        if object_name and object_description:
            return (
                f"Object name: {object_name}. "
                f"Object description: {object_description}."
            )

        if object_name:
            return f"Object name: {object_name}."

        return (
            f"Object description: "
            f"{object_description}."
        )

    # =========================================================
    # SINGLE OBJECT
    # =========================================================

    def embed(self, crop: Any) -> list[float]:
        """
        Generate an embedding for a single crop.

        Returns:
            list[float]: Text embedding vector.
        """

        text = self._object_to_text(crop)

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.output_dimensionality,
            ),
        )

        if not response.embeddings:
            raise RuntimeError(
                "Gemini returned no embedding."
            )

        return response.embeddings[0].values

    # =========================================================
    # BATCH
    # =========================================================

    def embed_batch(
        self,
        crops: list[Any],
    ) -> list[list[float]]:
        """
        Generate embeddings for a batch of crops.

        The returned embeddings correspond to the input crops
        in exactly the same order.

        Args:
            crops:
                List of crop objects containing object_data.

        Returns:
            List of embedding vectors.
        """

        if not crops:
            return []

        texts = [
            self._object_to_text(crop)
            for crop in crops
        ]

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.output_dimensionality,
            ),
        )

        embeddings = response.embeddings

        if len(embeddings) != len(crops):
            raise RuntimeError(
                "Gemini returned an unexpected number of "
                f"embeddings: expected {len(crops)}, "
                f"received {len(embeddings)}."
            )

        return [
            embedding.values
            for embedding in embeddings
        ]

