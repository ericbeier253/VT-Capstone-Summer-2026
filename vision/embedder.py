from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModel,
)


class DinoEmbedder:

    def __init__(

        self,

        model_name: str = "facebook/dinov2-base",

        device: str | None = None,

    ):

        self.device = (

            device

            if device is not None

            else (
                "cuda"

                if torch.cuda.is_available()

                else "cpu"
            )

        )

        self.processor = (

            AutoImageProcessor
            .from_pretrained(
                model_name
            )

        )

        self.model = (

            AutoModel
            .from_pretrained(
                model_name
            )
            .to(self.device)

        )

        self.model.eval()

    @torch.inference_mode()

    def embed(

        self,

        image_path: str | Path,

    ) -> list[float]:

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = self.processor(

            images=image,

            return_tensors="pt",

        )

        inputs = {

            k: v.to(self.device)

            for k, v in inputs.items()

        }

        outputs = self.model(

            **inputs

        )

        embedding = (

            outputs
            .last_hidden_state[:, 0]

        )

        embedding = F.normalize(

            embedding,

            p=2,

            dim=1,

        )

        embedding = (

            embedding

            .squeeze()

            .cpu()

            .numpy()

        )

        return embedding.tolist()