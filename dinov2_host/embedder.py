from __future__ import annotations

import io
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class DinoEmbedder:
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str | None = None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        return self.embed_image_bytes_batch([image_bytes])[0]

    @torch.inference_mode()
    def embed_image_bytes_batch(self, images_bytes: list[bytes]) -> list[list[float]]:
        images = [Image.open(io.BytesIO(img_bytes)).convert("RGB") for img_bytes in images_bytes]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu().numpy()
        return embeddings.tolist()
