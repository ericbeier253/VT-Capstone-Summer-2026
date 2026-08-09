import torch
from transformers import AutoImageProcessor, AutoModel

print(torch.__version__)
print(torch.cuda.is_available())

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

print("Loaded successfully!")