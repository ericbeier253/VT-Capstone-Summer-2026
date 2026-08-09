from pathlib import Path

from PIL import Image

from google.genai import types

from vision.schemas import ImageAnalysis


PROMPT = """
Analyze the image.

The red crosshair indicates the user's gaze.

Detect every visible object.

For each object provide:

- object_name
- object_description
- object_location
- bounding_boxes
- is_gaze_target

Bounding boxes must tightly enclose the object.

If multiple bounding boxes exist for the same object,
return every bounding box.
"""


class GeminiAnalyzer:

    def __init__(
        self,
        client,
        model="gemini-3.5-flash-lite",
    ):

        self.client = client
        self.model = model

    def analyze(
        self,
        image_path: str | Path,
    ) -> ImageAnalysis:

        image = Image.open(image_path)

        response = self.client.models.generate_content(

            model=self.model,

            contents=[
                PROMPT,
                image,
            ],

            config=types.GenerateContentConfig(

                temperature=0,

                response_mime_type="application/json",

                response_schema=ImageAnalysis,

            ),

        )

        return response.parsed