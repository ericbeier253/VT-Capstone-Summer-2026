from typing import List

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


class BoundingBox(BaseModel):
    """
    Bounding box in image coordinates.
    """

    x1: int = Field(description="Left coordinate")
    y1: int = Field(description="Top coordinate")
    x2: int = Field(description="Right coordinate")
    y2: int = Field(description="Bottom coordinate")

    @model_validator(mode="after")
    def validate_box(self):

        if self.x1 < 0 or self.y1 < 0:
            raise ValueError(
                "Bounding box coordinates must be non-negative."
            )

        if self.x1 >= self.x2:
            raise ValueError(
                "Bounding box must satisfy x1 < x2."
            )

        if self.y1 >= self.y2:
            raise ValueError(
                "Bounding box must satisfy y1 < y2."
            )

        return self


class SceneMeta(BaseModel):

    description: str

    environment: str

    lighting: str


class DetectedObject(BaseModel):

    object_name: str

    object_description: str

    object_location: str

    bounding_boxes: List[BoundingBox]

    is_gaze_target: bool


class ImageAnalysis(BaseModel):

    scene_meta: SceneMeta

    objects: List[DetectedObject]