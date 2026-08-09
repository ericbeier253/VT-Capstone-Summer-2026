from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image

from vision.schemas import ImageAnalysis, DetectedObject


@dataclass(slots=True)
class CropResult:
    """
    Result of cropping a detected object.
    """

    crop_path: Path
    object_data: DetectedObject
    image_width: int
    image_height: int


class ObjectCropper:

    def __init__(
        self,
        output_root: str | Path = "cropped_objects",
        padding: int = 5,
    ):
        self.output_root = Path(output_root)
        self.padding = padding

    def crop_objects(
        self,
        image_path: str | Path,
        analysis: ImageAnalysis,
    ) -> List[CropResult]:

        image_path = Path(image_path)

        image = Image.open(image_path)

        width, height = image.size

        image_folder = (
            self.output_root /
            image_path.stem
        )

        image_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        results: List[CropResult] = []

        for object_index, obj in enumerate(analysis.objects):

            for bbox_index, bbox in enumerate(obj.bounding_boxes):

                x1 = max(
                    0,
                    bbox.x1 - self.padding,
                )

                y1 = max(
                    0,
                    bbox.y1 - self.padding,
                )

                x2 = min(
                    width,
                    bbox.x2 + self.padding,
                )

                y2 = min(
                    height,
                    bbox.y2 + self.padding,
                )

                crop = image.crop(
                    #(
                        #x1,
                        #y1,
                        #x2,
                        #y2,
                    #)
                    (int(2560*x1/1000), int(1920*y1/1000), int(2560*x2/1000), int(1920*y2/1000))
                )

                filename = (
                    f"object_"
                    f"{object_index:03d}_"
                    f"{bbox_index:02d}.jpg"
                )

                crop_path = (
                    image_folder /
                    filename
                )

                crop.save(crop_path)

                results.append(

                    CropResult(

                        crop_path=crop_path,

                        object_data=obj,

                        image_width=width,

                        image_height=height,

                    )

                )

        return results