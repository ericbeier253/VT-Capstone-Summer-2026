from pathlib import Path

from vision.cropper import ObjectCropper
from vision.schemas import (
    BoundingBox,
    DetectedObject,
    SceneMeta,
    ImageAnalysis,
)


def main():

    analysis = ImageAnalysis(

        scene_meta=SceneMeta(
            description="A laptop open on a table in a dimly lit restaurant or bar, with a person's hands typing on the keyboard. In the background, there are tables, chairs, and a television screen displaying a soccer game.",
            environment="Restaurant or bar interior",
            lighting="Dim interior lighting, highlighted by the laptop screen and small table lights",
        ),

        objects=[

            DetectedObject(

                object_name="laptop",

                object_description="Silver laptop showing a code editor on screen, placed on a dark countertop.",

                object_location="Center foreground",

                is_gaze_target=True,

                bounding_boxes=[
                    BoundingBox(
                        x1=414,
                        y1=381,
                        x2=794,
                        y2=961,
                    )
                ],
            )

        ],

    )

    cropper = ObjectCropper()

    crops = cropper.crop_objects(

        "test.jpg",

        analysis,

    )

    print()

    print(f"Crops: {len(crops)}")

    for crop in crops:

        print(crop.crop_path)

        assert crop.crop_path.exists()

    print()

    print("Cropper test passed.")


if __name__ == "__main__":

    main()