import viewer

cases = [
    'gs://project-aria-gaze-photos-eb-01/run_20260714_132851/gaze_trigger_001_13257.750.jpg',
    'cropped_objects/run_20260725_170104/gaze_trigger_003_88100.348/keyboard_0.jpg',
    '/cropped_objects/run_20260725_170104/gaze_trigger_003_88100.348/keyboard_0.jpg',
    'foo/bar.jpg',
]
print('resolve_gcs_uri outputs:')
for c in cases:
    print(c, '->', viewer.resolve_gcs_uri(c))

obj_docs = [
    {'parent_image': 'webcam_0.jpg'},
    {'crop_path': 'cropped_objects/run_20260725_170104/gaze_trigger_003_88100.348/keyboard_0.jpg'},
    {'path': 'cropped_objects/run_20260725_170104/gaze_trigger_003_88100.348/keyboard_0.jpg'},
    {'img_path': 'gs://project-aria-gaze-photos-eb-01/run_20260714_132851/gaze_trigger_001_13257.750.jpg'},
]
print('\nobject_image_key outputs:')
for doc in obj_docs:
    print(doc, '->', viewer.object_image_key(doc))
