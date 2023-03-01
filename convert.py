import fiftyone as fo
import fiftyone.types as ft

oid_train = fo.Dataset.from_dir(
    dataset_dir="/content/voxel/oid/train",
    dataset_type=ft.OpenImagesV6Dataset,
    label_types="detections",
    label_field="detections"
)

oid_validation = fo.Dataset.from_dir(
    dataset_dir="/content/voxel/oid/validation",
    dataset_type=ft.OpenImagesV6Dataset,
    label_types="detections",
    label_field="detections"
)

oid_train.export(
    export_dir="/content/voxel/yolo/train",
    dataset_type=ft.YOLOv5Dataset,
    label_field="detections",
    split="train"
)

oid_validation.export(
    export_dir="/content/voxel/yolo/val",
    dataset_type=ft.YOLOv5Dataset,
    label_field="detections",
    split="val"
)
