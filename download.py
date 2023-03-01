import fiftyone.zoo as foz
oid_train = foz.download_zoo_dataset(
    "open-images-v6",
    dataset_dir="/content/voxel/oid",
    split="train",
    classes="Person",
    label_types="detections",
    max_samples=10000,
    shuffle=True,
    seed=5,
    num_workers=32
)
oid_val = foz.download_zoo_dataset(
    "open-images-v6",
    dataset_dir="/content/voxel/oid",
    split="validation",
    classes="Person",
    label_types="detections",
    max_samples=2000,
    shuffle=True,
    seed=5,
    num_workers=64
)
