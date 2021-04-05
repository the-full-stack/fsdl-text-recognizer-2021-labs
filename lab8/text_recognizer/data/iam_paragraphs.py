"""IAM Paragraphs Dataset class."""
import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import json
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.emnist import EMNIST
from text_recognizer.data.iam import IAM
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels, split_dataset


PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "iam_paragraphs"

NEW_LINE_TOKEN = "\n"
TRAIN_FRAC = 0.8

IMAGE_SCALE_FACTOR = 2
IMAGE_HEIGHT = 1152 // IMAGE_SCALE_FACTOR
IMAGE_WIDTH = 1280 // IMAGE_SCALE_FACTOR
MAX_LABEL_LENGTH = 682


class IAMParagraphs(BaseDataModule):
    """
    IAM Handwriting database paragraphs.
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "true").lower() == "true"

        mapping = EMNIST().mapping
        assert mapping is not None
        self.mapping = [*mapping, NEW_LINE_TOKEN]
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}

        self.dims = (1, IMAGE_HEIGHT, IMAGE_WIDTH)  # We assert that this is correct in setup()
        self.output_dims = (MAX_LABEL_LENGTH, 1)  # We assert that this is correct in setup()

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        if PROCESSED_DATA_DIRNAME.exists():
            return
        print("IAMParagraphs.prepare_data: Cropping IAM paragraph regions and saving them along with labels...")

        iam = IAM()
        iam.prepare_data()

        properties = {}
        for split in ["trainval", "test"]:
            crops, labels = get_paragraph_crops_and_labels(iam=iam, split=split)
            save_crops_and_labels(crops=crops, labels=labels, split=split)

            properties.update(
                {
                    id_: {
                        "crop_shape": crops[id_].size[::-1],
                        "label_length": len(label),
                        "num_lines": _num_lines(label),
                    }
                    for id_, label in labels.items()
                }
            )

        with open(PROCESSED_DATA_DIRNAME / "_properties.json", "w") as f:
            json.dump(properties, f, indent=4)

    def setup(self, stage: str = None) -> None:
        def _load_dataset(split: str, augment: bool) -> BaseDataset:
            crops, labels = load_processed_crops_and_labels(split)
            X = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops]
            Y = convert_strings_to_labels(strings=labels, mapping=self.inverse_mapping, length=self.output_dims[0])
            transform = get_transform(image_shape=self.dims[1:], augment=augment)  # type: ignore
            return BaseDataset(X, Y, transform=transform)

        print(f"IAMParagraphs.setup({stage}): Loading IAM paragraph regions and lines...")
        validate_input_and_output_dimensions(input_dims=self.dims, output_dims=self.output_dims)

        if stage == "fit" or stage is None:
            data_trainval = _load_dataset(split="trainval", augment=self.augment)
            self.data_train, self.data_val = split_dataset(base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)

        if stage == "test" or stage is None:
            self.data_test = _load_dataset(split="test", augment=False)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Paragraphs Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


def validate_input_and_output_dimensions(
    input_dims: Optional[Tuple[int, ...]], output_dims: Optional[Tuple[int, ...]]
) -> None:
    """Validate input and output dimensions against the properties of the dataset."""
    properties = get_dataset_properties()

    max_image_shape = properties["crop_shape"]["max"] / IMAGE_SCALE_FACTOR
    assert input_dims is not None and input_dims[1] >= max_image_shape[0] and input_dims[2] >= max_image_shape[1]

    # Add 2 because of start and end tokens
    assert output_dims is not None and output_dims[0] >= properties["label_length"]["max"] + 2


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((image.width // scale_factor, image.height // scale_factor), resample=Image.BILINEAR)


def get_paragraph_crops_and_labels(iam: IAM, split: str) -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """Load IAM paragraph crops and labels for a given split."""
    crops = {}
    labels = {}
    for form_filename in iam.form_filenames:
        id_ = form_filename.stem
        if not iam.split_by_id[id_] == split:
            continue
        image = Image.open(form_filename)
        image = ImageOps.grayscale(image)
        image = ImageOps.invert(image)

        line_regions = iam.line_regions_by_id[id_]
        para_bbox = [
            min([_["x1"] for _ in line_regions]),
            min([_["y1"] for _ in line_regions]),
            max([_["x2"] for _ in line_regions]),
            max([_["y2"] for _ in line_regions]),
        ]
        lines = iam.line_strings_by_id[id_]

        crops[id_] = image.crop(para_bbox)
        labels[id_] = NEW_LINE_TOKEN.join(lines)
    assert len(crops) == len(labels)
    return crops, labels


def save_crops_and_labels(crops: Dict[str, Image.Image], labels: Dict[str, str], split: str):
    """Save crops, labels and shapes of crops of a split."""
    (PROCESSED_DATA_DIRNAME / split).mkdir(parents=True, exist_ok=True)

    with open(_labels_filename(split), "w") as f:
        json.dump(labels, f, indent=4)

    for id_, crop in crops.items():
        crop.save(_crop_filename(id_, split))


def load_processed_crops_and_labels(split: str) -> Tuple[Sequence[Image.Image], Sequence[str]]:
    """Load processed crops and labels for given split."""
    with open(_labels_filename(split), "r") as f:
        labels = json.load(f)

    sorted_ids = sorted(labels.keys())
    ordered_crops = [Image.open(_crop_filename(id_, split)).convert("L") for id_ in sorted_ids]
    ordered_labels = [labels[id_] for id_ in sorted_ids]

    assert len(ordered_crops) == len(ordered_labels)
    return ordered_crops, ordered_labels


def get_transform(image_shape: Tuple[int, int], augment: bool) -> transforms.Compose:
    """Get transformations for images."""
    if augment:
        transforms_list = [
            transforms.RandomCrop(  # random pad image to image_shape with 0
                size=image_shape, padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
            ),
            transforms.ColorJitter(brightness=(0.8, 1.6)),
            transforms.RandomAffine(
                degrees=1,
                shear=(-10, 10),
                resample=Image.BILINEAR,
            ),
        ]
    else:
        transforms_list = [transforms.CenterCrop(image_shape)]  # pad image to image_shape with 0
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)


def get_dataset_properties() -> dict:
    """Return properties describing the overall dataset."""
    with open(PROCESSED_DATA_DIRNAME / "_properties.json", "r") as f:
        properties = json.load(f)

    def _get_property_values(key: str) -> list:
        return [_[key] for _ in properties.values()]

    crop_shapes = np.array(_get_property_values("crop_shape"))
    aspect_ratios = crop_shapes[:, 1] / crop_shapes[:, 0]
    return {
        "label_length": {
            "min": min(_get_property_values("label_length")),
            "max": max(_get_property_values("label_length")),
        },
        "num_lines": {"min": min(_get_property_values("num_lines")), "max": max(_get_property_values("num_lines"))},
        "crop_shape": {"min": crop_shapes.min(axis=0), "max": crop_shapes.max(axis=0)},
        "aspect_ratio": {"min": aspect_ratios.min(), "max": aspect_ratios.max()},
    }


def _labels_filename(split: str) -> Path:
    """Return filename of processed labels."""
    return PROCESSED_DATA_DIRNAME / split / "_labels.json"


def _crop_filename(id_: str, split: str) -> Path:
    """Return filename of processed crop."""
    return PROCESSED_DATA_DIRNAME / split / f"{id_}.png"


def _num_lines(label: str) -> int:
    """Return number of lines of text in label."""
    return label.count("\n") + 1


if __name__ == "__main__":
    load_and_print_info(IAMParagraphs)
