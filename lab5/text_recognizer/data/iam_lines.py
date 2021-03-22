"""
IamLinesDataset class.

We will use a processed version of this dataset, without including code that did the processing.
We will look at how to generate processed data from raw IAM data in the IamParagraphsDataset.
"""
from pathlib import Path
from typing import Sequence
import argparse
import json
import random

from PIL import Image, ImageFile, ImageOps
import numpy as np
import torch
from torchvision import transforms

from text_recognizer.data.util import BaseDataset, convert_strings_to_labels
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.emnist import EMNIST
from text_recognizer.data.iam import IAM

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "iam_lines"
TRAIN_FRAC = 0.8
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 1024  # Rounding up the actual empirical max to a power of 2

class IAMLines(BaseDataModule):
    """
    IAM Handwriting database lines.
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "true") == "true"
        self.mapping = EMNIST().mapping
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.dims = (1, IMAGE_HEIGHT, IMAGE_WIDTH)  # We assert that this is correct in setup()
        self.output_dims = (89, 1)  # We assert that this is correct in setup()
        self.data_train = None
        self.data_val = None
        self.data_test = None

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self):
        if PROCESSED_DATA_DIRNAME.exists():
            return

        print("Cropping IAM line regions...")
        iam = IAM()
        iam.prepare_data()
        crops_trainval, labels_trainval = line_crops_and_labels(iam, 'trainval')
        crops_test, labels_test = line_crops_and_labels(iam, 'test')

        shapes = np.array([crop.size for crop in crops_trainval + crops_test])
        aspect_ratios = shapes[:, 0] / shapes[:, 1]

        print("Saving images, labels, and statistics...")
        save_images_and_labels(crops_trainval, labels_trainval, 'trainval', PROCESSED_DATA_DIRNAME)
        save_images_and_labels(crops_test, labels_test, 'test', PROCESSED_DATA_DIRNAME)
        with open(PROCESSED_DATA_DIRNAME / '_max_aspect_ratio.txt', 'w') as file:
            file.write(str(aspect_ratios.max()))

    def setup(self, stage: str = None):
        with open(PROCESSED_DATA_DIRNAME / '_max_aspect_ratio.txt') as file:
            max_aspect_ratio = float(file.read())
            image_width = int(IMAGE_HEIGHT * max_aspect_ratio)
            assert image_width <= IMAGE_WIDTH

        if stage == "fit" or stage is None:
            x_trainval, labels_trainval = load_line_crops_and_labels('trainval', PROCESSED_DATA_DIRNAME)
            assert self.output_dims[0] >= max([len(l) for l in labels_trainval]) + 2  # Add 2 because of start and end tokens.

            y_trainval = convert_strings_to_labels(labels_trainval, self.inverse_mapping, length=self.output_dims[0])
            data_trainval = BaseDataset(x_trainval, y_trainval, transform=get_transform(IMAGE_WIDTH, self.augment))

            train_size = int(TRAIN_FRAC * len(data_trainval))
            val_size = len(data_trainval) - train_size
            self.data_train, self.data_val = torch.utils.data.random_split(
                data_trainval, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

        # Note that test data does not go through augmentation transforms
        if stage == "test" or stage is None:
            x_test, labels_test = load_line_crops_and_labels('test', PROCESSED_DATA_DIRNAME)
            assert self.output_dims[0] >= max([len(l) for l in labels_test]) + 2  # Add 2 because of start and end tokens.

            y_test = convert_strings_to_labels(labels_test, self.inverse_mapping, length=self.output_dims[0])
            self.data_test = BaseDataset(x_test, y_test, transform=get_transform(IMAGE_WIDTH))

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Lines Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
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


def line_crops_and_labels(iam: IAM, split: str):
    """Load IAM line labels and regions, and load line image crops."""
    crops = []
    labels = []
    for filename in iam.form_filenames:
        if not iam.split_by_id[filename.stem] == split:
            continue
        image = Image.open(filename)
        image = ImageOps.grayscale(image)
        image = ImageOps.invert(image)
        labels += iam.line_strings_by_id[filename.stem]
        crops += [
            image.crop([region[_] for _ in ['x1', 'y1', 'x2', 'y2']])
            for region in iam.line_regions_by_id[filename.stem]
        ]
    assert len(crops) == len(labels)
    return crops, labels


def save_images_and_labels(crops: Sequence[Image.Image], labels: Sequence[str], split: str, data_dirname: Path):
    (data_dirname / split).mkdir(parents=True, exist_ok=True)

    with open(data_dirname / split / '_labels.json', 'w') as f:
        json.dump(labels, f)
    for ind, crop in enumerate(crops):
        crop.save(data_dirname / split / f'{ind}.png')


def load_line_crops_and_labels(split: str, data_dirname: Path):
    """Load line crops and labels for given split from processed directory."""
    with open(data_dirname / split / '_labels.json') as file:
        labels = json.load(file)

    crop_filenames = sorted(
        (data_dirname / split).glob('*.png'),
        key=lambda filename: int(Path(filename).stem)
    )
    crops = [Image.open(filename).convert("L") for filename in crop_filenames]
    assert len(crops) == len(labels)
    return crops, labels


def get_transform(image_width, augment=False):
    """Augment with brightness, slight rotation, slant, translation, scale, and Gaussian noise."""
    def embed_crop(crop, augment=augment, image_width=image_width):
        image = Image.new("L", (image_width, IMAGE_HEIGHT))

        # Resize crop
        crop_width, crop_height = crop.size
        new_crop_height = IMAGE_HEIGHT
        new_crop_width = int(new_crop_height / crop_height * crop_width)
        if augment:
            # Add random stretching
            new_crop_width = int(new_crop_width * random.uniform(0.9, 1.1))
            new_crop_width = min(new_crop_width, image_width)
        crop_resized = crop.resize((new_crop_width, new_crop_height), resample=Image.BILINEAR)

        # Embed in the image
        x, y = 28, 0
        # if augment:
        #     x = random.randint(0, (image_width - new_crop_width))
        #     y = random.randint(0, (IMAGE_HEIGHT - new_crop_height))
        image.paste(crop_resized, (x, y))

        return image

    transforms_list = [transforms.Lambda(embed_crop)]
    if augment:
        transforms_list += [
            transforms.ColorJitter(brightness=(0.8, 1.6)),
            transforms.RandomAffine(
                degrees=1,
                shear=(-30, 20),
                resample=Image.BILINEAR,
            ),
        ]
    transforms_list += [
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x - 0.5)
    ]
    return transforms.Compose(transforms_list)


if __name__ == "__main__":
    load_and_print_info(IAMLines)
