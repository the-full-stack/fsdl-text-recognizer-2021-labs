from collections import defaultdict
import argparse

from torchvision import transforms
from PIL import Image
import h5py
import numpy as np
import torch

from text_recognizer.data.util import BaseDataset, convert_strings_to_labels
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.emnist import EMNIST


DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist_lines2"

MAX_LENGTH = 43
MIN_OVERLAP = 0.2
MAX_OVERLAP = 0.5
NUM_TRAIN = 10000
NUM_VAL = 2000
NUM_TEST = 2000
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 1024
IMAGE_X_PADDING = 28
MAX_OUTPUT_LENGTH = 89  # Same as IAMLines


class EMNISTLines2(BaseDataModule):
    """EMNIST Lines dataset: synthetic handwriting lines dataset made from EMNIST characters."""

    def __init__(
        self,
        args: argparse.Namespace = None,
    ):
        super().__init__(args)

        self.augment = self.args.get("augment_data", "true") == "true"
        self.max_length = self.args.get("max_length", MAX_LENGTH)
        self.min_overlap = self.args.get("min_overlap", MIN_OVERLAP)
        self.max_overlap = self.args.get("max_overlap", MAX_OVERLAP)
        self.num_train = self.args.get("num_train", NUM_TRAIN)
        self.num_val = self.args.get("num_val", NUM_VAL)
        self.num_test = self.args.get("num_test", NUM_TEST)

        self.emnist = EMNIST()
        self.mapping = self.emnist.mapping

        max_width = int(self.emnist.dims[2] * (self.max_length + 1) * (1 - self.min_overlap)) + IMAGE_X_PADDING
        assert max_width <= IMAGE_WIDTH

        self.dims = (
            self.emnist.dims[0],
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )
        assert self.max_length <= MAX_OUTPUT_LENGTH
        self.output_dims = (MAX_OUTPUT_LENGTH, 1)

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Max line length in characters.")
        parser.add_argument(
            "--min_overlap",
            type=float,
            default=MIN_OVERLAP,
            help="Min overlap between characters in a line, between 0 and 1.",
        )
        parser.add_argument(
            "--max_overlap",
            type=float,
            default=MAX_OVERLAP,
            help="Max overlap between characters in a line, between 0 and 1.",
        )
        return parser

    @property
    def data_filename(self):
        return (
            DATA_DIRNAME
            / f"ml_{self.max_length}_o{self.min_overlap:f}_{self.max_overlap:f}_ntr{self.num_train}_ntv{self.num_val}_nte{self.num_test}.h5"  # pylint: disable=line-too-long
        )

    def prepare_data(self, *args, **kwargs) -> None:
        if self.data_filename.exists():
            return
        np.random.seed(42)
        self._generate_data("train")
        self._generate_data("val")
        self._generate_data("test")

    def setup(self, stage: str = None) -> None:
        print("EMNISTLines2 loading data from HDF5...")
        if stage == "fit" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_train = [Image.fromarray(_) for _ in f["x_train"][:]]
                y_train = torch.LongTensor(f["y_train"][:])
                x_val = [Image.fromarray(_) for _ in f["x_val"][:]]
                y_val = torch.LongTensor(f["y_val"][:])

            self.data_train = BaseDataset(x_train, y_train, transform=get_transform(augment=self.augment))
            self.data_val = BaseDataset(x_val, y_val, transform=get_transform(augment=self.augment))

        if stage == "test" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_test = f["x_test"][:]
                y_test = torch.LongTensor(f["y_test"][:])
            self.data_test = BaseDataset(x_test, y_test, transform=get_transform(augment=False))

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "EMNISTLines2 Dataset\n"  # pylint: disable=no-member
            f"Min overlap: {self.min_overlap}\n"
            f"Max overlap: {self.max_overlap}\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data

    def _generate_data(self, split: str) -> None:
        print(f"EMNISTLines2 generating data for {split}...")

        # pylint: disable=import-outside-toplevel
        from text_recognizer.data.sentence_generator import SentenceGenerator

        sentence_generator = SentenceGenerator(self.max_length - 2)  # Subtract two because we will add start/end tokens

        emnist = self.emnist
        emnist.prepare_data()
        emnist.setup()

        if split == "train":
            samples_by_char = get_samples_by_char(emnist.x_trainval, emnist.y_trainval, emnist.mapping)
            num = self.num_train
        elif split == "val":
            samples_by_char = get_samples_by_char(emnist.x_trainval, emnist.y_trainval, emnist.mapping)
            num = self.num_val
        else:
            samples_by_char = get_samples_by_char(emnist.x_test, emnist.y_test, emnist.mapping)
            num = self.num_test

        DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.data_filename, "a") as f:
            x, y = create_dataset_of_images(
                num, samples_by_char, sentence_generator, self.min_overlap, self.max_overlap, self.dims
            )
            y = convert_strings_to_labels(
                y,
                emnist.inverse_mapping,
                length=MAX_OUTPUT_LENGTH,
            )
            f.create_dataset(f"x_{split}", data=x, dtype="u1", compression="lzf")
            f.create_dataset(f"y_{split}", data=y, dtype="u1", compression="lzf")


def get_samples_by_char(samples, labels, mapping):
    samples_by_char = defaultdict(list)
    for sample, label in zip(samples, labels):
        samples_by_char[mapping[label]].append(sample)
    return samples_by_char


def select_letter_samples_for_string(string, samples_by_char):
    zero_image = torch.zeros((28, 28), dtype=torch.uint8)
    sample_image_by_char = {}
    for char in string:
        if char in sample_image_by_char:
            continue
        samples = samples_by_char[char]
        sample = samples[np.random.choice(len(samples))] if samples else zero_image
        sample_image_by_char[char] = sample.reshape(28, 28)
    return [sample_image_by_char[char] for char in string]


def construct_image_from_string(
    string: str, samples_by_char: dict, min_overlap: float, max_overlap: float, width: int
) -> torch.Tensor:
    overlap = np.random.uniform(min_overlap, max_overlap)
    sampled_images = select_letter_samples_for_string(string, samples_by_char)
    H, W = sampled_images[0].shape
    next_overlap_width = W - int(overlap * W)
    concatenated_image = torch.zeros((H, width), dtype=torch.uint8)
    x = IMAGE_X_PADDING
    for image in sampled_images:
        concatenated_image[:, x : (x + W)] += image
        x += next_overlap_width
    return torch.minimum(torch.Tensor([255]), concatenated_image)


def create_dataset_of_images(N, samples_by_char, sentence_generator, min_overlap, max_overlap, dims):
    images = torch.zeros((N, IMAGE_HEIGHT, dims[2]))
    labels = []
    for n in range(N):
        label = sentence_generator.generate()
        crop = construct_image_from_string(label, samples_by_char, min_overlap, max_overlap, dims[-1])
        height = crop.shape[0]
        y1 = (IMAGE_HEIGHT - height) // 2
        images[n, y1 : (y1 + height), :] = crop
        labels.append(label)
    return images, labels


def get_transform(augment=False):
    """Augment with brightness, slight rotation, slant, translation, scale, and Gaussian noise."""
    if not augment:
        return transforms.Compose([transforms.ToTensor()])
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=(0.5, 1)),
            transforms.RandomAffine(
                degrees=3, translate=(0, 0.05), scale=(0.4, 1.1), shear=(-40, 50), resample=Image.BILINEAR, fillcolor=0
            ),
            transforms.ToTensor(),
        ]
    )


if __name__ == "__main__":
    load_and_print_info(EMNISTLines2)
