"""Utility functions for text_recognizer module."""
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve

# import base64
import hashlib

from PIL import Image
from tqdm import tqdm
import numpy as np


def to_categorical(y, num_classes):
    """1-hot encode a tensor."""
    return np.eye(num_classes, dtype="uint8")[y]


def read_image_pil(image_uri: Union[Path, str], grayscale=False) -> Image:
    with Image.open(image_uri) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
        return image


# Hide lines below until Lab 8
def read_b64_image(b64_string, grayscale=False):  # pylint: disable=unused-argument
    """Load base64-encoded images."""
    try:
        _, b64_data = b64_string.split(",")  # pylint: disable=unused-variable
        # return cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), imread_flag)
        return
    except Exception as exception:
        raise ValueError("Could not load image from b64 {}: {}".format(b64_string, exception)) from exception


# Hide lines above until Lab 8


def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec
