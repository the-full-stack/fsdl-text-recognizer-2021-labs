"""Class for loading the IAM dataset, which encompasses both paragraphs and lines, with associated utilities."""
from pathlib import Path
from typing import Dict, List
import argparse
import os
import xml.etree.ElementTree as ElementTree
import zipfile

from boltons.cacheutils import cachedproperty
import toml

from text_recognizer.data.base_data_module import BaseDataModule, _download_raw_dataset, load_and_print_info


RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "iam"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "iam"
EXTRACTED_DATASET_DIRNAME = DL_DATA_DIRNAME / "iamdb"

DOWNSAMPLE_FACTOR = 2  # If images were downsampled, the regions must also be.
LINE_REGION_PADDING = 16  # add this many pixels around the exact coordinates


class IAM(BaseDataModule):
    """
    "The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.
    From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.metadata = toml.load(METADATA_FILENAME)

    def prepare_data(self, *args, **kwargs) -> None:
        if self.xml_filenames:
            return
        filename = _download_raw_dataset(self.metadata, DL_DATA_DIRNAME)  # type: ignore
        _extract_raw_dataset(filename, DL_DATA_DIRNAME)

    @property
    def xml_filenames(self):
        return list((EXTRACTED_DATASET_DIRNAME / "xml").glob("*.xml"))

    @property
    def form_filenames(self):
        return list((EXTRACTED_DATASET_DIRNAME / "forms").glob("*.jpg"))

    @property
    def form_filenames_by_id(self):
        return {filename.stem: filename for filename in self.form_filenames}

    @property
    def split_by_id(self):
        return {
            filename.stem: "test" if filename.stem in self.metadata["test_ids"] else "trainval"
            for filename in self.form_filenames
        }

    @cachedproperty
    def line_strings_by_id(self):
        """Return a dict from name of IAM form to a list of line texts in it."""
        return {filename.stem: _get_line_strings_from_xml_file(filename) for filename in self.xml_filenames}

    @cachedproperty
    def line_regions_by_id(self):
        """Return a dict from name of IAM form to a list of (x1, x2, y1, y2) coordinates of all lines in it."""
        return {filename.stem: _get_line_regions_from_xml_file(filename) for filename in self.xml_filenames}

    def __repr__(self):
        """Print info about the dataset."""
        return "IAM Dataset\n" f"Num total: {len(self.xml_filenames)}\nNum test: {len(self.metadata['test_ids'])}\n"


def _extract_raw_dataset(filename: Path, dirname: Path) -> None:
    print("Extracting IAM data")
    curdir = os.getcwd()
    os.chdir(dirname)
    with zipfile.ZipFile(filename, "r") as zip_file:
        zip_file.extractall()
    os.chdir(curdir)


def _get_line_strings_from_xml_file(filename: str) -> List[str]:
    """Get the text content of each line. Note that we replace &quot; with "."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall("handwritten-part/line")
    return [el.attrib["text"].replace("&quot;", '"') for el in xml_line_elements]


def _get_line_regions_from_xml_file(filename: str) -> List[Dict[str, int]]:
    """Get the line region dict for each line."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall("handwritten-part/line")
    return [_get_line_region_from_xml_element(el) for el in xml_line_elements]


def _get_line_region_from_xml_element(xml_line) -> Dict[str, int]:
    """
    Parameters
    ----------
    xml_line
        xml element that has x, y, width, and height attributes
    """
    word_elements = xml_line.findall("word/cmp")
    x1s = [int(el.attrib["x"]) for el in word_elements]
    y1s = [int(el.attrib["y"]) for el in word_elements]
    x2s = [int(el.attrib["x"]) + int(el.attrib["width"]) for el in word_elements]
    y2s = [int(el.attrib["y"]) + int(el.attrib["height"]) for el in word_elements]
    return {
        "x1": min(x1s) // DOWNSAMPLE_FACTOR - LINE_REGION_PADDING,
        "y1": min(y1s) // DOWNSAMPLE_FACTOR - LINE_REGION_PADDING,
        "x2": max(x2s) // DOWNSAMPLE_FACTOR + LINE_REGION_PADDING,
        "y2": max(y2s) // DOWNSAMPLE_FACTOR + LINE_REGION_PADDING,
    }


if __name__ == "__main__":
    load_and_print_info(IAM)
