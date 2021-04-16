from pathlib import Path
from typing import Sequence, Union
import argparse
import json

from PIL import Image
import torch

from text_recognizer.data import IAMParagraphs
from text_recognizer.data.iam_paragraphs import resize_image, IMAGE_SCALE_FACTOR, get_transform
from text_recognizer.lit_models import TransformerLitModel
from text_recognizer.models import ResnetTransformer
import text_recognizer.util as util


CONFIG_AND_WEIGHTS_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "paragraph_text_recognizer"


class ParagraphTextRecognizer:
    """Class to recognize paragraph text in an image."""

    def __init__(self):
        data = IAMParagraphs()
        self.mapping = data.mapping
        inv_mapping = data.inverse_mapping
        self.ignore_tokens = [inv_mapping["<S>"], inv_mapping["<B>"], inv_mapping["<E>"], inv_mapping["<P>"]]
        self.transform = get_transform(image_shape=data.dims[1:], augment=False)

        with open(CONFIG_AND_WEIGHTS_DIRNAME / "config.json", "r") as file:
            config = json.load(file)
        args = argparse.Namespace(**config)

        model = ResnetTransformer(data_config=data.config(), args=args)
        self.lit_model = TransformerLitModel.load_from_checkpoint(
            checkpoint_path=CONFIG_AND_WEIGHTS_DIRNAME / "model.pt", args=args, model=model
        )
        self.lit_model.eval()
        self.scripted_model = self.lit_model.to_torchscript(method="script", file_path=None)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict/infer text in input image (which can be a file path)."""
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = util.read_image_pil(image, grayscale=True)

        image_pil = resize_image(image_pil, IMAGE_SCALE_FACTOR)
        image_tensor = self.transform(image_pil)

        y_pred = self.scripted_model(image_tensor.unsqueeze(axis=0))[0]
        pred_str = convert_y_label_to_string(y=y_pred, mapping=self.mapping, ignore_tokens=self.ignore_tokens)

        return pred_str


def convert_y_label_to_string(y: torch.Tensor, mapping: Sequence[str], ignore_tokens: Sequence[int]) -> str:
    return "".join([mapping[i] for i in y if i not in ignore_tokens])


def main():
    """
    Example runs:
    ```
    python text_recognizer/paragraph_text_recognizer.py text_recognizer/tests/support/paragraphs/a01-077.png
    python text_recognizer/paragraph_text_recognizer.py https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png
    """
    parser = argparse.ArgumentParser(description="Recognize handwritten text in an image file.")
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    text_recognizer = ParagraphTextRecognizer()
    pred_str = text_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
