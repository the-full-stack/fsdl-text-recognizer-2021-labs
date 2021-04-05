import argparse
from pathlib import Path
from typing import Sequence
import json
import torch

from text_recognizer.data import IAMParagraphs
from text_recognizer.data.iam_paragraphs import resize_image, IMAGE_SCALE_FACTOR, get_transform
from text_recognizer.models import ResnetTransformer
from text_recognizer.lit_models import TransformerLitModel
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

    @torch.no_grad()
    def predict(self, image_filename: Path) -> str:
        """Predict/infer text in input image filename."""
        pil_img = util.read_image_pil(image_filename, grayscale=True)
        pil_img = resize_image(pil_img, IMAGE_SCALE_FACTOR)  # ideally resize should have been part of transform
        img_tensor = self.transform(pil_img)

        y_pred = self.lit_model(img_tensor.unsqueeze(axis=0))[0]
        pred_str = convert_y_label_to_string(y=y_pred, mapping=self.mapping, ignore_tokens=self.ignore_tokens)

        return pred_str


def convert_y_label_to_string(y: torch.Tensor, mapping: Sequence[str], ignore_tokens: Sequence[int]) -> str:
    return "".join([mapping[i] for i in y if i not in ignore_tokens])
