"""AWS Lambda function serving text_recognizer predictions."""
from PIL import ImageStat

from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

model = ParagraphTextRecognizer()


def handler(event, _context):
    """Provide main prediction API"""
    image = _load_image(event)
    pred = model.predict(image)
    image_stat = ImageStat.Stat(image)
    print("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
    print("METRIC image_area {}".format(image.size[0] * image.size[1]))
    print("METRIC pred_length {}".format(len(pred)))
    print("INFO pred {}".format(pred))
    return {"pred": str(pred)}


def _load_image(event):
    image_url = event.get("image_url")
    if image_url is None:
        return "no image_url provided in event"
    print("INFO url {}".format(image_url))
    return util.read_image_pil(image_url, grayscale=True)
