"""Flask web server serving text_recognizer predictions."""
import os
import logging

from flask import Flask, request, jsonify
from PIL import ImageStat

from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU

app = Flask(__name__)  # pylint: disable=invalid-name
model = ParagraphTextRecognizer()
logging.basicConfig(level=logging.INFO)

@app.route("/")
def index():
    """Provide simple health check route."""
    return "Hello, world!"


@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    image = _load_image()
    pred = model.predict(image)
    image_stat = ImageStat.Stat(image)
    logging.info("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
    logging.info("METRIC image_area {}".format(image.size[0] * image.size[1]))
    logging.info("METRIC pred_length {}".format(len(pred)))
    logging.info("pred {}".format(pred))
    return jsonify({"pred": str(pred)})


def _load_image():
    if request.method == "POST":
        data = request.get_json()
        if data is None:
            return "no json received"
        return util.read_b64_image(data["image"], grayscale=True)
    if request.method == "GET":
        image_url = request.args.get("image_url")
        if image_url is None:
            return "no image_url defined in query string"
        logging.info("url {}".format(image_url))
        return util.read_image_pil(image_url, grayscale=True)
    raise ValueError("Unsupported HTTP method")


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=False)  # nosec


if __name__ == "__main__":
    main()
