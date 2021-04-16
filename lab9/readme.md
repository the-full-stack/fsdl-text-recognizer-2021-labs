# Lab 9: Web Deployment

## Goal of the lab

- First, we speed up our ParagraphTextRecognizer model with TorchScript
- Next, we wrap the model in a web app, and send it some requests
- We package up the web app and model as a Docker container, and run it that way
- Lastly, we prepare to deploy as a serverless function using AWS Lambda, getting it working locally

## Follow along

Let's update to the most recent version of the labs.

In this repository, do:

```
git pull
conda activate fsdl-text-recognizer-2021  # If on own machine
make pip-tools
cd lab9
```

## New files

This lab has quite a few new files, mostly in the new `api/` directory.

```
api_server/
├── app.py
├── Dockerfile
├── __init__.py
└── tests
    └── test_app.py
api_serverless/
├── app.py
├── Dockerfile
└── __init__.py
notebooks/
requirements/
text_recognizer/
training/
```

## First, TorchScript

To speed up inference, we can compile our trained PyTorch model using TorchScript.

Check out the minimal changes in `text_recognizer/paragraph_text_recognizer.py`:

```python
self.scripted_model = self.lit_model.to_torchscript(method="script", file_path=None)
...
y_pred = self.scripted_model(image_tensor.unsqueeze(axis=0))[0]
```

The conversion takes a few seconds, but then all inference calls are significantly sped up.

## Serving predictions from a web server

Now, let's get a Flask web server up and running and serving predictions.

Flask is a standard Python library for web servers.
If we were actually deploying to production using this method, we would use FastAPI, which works almost exactly the same but is more modern and is faster at responding to requests.

```
python api_server/app.py
```

Open up another terminal tab.
In this terminal, we'll send some test image to the web server we're running in the first terminal.

**Make sure to `cd` into the `lab9` directory in this new terminal.**

```
export API_URL=http://0.0.0.0:8000
(echo -n '{ "image": "data:image/png;base64,'$(base64 -w0 -i text_recognizer/tests/support/paragraphs/a01-077.png)'" }') | curl -X POST "${API_URL}/v1/predict" -H 'Content-Type: application/json' -d @-
```

What we did is send a base64-encoded image as a POST request to the web server.

If you want to look at the image you just sent, you can navigate to
`lab9/text_recognizer/tests/support/paragraphs` in the file browser.

In the POST request, we encoded the whole image.

We can instead send a GET request specifying a URL to an image as a query paramter of the API URL:
```
curl "${API_URL}/v1/predict?image_url=https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png"
```

Both ways are common.

You can shut down your web server now (Ctrl + C).

## Adding web server tests

The web server code should have a unit test just like the rest of our code.

Let's check it out: the tests are in `api_server/tests/test_app.py`.
You can run them with

```sh
pytest -s api_server/tests
```

## Running web server in Docker

What we have now, we *could* deploy.
As part of deployment, we would have to ensure that the Python version is correct, install dependencies, check out the whole repo, etc.
To make this simpler, we can build a Docker image with everything our application needs, and nothing it doesn't (like training code).

First off, if you don't already have `docker` installed on your system, do so: https://docs.docker.com/get-docker/
(You won't be able to follow this part on Google Colab, unfortunately.)

Still in the `lab9` directory, run:

```sh
cp ../requirements/prod.txt requirements.txt
docker build -t text-recognizer/api-server -f api_server/Dockerfile .
```

This should take a couple of minutes to complete.
While we wait, let's look at the Dockerfile in `api_server/Dockerfile`, which defines how we're building the Docker image.

When it's finished, you can run the server with

```sh
docker run -p8000:8000 -it --rm text-recognizer/api-server
```

You can run the same `curl` commands as you did when you ran the flask server earlier, and see that you're getting the same results.

```sh
curl "${API_URL}/v1/predict?image_url=https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png"
```

You can shut down your docker container now.

We could deploy this container to a number of platforms.

For example, you could deploy it very easily using https://render.com

## Serverless

Another way to deploy this app is as a serverless function.
Both AWS and GCP have support for this kind of deployment.
We will use AWS Lambda.

Check out `api_serverless/app.py` and `api_serverless/Dockerfile`.

We can build the container with

```sh
cp ../requirements/prod.txt requirements.txt
docker build -t text-recognizer/api-serverless -f api_serverless/Dockerfile .
```

Let's run the container with

```sh
docker run -p9000:8080 -it --rm text-recognizer/api-serverless
```

We can send GET requests to the serverless function running locally with

```sh
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{ "image_url": "https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png"}'
```

We can now easily deploy this to AWS Lambda and hook it up to images being uploaded to S3, for example.

Or, we can put a lightweight REST gateway in front of it, and also set up some basic monitoring -- as we will do in the next lab!

## Homework

Follow these instructions!
If you're able to, then you're able to get the basics of Docker and Python web servers.

For a deeper understanding, try using FastAPI instead of Flask.
