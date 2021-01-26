# Lab 2: Convolutional Neural Nets and Synthetic Lines

We're working up to translating images of handwriting to text.
In this lab, we're going to make synthetic data using the EMNIST dataset, and use Convolutional Neural Networks to first recognize EMNIST characters, and then translate whole lines.
However, our approach will be brittle, and will fail when spacing between characters in a line is not exact.

- EMNIST dataset
- Synthetic fixed-length lines: `EMNISTLines` dataset
- `LineReshapeCNN` model will succeed
- Synthetic variable-length lines: `EMNISTLines` dataset
- `LineReshapeCNN` will fail

## Goal of the lab

- Use a simple convolutional network to recognize EMNIST characters.
- Construct a synthetic dataset of EMNIST lines.
- Move from reading single characters to reading lines.

## Before you begin, make sure to set up!

Please complete [Lab Setup](/setup/readme.md) before proceeding!

Then, in the `fsdl-text-recognizer-2021-labs` repo, let's pull the latest changes, and enter the correct directory.

```
git pull
cd lab2
```

## Intro to EMNIST

- EMNIST = Extended Mini-NIST :)
- All English letters and digits presented in the MNIST format.
- Look at: `notebooks/01-look-at-emnist.ipynb`

Note that we now have a new directory in `lab2`: `notebooks`.
While we don't do training of our models in notebooks, we use them for exploring the data, and perhaps presenting the results of our model training.

### Brief aside: data directory structure

```
(fsdl-text-recognizer-2021) ➜  lab2 git:(main) ✗ tree -I "lab*|__pycache__" ..
..
├── data
│   ├── downloaded
│   └── raw
│       ├── emnist
│       │   ├── metadata.toml
│       │   └── readme.md
├── environment.yml
├── Makefile
├── readme.md
├── requirements
└── setup
```

We specify the EMNIST dataset with `metadata.toml` and `readme.md` which contain information on how it should be downloaded and its provenance.

## Using a convolutional network for recognizing MNIST

We left off in Lab 1 having trained an MLP model on the MNIST digits dataset.

We can train a CNN for the same purpose:

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=MNIST --max_epochs=5 --gpus=1
```

## Doing the same for EMNIST

We can do the same on the larger EMNIST dataset:

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=5 --gpus=1
```

Training the single epoch will take about 2 minutes (that's why we only do one epoch in this lab :)).
Leave it running while we go on to the next part.

## Intentional overfitting

It is very useful to be able to subsample the dataset for quick experiments and to make sure that the model is robust enough to represent the data (more on this in the Training & Debugging lecture).

This is possible by passing `--overfit_batches=0.01` (or some other fraction).
You can also provide an int `> 1` instead for a concrete number of batches.
https://pytorch-lightning.readthedocs.io/en/stable/debugging.html#make-model-overfit-on-subset-of-data

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=50 --gpus=1 --overfit_batches=2
```

## Speeding up training

One way we can make sure that our GPU stays consistently highly utilized is to do data pre-processing in separate worker processes, using the `--num_workers=X` flag.

```sh
python3 training/run_experiment.py --model_class=CNN --data_class=EMNIST --max_epochs=50 --gpus=1 --num_workers=4
```

## Making a synthetic dataset of EMNIST Lines

- Synthetic dataset we built for this project
- Sample sentences from Brown corpus
- For each character, sample random EMNIST character and place on a line (optionally, with some random overlap)
- Look at: `notebooks/02-look-at-emnist-lines.ipynb`

## Reading multiple characters at once

Now that we have a dataset of lines and not just single characters, we can apply our convolutional net to it.

Let's look at `notebooks/02b-cnn-for-simple-emnist-lines.ipynb`, where we generate a dataset with at most 8 characters and no overlap.

## LineReshapeCNN

The first model we will try is a simple wrapper around `CNN` that applies it to each square slice of the input image in sequence: `LineReshapeCNN`

We can train this with

```sh
python training/run_experiment.py --max_epochs=5 --gpus=1 --num_workers=4 --data_class=EMNISTLines --max_length=16 --max_overlap=0 --model_class=LineReshapeCNN
```

We can easily get to >90% accuracy.

## LineCNN

The second model we will try is `LineCNN`: a fully-convolutional model.

We can train it with

```sh
python training/run_experiment.py --max_epochs=5 --gpus=1 --num_workers=4 --data_class=EMNISTLines --max_length=16 --max_overlap=0 --model_class=LineCNN
```

We can easily get to >90% accuracy.

## Decreasing accuracy on overlap

However, our models will fail when presented with text that is not uniformly spaced:

```sh
python training/run_experiment.py --max_epochs=5 --gpus=1 --num_workers=4 --data_class=EMNISTLines --max_length=16 --max_overlap=0.33 --model_class=LineCNN
```

This only gets around 80% accuracy.
