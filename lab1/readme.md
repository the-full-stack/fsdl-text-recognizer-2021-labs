# Lab 1: Basic project structure for recognizing MNIST

To go through the main project structure in place, let's train an MLP on MNIST data.

We will cover:

- Project structure
- A simple PyTorch MLP model
- PyTorch-Lightning based training
- A single point of entry to running experiments: `python3 run_experiment.py`

## Before you begin, make sure to set up!

Please complete [Lab Setup](/setup/readme.md) before proceeding!

Then, in the `fsdl-text-recognizer-2021-labs` repo, let's pull the latest changes, and enter the lab1 directory.

```
git pull
cd lab1/
```

## Directory Structure

In our labs, we will be building a code base incrementally.
Each week, a new lab will be released, showing more of the codebase.

Today, we start with the bare minimum:

```sh
(fsdl-text-recognizer-2021) ➜  lab1 git:(main) ✗ tree -I "logs|admin|wandb|__pycache__"
.
├── readme.md
├── text_recognizer
│   ├── data
│   │   ├── base_data_module.py
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   └── util.py
│   ├── __init__.py
│   ├── lit_models
│   │   ├── base.py
│   │   └── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   └── mlp.py
│   └── util.py
└── training
    ├── __init__.py
    └── run_experiment.py
```

We can see that the main breakdown of the codebase is between `text_recognizer` and `training`.

The former, `text_recognizer`, should be thought of as a Python package that we are developing and will eventually deploy in some way.

The latter, `training`, should be thought of as support code for developing `text_recognizer`, which currently consists simply of `run_experiment.py`.

Within `text_recognizer`, there is further breakdown between `data`, `models`, and `lit_models`.

Let's go through them in sequence.

### Data

There are three scopes of our code dealing with data, with slightly overlapping names: `DataModule`, `DataLoader`, and `Dataset`.

At the top level are `DataModule` classes, which are responsible for quite a few things:

- Downloading raw data and/or generating synthetic data
- Processing data as needed to get it ready to go through PyTorch models
- Splitting data into train/val/test sets
- Specifying dimensions of the inputs (e.g. `(C, H, W) float tensor`
- Specifying information about the targets (e.g. a class mapping)
- Specifying data augmentation transforms to apply in training

In the process of doing the above, `DataModule`s make use of a couple of other classes:

1. They wrap underlying data in a `torch Dataset`, which returns individual (and optionally, transformed) data instances.
2. They wrap the `torch Dataset` in a `torch DataLoader`, which samples batches, shuffles their order, and delivers them to the GPU.

If need be, you can read more about these [PyTorch data interfaces](https://pytorch.org/docs/stable/data.html).

To avoid writing same old boilerplate for all of our data sources, we define a simple base class `text_recognizer.data.BaseDataModule` which in turn inherits from [`pl.LightningDataModule`](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html).
This inheritance will let us use the data very simply with PyTorch-Lightning `Trainer` and avoid common problems with distributed training.

### Models

Models are what is commonly known as "neural nets": code that accepts an input, processes it through layers of computations, and produces an output.

Most importantly, the code is partially written (the architecture of the neural net), and partially **learned** (the parameters, or weights, of all the layers in the architecture).
Therefore, the computation of the model must be back-propagatable.

Since we are using PyTorch, all of our models sublcass `torch.nn.Module`, which makes them learnable in this way.

### Lit Models

We use PyTorch-Lightning for training, which defines the `LightningModule` interface that handles not only everything that a Model (as defined above) handles, but also specifies the details of the learning algorithm: what loss should be computed from the output of the model and the ground truth, which optimizer should be used, with what learning rate, etc.

## Training

Now we understand enough to train.

Our `training/run_experiment.py` is a script that handles many command-line parameters.

Here's a command we can run:

```sh
python3 training/run_experiment.py --model_class=MLP --data_class=MNIST --max_epochs=5 --gpus=1
```

While `model_class` and `data_class` are our own arguments, `max_epochs` and `gpus` are arguments automatically picked up from `pytorch_lightning.Trainer`.
You can use any other `Trainer` flag (see [docs](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags)) on the command line, for example `--batch_size=512`.

The `run_experiment.py` script also picks up command-line flags from the model and data classes that are specified.
For example, in `text_recognizer/models/mlp.py` we specify the `MLP` class, and add a couple of command-line flags: `--fc1` and `--fc2`.

Accordingly, we can run

```sh
python3 training/run_experiment.py --model_class=MLP --data_class=MNIST --max_epochs=5 --gpus=1 --fc1=4 --fc2=8
```

And watch the model fail to achieve high accuracy due to too few parameters :)

## Homework

- Try `training/run_experiment.py` with different MLP hyper-parameters (e.g. `--fc1=128 --fc2=64`).
- Try editing the MLP architecture in `text_recognizers/models/mlp.py`
