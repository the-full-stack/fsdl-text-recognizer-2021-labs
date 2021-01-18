# Lab 1: Single-character prediction

## Before you begin, make sure to set up!

Please complete [Lab Setup](/setup.md) before proceeding!

## Lab 1: Basic project structure for recognizing MNIST

To get all of the project structure in place, let's train an MLP on MNIST data.

We show:

- Basic directory layout
- PyTorch MLP and LeNet models
- PyTorch-Lightning based training
- A single point of entry to running experiments: `python training/run_experiment.py`
  - `python training/run_experiment.py --max_epochs=10 --gpus='0,1' --accelerator=ddp --num_workers=20 --model_class=lenet.LeNet`
- Logs in Tensorboard: `tensorboard --logdir=training/logs`

Tasks:

Try different MLP architectures.
Paste your MLP architecture and the training output into Gradescope.

## Follow along

```
git pull
cd lab1/
```

## Directory Structure

### Data

There are three scopes of our code dealing with data, with slightly overlapping names.
Let's go through them from the top down to make sure we understand the pattern.

At the top level are `DataModule` classes.
They are responsible for:
- Downloading raw data from information specified in `data/raw/<dataset_name>/metadata.toml`  (or generating data, for synthetic datasets)
- Splitting data into train/val/test sets
- Processing data as needed to get it ready to go through PyTorch models
- Specifying data augmentation transforms to apply in training
- Specifying dimensions of the input data (e.g. `(C, H, W) float tensor` and any semantic information about the targets (e.g. a mapping for how an integer maps to characters)
- Wrapping underlying data in a `torch Dataset`, which simply returns individual data instances after optionally processing them with a transform function.
- Wrapping the `torch Dataset` in a `torch DataLoader`, which samples batches and delivers them to the GPU.

To avoid writing same old boilerplate for all of our data sources, we define a simple base class `text_recognizer.data.BaseDataModule` which in turn inherits from `pl.LightningDataModule`.
This inheritance will let us use the data very simply with PyTorch Lightning `Trainer` and avoid common problems with distributed training.

More details about how PyTorch deals with data are at https://pytorch.org/docs/stable/data.html.
More details about `LightningDataModule` are at https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

### Model

There are two scopes of our code dealing with models: `model`s and `lit_model`s.

## Intro to EMNIST

- EMNIST = Extended Mini-NIST :)
- All English letters and digits presented in the MNIST format.
- Look at: `notebooks/01-look-at-emnist.ipynb`

## Networks and training code

```
- text_recognizer/networks/mlp.py
- text_recognizer/networks/lenet.py
- text_recognizer/models/base.py
- text_recognizer/models/character_model.py
- training/util.py
```

## Train MLP and CNN

You can run the shortcut command `tasks/train_character_predictor.sh`, which runs the following:

```sh
training/run_experiment.py --save \
  '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp",  "train_args": {"batch_size": 256}}'
```

It will take a couple of minutes to train your model.

Just for fun, you could also try a larger MLP, with a smaller batch size:

```sh
training/run_experiment.py \
  '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 8}, "train_args": {"batch_size": 128}}'
```

## Testing

First, let's take a look at how the test works at

```
text_recognizer/tests/test_character_predictor.py
```

Now let's see if it works by running:

```sh
pytest -s text_recognizer/tests/test_character_predictor.py
```

Or, use the shorthand `tasks/test_functionality.sh`

Testing should finish quickly.
