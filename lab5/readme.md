# Lab 5: Experiment Management

## Goals of this lab

- Introduce IAMLines handwriting dataset
- Make EMNISTLines look more like IAMLines with additional data augmentations
- Introduce Weights & Biases
- Run some `LineCNNTransformer` experiments on EMNISTLines, writing notes in W&B
- Start a hyper-parameter sweep

## Follow along

```
git pull
cd lab5
```

## IAMLines Dataset

- Look at `notebooks/03-look-at-iam-lines.ipynb`.
- Code is in `text_recognizer/data/iam_lines.py`, which depends on `text_recognizer/data/iam.py`

## Make EMNISTLines more like IAMLines

To make our synthetic data look more like the real data we want to get to, we need to introduce data augmentations.

- Look at `notebooks/02b-look-at-emnist-lines2.ipynb`
- Code is in `text_recognizer/data/emnist_lines2.py`

## Weights & Biases

Weights & Biases is an experiment tracking tool that ensures you never lose track of your progress.

- Keep track of all experiments in one place
- Easily compare runs
- Create reports to document your progress
- Look at results from the whole team

### Set up W&B

```
wandb init
```

You should see something like:

```
? Which team should we use? (Use arrow keys)
> your_username
Manual Entry
```

Select your username.

```
Which project should we use?
> Create New
```

Select `fsdl-text-recognizer-project`.

How to implement W&B in training code?

Look at `training/run_experiment.py`.

### Your first W&B experiment

```
wandb login

python training/run_experiment.py --max_epochs=100 --gpus='0,' --num_workers=20 --model_class=LineCNNTransformer --data_class=EMNISTLines --window_stride=8 --loss=transformer --wandb
```

You should see a W&B experiment link in the console output.
Click the link to see the progress of your training run.

## Configuring sweeps

Sweeps enable automated trials of hyper-parameters.
W&B provides built in support for running [sweeps](https://docs.wandb.com/library/sweeps).

We've setup an initial configuration file for sweeps in `training/emnist_lines_line_cnn_transformer_sweep.yaml`.
It performs a basic random search across 3 parameters.

There are lots of different [configuration options](https://docs.wandb.com/library/sweeps/configuration) for defining more complex sweeps.
Anytime you modify this configuration you'll need to create a sweep in wandb by running:

```bash
wandb sweep training/emnist_lines_line_cnn_transformer_sweep.yaml
```

```bash
wandb agent $SWEEP_ID
```

### Stopping a sweep

If you choose the **random** sweep strategy, the agent will run forever. Our **grid** search strategy will stop once all options have been tried. You can stop a sweep from the W&B UI, or directly from the terminal. Hitting CTRL-C once will prevent the agent from running a new experiment but allow the current experiment to finish. Hitting CTRL-C again will kill the current running experiment.

## Things to try

- Try to find a settings of hyperparameters for `LineCNNTransformer` (don't forget -- it includes `LineCNN` hyperparams) that trains fastest while reaching low CER
- Perhaps do that by running a sweep!
- Try some experiments with `LineCNNLSTM` if you want
- You can also experiment with
