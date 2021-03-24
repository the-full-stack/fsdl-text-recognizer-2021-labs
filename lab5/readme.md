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

If you're on Colab, there are a couple of new packages to install, so make sure to run

```
!pip install boltons pytorch_lightning==1.1.4 wandb
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

If you're on Colab, make sure to install the `wandb` package first with `!pip install wandb`.

Then, run `wandb init`.

You've likely never signed into W&B before, so you'll see something like:

```text
Let's setup this directory for W&B!
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter:
```

Clicking on that URL will allow you to Sign In via Github (recommended) or Google, or Sign Up for an account.

The API key that you're being asked for will be available here: https://wandb.ai/settings

Then you'll be asked to enter the name of your project.
You can go with the name of our repo: `fsdl-text-recognizer-2021-labs`

### W&B Integration code

Check out `training/run_experiment.py` for the few lines of code that it takes to integrate this logger with our experiment-running setup.

### Your first W&B experiment

```
python training/run_experiment.py --wandb --max_epochs=10 --gpus='0,' --num_workers=4 --data_class=EMNISTLines2 --model_class=LineCNNTransformer --loss=transformer
```

You should see a W&B experiment link in the console output, something like this:

```text
wandb: Tracking run with wandb version 0.10.14
wandb: Syncing run rosy-shape-1
wandb: ⭐️ View project at https://wandb.ai/USER/PROJECT
wandb: 🚀 View run at https://wandb.ai/USER/PROJECT/runs/35hje5h9
```

Click the link to see the progress of your training run.

Note that you can navigate to a table of all of your runs, and add notes and tags to stay organized.

Adding a brief note describing what this experiment is about is invaluable to your future self :)

## Configuring sweeps

Now that we can run experiments that are stored on Weights & Biases, we can more quickly find a good operating point, where your model is converging to something.

Then, we'd like to search a bit around that point -- this is often called hyper-parameter optimization.

W&B provides built in support for running [sweeps](https://docs.wandb.com/library/sweeps) of such experiments.

We've setup an initial configuration file for sweeps in `training/emnist_lines_line_cnn_transformer_sweep.yaml`.

Start a sweep server on W&B by running:

```bash
wandb sweep training/sweeps/emnist_lines2_line_cnn_transformer.yml
```

Now you can run the sweep agent by executing the command that's printed out, for example:

```bash
wandb agent USER/PROJECT/SWEEPID
```

A nice trick in case you have multiple GPUs and want to run multiple experiments in parallel is to precede the command with `CUDA_VISIBLE_DEVICES=<GPU_IND>`.
For example, you can run the following two commands in two terminal sessions to have two experiments going at once, each using one of your GPUs:

```
# In one terminal session:
CUDA_VISIBLE_DEVICES=0 wandb agent USER/PROJECT/SWEEPID

# In another terminal session:
CUDA_VISIBLE_DEVICES=1 wandb agent USER/PROJECT/SWEEPID
```

### Stopping a sweep

The **grid** search strategy will stop once all options have been tried. If you choose the **random** sweep strategy, the agent will run forever (although this may be changing in a future version).

You can also stop a sweep from the W&B web app UI, or directly from the terminal. Hitting CTRL-C once will prevent the agent from running a new experiment but allow the current experiment to finish. Hitting CTRL-C again will kill the current running experiment.

## Things to try

- Try to find a settings of hyperparameters for `LineCNNTransformer` (don't forget -- it includes `LineCNN` hyperparams) that trains fastest while reaching low CER
- Perhaps do that by running a sweep!
- Try some experiments with `LineCNNLSTM` if you want
