# Setup

## 1. Check out the repo

You should already have the repo in your home directory. Go into it and make sure you have the latest.

```sh
cd fsdl-text-recognizer-2021-labs
git pull origin master
```

If not, open a shell in your JupyterLab instance and run

```sh
git clone https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project.git
cd fsdl-text-recognizer-project
```

## 2. Set up the Python environment

We use `conda` for managing Python and CUDA versions, and `pip-tools` for managing Python package dependencies.

We add a `Makefile` for making setup dead-simple.

### First: Install the Python + CUDA environment using Conda (not necessary for GCP AI Platform)

**Note**: this step is not necessary if you are using the GCP AI Platform Notebooks instance.

 run `make conda-update` to create an environment called `fsdl-text-recognizer-2021`, as defined in `environment.yml`.
This environment will provide us with the right Python version as well as the CUDA and CUDNN libraries.

If you edit `environment.yml`, just run `make conda-update` again to get the latest changes.

Next, activate the conda environment.

```sh
conda activate fsdl-text-recognizer-2021
```

**IMPORTANT**: every time you work in this directory, make sure to start your session with `conda activate fsdl-text-recognizer-2021`.

### Next: install Python packages

Next, install all necessary Python packages by running `make pip-tools`

Using `pip-tools` lets us do three nice things:

1. Separate out dev from production dependencies (`requirements-dev.in` vs `requirements.in`).
2. Have a lockfile of exact versions for all dependencies (the auto-generated `requirements-dev.txt` and `requirements.txt`).
3. Allow us to easily deploy to targets that may not support the `conda` environment.

If you add, remove, or need to update versions of some requirements, edit the `.in` files, and simply run `make pip-tools` again.

### Set PYTHONPATH

Last, run ```export PYTHONPATH=.``` before executing any commands later on, or you will get errors like `ModuleNotFoundError: No module named 'text_recognizer'`.

In order to not have to set `PYTHONPATH` in every terminal you open, just add that line as the last line of the `~/.bashrc` file using a text editor of your choice (e.g. `nano ~/.bashrc`)

## Ready

Now you should be setup for the labs. The instructions for each lab are in readme files in their folders.

## Summary

- `environment.yml` specifies python and optionally cuda/cudnn
- `make conda-update` creates/updates the conda env
- `conda activate fsdl-text-recognizer-2021` activates the conda env
- `requirements/prod.in` and `requirements/dev.in` specify python package requirements
- `make pip-tools` resolves and install all Python packages
- add `export PYTHONPATH=.:$PYTHONPATH` to your `~/.bashrc` and `source ~/.bashrc`
