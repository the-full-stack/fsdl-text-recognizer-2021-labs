# Lab 8: Testing and Continuous Integration

## Goal of the lab

- Add linting
- Add prediction tests
- Add evaluation tests
- Set up continuous integration using CircleCI

## Follow along

Let's update to the most recent version of the labs.

In the repository, do

```
git pull
conda activate fsdl-text-recognizer-2021  # If on own machine
make pip-tools
cd lab8/
```

## New files

In the top-level directory, the new files are:

```
- .circleci/config.yml
```

And in the `lab8` directory, we added:

```
- .pylintrc
- pyproject.toml
- setup.cfg
- tasks/lint.sh
- text_recognizer/data/fake_images.py
- text_recognizer/evaluation/evaluate_paragraph_text_recognizer.py
- text_recognizer/tests/test_paragraph_text_recognizer.py
- text_recognizer/tests/support/paragraphs/a01-107.png
- text_recognizer/tests/support/paragraphs/a02-046.png
- text_recognizer/tests/support/paragraphs/data_by_file_id.json
- text_recognizer/tests/support/paragraphs/a01-087.png
- text_recognizer/tests/support/paragraphs/a01-077.png
- training/tests/test_run_experiment.sh
```

## Linting

Linting refers to automatically checking code files for style, documentation, and some basic bugs using static analysis.
Setting up a linting system is a little tricky at first, but pays off very quickly.
It is a must for any multi-developer codebase, in order to maintain a basic code quality and prevent endless back-and-forth about code style and conventions.

Running the new file `tasks/lint.sh` fully lints our codebase with a few different checkers:

- `safety` scans our Python package dependencies for known security vulnerabilities
- `pylint` does static analysis of Python files and reports both style and bug problems
- `pycodestyle` checks for simple code style guideline violations (somewhat overlapping with `pylint`)
- `pydocstyle` checks for docstring guideline violations
- `mypy` performs static type checking of Python files
- `bandit` finds common security vulnerabilities
- `shellcheck` finds bugs and potential bugs in shell scrips

A note: in writing Bash scripts, I often refer to [this excellent guide](http://redsymbol.net/articles/unofficial-bash-strict-mode/).

Note that the linters are configured using `.pylintrc` and `setup.cfg` files, as well as flags specified in `lint.sh`.

Lastly, we use the automated Python formatter `black` to make it even easier for us to be compatible with some linters and avoid arguing about style.
You can run it in the direcotry with `black .`, or better yet, configure your text editor to automatically run it every time you save a file: https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0

## Prediction tests

In Lab 7, we added `ParagraphTextRecognizer` which takes an image and returns the text recognized in it.

We now add a test for this module in `text_recognizer/tests/test_paragraph_text_recognizer.py`

The test simply runs several known images from `text_recognizer/tests/support` through `ParagraphTextRecognizer` and compare the results to expected results, stored in `text_recognizer/tests/support/paragraphs/data_by_file_id.json`.

To run this test (along with any other tests in `test_` files, along with docstring tests), simply run `pytest -s .`.

These tests shouldn't take more than a minute or so to run.

## Evaluation tests

In addition to the quick prediction tests, we want to evaluate performance of our current model on the test dataset.

This is a longer test, and benefits from having a GPU available.

To run, do `pytest -s text_recognizer/evaluation/evaluate_paragraph_text_recognizer.py`

## Training tests

Lastly, we want to have a test for our training system (`training/run_experiment.py`), to make sure it correctly reads command-line flags and doesn't fail in training.

To do this, we add `training/tests/test_run_experiment.sh`, which runs a quick experiment using a special `FakeImages` dataset (so we don't have to download any data).

## Setting up CircleCI

The linting, as well prediction and training tests, should be run every time we push code upstream to our repo.

We use CircleCI to do this, and so add a file outside of the `lab8` directory (in the top-level directory): `.circleci/config.yml`

To set this up:
- Fork our repository at https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs (click the Fork button in the top right)
- Go to https://circleci.com and log in with your Github account.
- Click on Add Project. Select your fork
- It will ask you to place the `config.yml` file in the repo. Since it's already there, you can just hit the "Start building" button.

While CircleCI starts the build, let's look at the `config.yml` file.
It simply downloads and installs requirements which are not already present on the circleci Python image, then runs linting via `tasks/lint.sh` and tests via `tasks/test.sh`.
If anything fails, CircleCI will mark the build as failing, and GitHub will alert us.

Now that CircleCI is done building, let's push a commit so that we can see it build again, and check out the nice green chechmark in our commit history.

## Homework

Fork the repo, set up CircleCI, and see it build.
