"""Experiment-running framework."""
import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import wandb

from text_recognizer.lit_models import base


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[
        1
    ].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--data_class", type=str, default="mnist.MNIST")
    parser.add_argument("--model_class", type=str, default="mlp.MLP")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"text_recognizer.data.{temp_args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    base.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=mlp.MLP --data_class=mnist.MNIST
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"text_recognizer.data.{args.data_class}")
    model_class = _import_class(f"text_recognizer.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    if args.loss not in ("ctc", "transformer"):
        lit_model_class = base.BaseLitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(
            args.load_checkpoint, args=args, model=model
        )
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}",
        monitor="val_loss",
        mode="min",
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, logger=logger, weights_save_path="training/logs"
    )

    # pylint: disable=no-member
    trainer.tune(
        lit_model, datamodule=data
    )  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member


if __name__ == "__main__":
    main()
