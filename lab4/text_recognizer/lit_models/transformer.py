import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import CharacterErrorRate


class TransformerLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.

    The module must take x, y as inputs, and have a special predict() method.
    """

    def __init__(self, args, model):
        super().__init__()
        self.model = model

        inverse_mapping = {val: ind for ind, val in enumerate(self.model.data_config["mapping"])}
        start_index = inverse_mapping["<S>"]
        padding_index = inverse_mapping["<P>"]

        self.optimizer_class = getattr(torch.optim, args.optimizer)
        self.lr = args.lr
        self.loss_fn = nn.CrossEntropyLoss()  # Tried (ignore_index=padding_index), but it made CER much worse
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        ignore_tokens = [start_index, padding_index]
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=1e-3)
        return parser

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("val_loss", loss, prog_bar=True)

        pred = self.model.predict(x)
        self.val_acc(pred, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_cer(pred, y)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        pred = self.model.predict(x)
        self.test_acc(pred, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_cer(pred, y)
        self.log("test_cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
