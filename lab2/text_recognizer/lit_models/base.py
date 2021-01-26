import pytorch_lightning as pl
import torch


class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.optimizer_class = getattr(torch.optim, args.optimizer)
        self.lr = args.lr
        if not args.loss == "transformer":
            self.loss_fn = getattr(torch.nn.functional, args.loss)
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--loss", type=str, default="cross_entropy", help="loss function from torch.nn.functional")
        return parser

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
