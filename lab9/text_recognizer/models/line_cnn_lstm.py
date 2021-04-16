from typing import Any, Dict
import argparse
import torch
import torch.nn as nn

from .line_cnn import LineCNN

LSTM_DIM = 512
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.2


class LineCNNLSTM(nn.Module):
    """Process the line through a CNN and process the resulting sequence through LSTM layers."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}

        num_classes = len(data_config["mapping"])
        lstm_dim = self.args.get("lstm_dim", LSTM_DIM)
        lstm_layers = self.args.get("lstm_layers", LSTM_LAYERS)
        lstm_dropout = self.args.get("lstm_dropout", LSTM_DROPOUT)

        self.line_cnn = LineCNN(data_config=data_config, args=args)
        # LineCNN outputs (B, C, S) log probs, with C == num_classes

        self.lstm = nn.LSTM(
            input_size=num_classes,
            hidden_size=lstm_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(lstm_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) input image

        Returns
        -------
        torch.Tensor
            (B, C, S) logits, where S is the length of the sequence and C is the number of classes
            S can be computed from W and CHAR_WIDTH
            C is num_classes
        """
        x = self.line_cnn(x)  # -> (B, C, S)
        B, _C, S = x.shape
        x = x.permute(2, 0, 1)  # -> (S, B, C)

        x, _ = self.lstm(x)  # -> (S, B, 2 * H) where H is lstm_dim

        # Sum up both directions of the LSTM:
        x = x.view(S, B, 2, -1).sum(dim=2)  # -> (S, B, H)

        x = self.fc(x)  # -> (S, B, C)

        return x.permute(1, 2, 0)  # -> (B, C, S)

    @staticmethod
    def add_to_argparse(parser):
        LineCNN.add_to_argparse(parser)
        parser.add_argument("--lstm_dim", type=int, default=LSTM_DIM)
        parser.add_argument("--lstm_layers", type=int, default=LSTM_LAYERS)
        parser.add_argument("--lstm_dropout", type=float, default=LSTM_DROPOUT)
        return parser
