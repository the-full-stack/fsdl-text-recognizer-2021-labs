from typing import Any, Dict
import argparse
import torch
import torch.nn as nn

from .cnn import CNN

CHAR_SIZE = 28


class LineReshapeCNN(nn.Module):
    """LeNet based model that takes a line of width that is a multiple of CHAR_WIDTH."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.num_classes = len(data_config["mapping"])
        self.cnn = CNN(data_config=data_config, args=args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parametersz
        ----------
        x
            (B, 1, H, W) input image

        Returns
        -------
        torch.Tensor
            (B, C, S) logits, where S is the length of the sequence and C is the number of classes
            S can be computed from W and CHAR_WIDTH
            C is self.num_classes
        """
        B, _C, H, _W = x.shape
        assert H == CHAR_SIZE

        # Reshape x to (B, S, H, W)
        # note that characters are square, so height = width
        x = x.squeeze()  # -> (B, H, W)
        x = x.permute(0, 2, 1)  # -> (B, W, H)
        x = x.view(B, -1, CHAR_SIZE, H)  # -> (B, S, w, H)
        x = x.permute(0, 1, 3, 2)  # -> (B, S, H, w)
        B, S, H, _w = x.shape

        # To debug, we can simply return x and inspect it
        # return x

        # NOTE: type_as properly sets device
        activations = torch.zeros((B, self.num_classes, S)).type_as(x)
        for s in range(S):
            window = x[:, s : s + 1, :, :]  # -> (B, 1, H, _w)
            activations[:, :, s] = self.cnn(window)
        return activations

    @staticmethod
    def add_to_argparse(parser):
        CNN.add_to_argparse(parser)
        return parser
