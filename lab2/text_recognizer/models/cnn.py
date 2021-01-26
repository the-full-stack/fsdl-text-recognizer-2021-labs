from typing import Any, Dict
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMAGE_SIZE = 28


class CNN(nn.Module):
    """Simple CNN for recognizing characters in a square image."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)

        self.conv1 = nn.Conv2d(input_dims[0], conv_dim, 3, 1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim, 3, 1)
        self.dropout = nn.Dropout(0.25)

        # Each 3x3 conv subtracts 2 from the input. The 2x2 max-pool divides by 2. Flattening squares it.
        conv_output_size = (IMAGE_SIZE - 2 - 2) / 2
        fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
        self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # see also https://github.com/pytorch/pytorch/issues/18182
        for m in self.modules():
            if type(m) in {
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            }:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        B_, C_, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        return parser
