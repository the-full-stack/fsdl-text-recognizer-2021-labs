from typing import Any, Dict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMAGE_SIZE = 28
NUM_RES_BLOCK = 1

class ConvBlock(nn.Module):
    """
    3x3 conv followed by Batch Normalization and a ReLU.
    """

    def __init__(
        self, 
        input_channels: int, 
        output_channels: int, 
        kernel_size: int = 3, 
        padding: int = 1, 
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu1 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x


class ResBlock(nn.Module):
    """
    Residual Block.
    """

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(input_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu2(out)
        return out


class CNN(nn.Module):
    """Simple CNN for recognizing characters in a square image."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        num_res_block = self.args.get("num_res_block", NUM_RES_BLOCK)
        self.features = [ConvBlock(input_dims[0], conv_dim, kernel_size=3, padding=1)]
        for _ in range(num_res_block):
            self.features.append(ResBlock(conv_dim))

        self.features.append(nn.Dropout(0.25))
        self.features.append(ConvBlock(conv_dim, conv_dim, kernel_size=2, padding=0, stride=2))
        self.features = nn.Sequential(*self.features)

        # Because our 3x3 convs have padding size 1, they leave the input size unchanged.
        # The 2x2 max-pool divides the input size by 2. Flattening squares it.
        # conv_output_size = IMAGE_SIZE // 2
        # fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(conv_dim, num_classes)
        # self.classifier = nn.Sequential(
        #    nn.Linear(fc_input_dim, fc_dim),
        #    nn.Linear(fc_dim, num_classes),
        #)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--num_res_block", type=int, default=NUM_RES_BLOCK)
        return parser
