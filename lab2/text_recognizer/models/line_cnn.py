from typing import Any, Dict
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_DIM = 64
FC_DIM = 128
WINDOW_WIDTH = 28
WINDOW_STRIDE = 28


class LineCNN(nn.Module):
    """
    Model that uses a simple CNN to process an image of a line of characters with a window, outputting a sequence of logits.
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super(LineCNN, self).__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}
        self.num_classes = len(data_config["mapping"])

        _C, H, _W = data_config["input_dims"]
        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        self.WW = self.args.get("window_width", WINDOW_WIDTH)
        self.WS = self.args.get("window_stride", WINDOW_STRIDE)

        # Input is (1, H, W)
        self.conv1 = nn.Conv2d(1, conv_dim, 3, 1, 1)  # -> (CONV_DIM, H, W)
        self.conv2a = nn.Conv2d(conv_dim, conv_dim, 3, 1, 1)  # -> (CONV_DIM, H, W)
        self.conv2b = nn.Conv2d(conv_dim, conv_dim, 3, 1, 1)  # -> (CONV_DIM, H, W)
        self.conv3 = nn.Conv2d(conv_dim, conv_dim, 3, 2, 1)  # -> (CONV_DIM, H // 2, W // 2)
        self.conv4a = nn.Conv2d(conv_dim, conv_dim, 3, 1, 1)  # -> (CONV_DIM, H // 2, W // 2)
        self.conv4b = nn.Conv2d(conv_dim, conv_dim, 3, 1, 1)  # -> (CONV_DIM, H // 2, W // 2)
        # Conv math! https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # OW = torch.floor((W // 2 - WW // 2) + 1)
        self.conv5 = nn.Conv2d(
            conv_dim, conv_dim, (H // 2, self.WW // 2), (H // 2, self.WS // 2)
        )  # -> (CONV_DIM, 1, OW)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(conv_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.num_classes)

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
            (B, 1, H, W) input image

        Returns
        -------
        torch.Tensor
            (B, C, S) logits, where S is the length of the sequence and C is the number of classes
            S can be computed from W and self.window_width
            C is self.num_classes
        """
        _B, _C, _H, W = x.shape
        x = F.relu(self.conv1(x))  # -> (B, CONV_DIM, H, W)
        x = F.relu(self.conv2b(F.relu(self.conv2a(x))) + x)  # -> (B, CONV_DIM, H, W)
        x = F.relu(self.conv3(x))  # -> (B, CONV_DIM, H//2, W//2)
        x = F.relu(self.conv4b(F.relu(self.conv4a(x))) + x)  # -> (B, CONV_DIM, H//2, W//2)
        OW = math.floor((W // 2 - self.WW // 2) / (self.WS // 2) + 1)
        x = F.relu(self.conv5(x))  # -> (B, CONV_DIM, 1, OW)
        assert x.shape[-1] == OW
        x = x.squeeze().permute(0, 2, 1)  # -> (B, OW, CONV_DIM)
        x = F.relu(self.fc1(x))  # -> (B, OW, FC_DIM)
        x = self.dropout(x)
        x = self.fc2(x)  # -> (B, OW, self.C)
        x = x.permute(0, 2, 1)  # -> (B, self.C, OW)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument(
            "--window_width",
            type=int,
            default=WINDOW_WIDTH,
            help="Width of the window that will slide over the input image.",
        )
        parser.add_argument(
            "--window_stride",
            type=int,
            default=WINDOW_STRIDE,
            help="Stride of the window that will slide over the input image.",
        )
        return parser
