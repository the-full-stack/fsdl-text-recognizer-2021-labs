"""
How to make inference faster: https://pgresia.medium.com/making-pytorch-transformer-twice-as-fast-on-sequence-generation-2a8a7f1e7389
"""
import argparse
from typing import Any, Dict
import math
import torch
import torch.nn as nn

from .line_cnn import LineCNN


TRANSFORMER_DIM = 128
TRANSFORMER_NHEAD = 4
TRANSFORMER_LAYERS = 4


class PositionalEncoding(nn.Module):
    """Classic Attention-is-all-you-need positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class LineCNNTransformer(nn.Module):
    """Process the line through a CNN and process the resulting sequence with a Transformer decoder"""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.input_dims = data_config["input_dims"]
        self.num_classes = len(data_config["mapping"])
        inverse_mapping = {val: ind for ind, val in enumerate(data_config["mapping"])}
        self.start_token = inverse_mapping["<S>"]
        self.padding_token = inverse_mapping["<P>"]
        self.max_output_length = data_config["output_dims"][0]
        self.args = vars(args) if args is not None else {}

        self.dim = self.args.get("transformer_dim", TRANSFORMER_DIM)
        nhead = self.args.get("transformer_nhead", TRANSFORMER_NHEAD)
        num_layers = self.args.get("transformer_layers", TRANSFORMER_LAYERS)

        # Instantiate LineCNN with "num_classes" set to self.dim
        data_config_for_line_cnn = {**data_config}
        data_config_for_line_cnn["mapping"] = list(range(self.dim))
        self.line_cnn = LineCNN(data_config=data_config_for_line_cnn, args=args)
        # LineCNN outputs (B, E, S) log probs, with E == dim

        self.embedding = nn.Embedding(self.num_classes, self.dim)
        self.fc = nn.Linear(self.dim, self.num_classes)

        self.pos_encoder = PositionalEncoding(d_model=self.dim)

        self.y_mask = generate_square_subsequent_mask(self.max_output_length)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.dim, nhead=nhead, dim_feedforward=self.dim),
            num_layers=num_layers,
        )

        self.init_weights()  # This is empirically important

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) image
        y
            (B, Sy) with elements in [0, C-1] where C is num_classes

        Returns
        -------
        torch.Tensor
            (B, C, Sy) logits
        """
        x = self.line_cnn(x)  # (B, E, Sx)
        x = x * math.sqrt(self.dim)

        x = x.permute(2, 0, 1)  # (Sx, B, E)
        x = self.pos_encoder(x)  # (Sx, B, E)

        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.dim)  # (Sy, B, E)
        y = self.pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(x)
        # TODO: can add tgt_key_padding_mask mask
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html

        output = self.transformer_decoder(y, x, y_mask)  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, C)
        return output.permute(1, 2, 0)  # (B, C, Sy)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, H, W) image

        Returns
        -------
        torch.Tensor
            (B, Sy) with elements in [0, C-1] where C is num_classes
        """
        B = x.shape[0]

        x = self.line_cnn(x)  # (B, E, Sx)
        x = x * math.sqrt(self.dim)

        x = x.permute(2, 0, 1)  # (Sx, B, E)
        x = self.pos_encoder(x)  # (Sx, B, E)

        y_mask = self.y_mask.type_as(x)
        output_tokens = (
            (torch.ones((self.max_output_length, B)) * self.padding_token).type_as(x).long()
        )  # (max_length, B)
        output_tokens[0] = self.start_token  # Set start token
        for Sy in range(1, self.max_output_length):
            y = self.embedding(output_tokens[:Sy]) * math.sqrt(self.dim)  # (Sy, B, E)
            y = self.pos_encoder(y)  # (Sy, B, E)
            output = self.transformer_decoder(y, x, y_mask[:Sy, :Sy])  # (Sy, B, E)
            output = self.fc(output)  # (Sy, B, C)
            output = torch.argmax(output, dim=-1)  # (Sy, B)
            output_tokens[Sy] = output[-1:]  # Set the last output token
            # TODO: stop decoding on the first '<P>' token (or better yet, the end token) to speed things up
        return output_tokens.T  # (B, Sy)

    @staticmethod
    def add_to_argparse(parser):
        LineCNN.add_to_argparse(parser)
        return parser
