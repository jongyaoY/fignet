# MIT License
#
# Copyright (c) [2024] [Zongyao Yi]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List

import torch
import torch.nn as nn


def build_mlp(
    input_size: int,
    hidden_layer_sizes: List[int],
    output_size: int = None,
    output_activation: nn.Module = nn.Identity,
    activation: nn.Module = nn.ReLU,
):
    """Build a MultiLayer Perceptron.

    Args:
      input_size: Size of input layer.
      layer_sizes: An array of input size for each hidden layer.
      output_size: Size of the output layer.
      output_activation: Activation function for the output layer.
      activation: Activation function for the hidden layers.

    Returns:
      mlp: An MLP sequential container.
    """
    # Size of each layer
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    act = [activation for i in range(nlayers)]
    act[-1] = output_activation

    # Create a torch sequential container
    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module(
            "NN-" + str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1])
        )
        mlp.add_module("Act-" + str(i), act[i]())

    return mlp


class Encoder(nn.Module):
    """Graph network encoder"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        """
        Encoder containing a single MLP

        Args:
            in_size: Number of node input features (for 2D = 30, calculated
                as [10 = 5 times steps * 2 positions (x, y) +
                4 distances to boundaries (top/bottom/left/right) +
                16 particle type embeddings]).
            out_size: Number of node output features (latent dimension of
                size 128).
            nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
            mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

        """
        super(Encoder, self).__init__()
        # Encode node/edge features as an MLP
        self.mlp = nn.Sequential(
            *[
                build_mlp(
                    in_size,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    out_size,
                ),
                nn.LayerNorm(out_size),
            ]
        )

    def forward(self, x: torch.Tensor):
        """The forward hook runs when the Encoder class is instantiated

        Args:
            x (torch.Tensor): Features to encode

        Returns:
            torch.Tensor: Latent features
        """
        return self.mlp(x)


class Decoder(nn.Module):
    """The Decoder extracts the
    dynamics information from the nodes of the final latent graph

    """

    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        """Initializer

        Args:
            nnode_in (int): Input latent node features
            nnode_out (int): Output node features
            nmlp_layers (int): Number of MLP layers
            mlp_hidden_dim (int): Hidden dimension of the MLP
        """
        super(Decoder, self).__init__()
        self.node_fn = build_mlp(
            nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out
        )

    def forward(self, x: torch.tensor):
        """Forward hook

        Returns:
            torch.tensor: Decoded node features
        """
        return self.node_fn(x)
