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


from typing import Dict

import torch
import torch.nn as nn

from fignet.data import HeteroGraph
from fignet.message_passing import EncodeProcessDecode
from fignet.normalization import Normalizer

# from fignet.types import KinematicType, MOEdge, NodeType, OMEdge


class LearnedSimulator(nn.Module):
    def __init__(
        self,
        mesh_dimensions: int,
        latent_dim: int,
        message_passing_steps: int,
        mlp_layers: int,
        mlp_hidden_dim: int,
        device="cpu",
        leave_out_mm: bool = False,
    ):
        """Initializer

        Args:
            mesh_dimensions (int): Mesh position dimension, 3 for 3d meshes
            latent_dim (int): Latent feature dimension
            nmessage_passing_steps (int): Number of message passing steps
            nmlp_layers (int): Number of MLP layers mlp_hidden_dim (int):
            Hidden MLP dimension
            device (str, optional): Defaults to "cpu".
            leave_out_mm (bool): Ignore mm edges
        """
        super(LearnedSimulator, self).__init__()

        self._mesh_dimensions = mesh_dimensions
        assert self._mesh_dimensions == 3

        self._is_initialized = False

        self._node_normalizers: Dict[str, Normalizer] = {}
        self._edge_normalizers: Dict[str, Normalizer] = {}
        self._output_normalizer: Normalizer = None
        self._device = device
        self._gnn_params = {
            "latent_dim": latent_dim,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            "message_passing_steps": message_passing_steps,
        }

    @property
    def initialized(self):
        return self._is_initialized

    def init(self, graph: HeteroGraph):
        """Lazy initializer"""
        node_dim_dict = graph.num_node_features
        edge_dim_dict = graph.num_edge_features
        # self.graph_meta = graph.metadata
        self._node_normalizers = {}
        self._edge_normalizers = {}
        for node_type, node_dim in node_dim_dict.items():
            self._node_normalizers[node_type] = Normalizer(
                size=node_dim,
                name="_".join([node_type, "node_normalizer"]),
                device=self._device,
            )
        for edge_type, edge_dim in edge_dim_dict.items():
            edge_type_name = edge_type[1]
            self._edge_normalizers[edge_type] = Normalizer(
                size=edge_dim,
                name="_".join([edge_type_name, "edge_normalizer"]),
                device=self._device,
            )
        self._output_normalizer = Normalizer(
            size=self._mesh_dimensions,
            name="output_normalizer",
            device=self._device,
        )
        input_dim_dict = {}
        input_dim_dict.update(node_dim_dict)
        input_dim_dict.update(edge_dim_dict)
        output_dim_dict = {}
        for node_type in node_dim_dict.keys():
            output_dim_dict.update({node_type: self._mesh_dimensions})
        self._encode_process_decode = EncodeProcessDecode(
            input_dim_dict=input_dim_dict,
            output_dim_dict=output_dim_dict,
            **self._gnn_params,
        )

        self._is_initialized = True

    def denormalize_accelerations(
        self,
        acclerations: torch.Tensor,
    ):
        """De-normalize accelerations for calculating future mesh positions

        Args:
            acclerations (torch.Tensor): Output from the networks

        Returns:
            torch.Tensor: De-normalized accelerations
        """
        return self._output_normalizer.inverse(acclerations)

    def normalize_accelerations(
        self,
        accelerations: torch.Tensor,
    ):
        """Normalize target acceleration, since the loss is calculated in the
        normalized space. If no statistics is provided, mean and std are
        calculated with training data

        Args:
            accelerations (torch.Tensor): Target acceleration to be normalized
            stats (Optional[dict]): Optional statistics of the acceleration.
            Defaults to None.

        Returns:
            torch.Tensor: Normalized target acceleration
        """
        return self._output_normalizer(accelerations, self.training)

    def _encoder_preprocessor(
        self,
        input: HeteroGraph,
    ):
        """Preprocess input including normalization and adding noise

        Args:
            input (dict): Input graph

        Returns:
            dict: Preprocessed graph
        """
        for k in input.x_dict.keys():
            input[k].x = self._node_normalizers[k](input[k].x)
        for k in input.edge_attr_dict.keys():
            if input[k].edge_index.shape[1] > 0:
                input[k].edge_attr = self._edge_normalizers[k](
                    input[k].edge_attr
                )

        return input

    def predict_accelerations(
        self,
        input: HeteroGraph,
    ):
        """Predict mesh and object node next accelerations (velocity change)

        Args:
            input (Dict[str, torch.Tensor]): Input graph

        Returns:
            torch.Tensor: Mesh node accelerations
            torch.Tensor: Object node accelerations
        """
        graph = self._encoder_preprocessor(input)
        m_pred_acc, o_pred_acc = self._encode_process_decode(graph)

        return m_pred_acc, o_pred_acc

    def save(self, path: str = "model.pt"):
        """Save model state

        Args:
            path: Model path
        """
        # TODO
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        # _node_normalizer = self._node_normalizer.get_variable()
        # _mo_edge_normalizer = self._mo_edge_normalizer.get_variable()
        # _om_edge_normalizer = self._om_edge_normalizer.get_variable()

        save_data = {
            "model": model,
            "_output_normalizer": _output_normalizer,
            # "_node_normalizer": _node_normalizer,
            # "_mo_edge_normalizer": _mo_edge_normalizer,
            # "_om_edge_normalizer": _om_edge_normalizer,
        }
        # for
        torch.save(save_data, path)

    def load(self, path: str):
        """Load model state from file

        Args:
            path: Model path
        """
        dicts = torch.load(path, map_location=self._device)
        self.load_state_dict(dicts["model"])

        keys = list(dicts.keys())
        keys.remove("model")

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval("self." + k)
                if isinstance(value, torch.Tensor):
                    setattr(object, para, value.to(self._device))
                else:
                    setattr(object, para, value)

                object.to(self._device)

        print("Simulator model loaded checkpoint %s" % path)
