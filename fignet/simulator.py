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


import torch
import torch.nn as nn

from fignet.data import HeteroGraph
from fignet.message_passing import EncodeProcessDecode, MOEdge, OMEdge
from fignet.normalization import Normalizer
from fignet.types import KinematicType, NodeType


class LearnedSimulator(nn.Module):
    def __init__(
        self,
        mesh_dimensions: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        input_seq_length: int = 3,
        property_dim: int = 5,
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
        self._leave_out_mm = leave_out_mm
        # Initialize the EncodeProcessDecode
        self._num_node_types = len(NodeType)

        # vel, kin, properties
        node_dim = (
            self._mesh_dimensions * (input_seq_length - 1)
            + property_dim
            + KinematicType.SIZE
        )
        norm_edge_dim = (
            self._mesh_dimensions + 1
        ) * 2  # [drs, |drs|, drs_ref, |drs_ref|]

        mo_edge_dim = (self._mesh_dimensions + 1) * 2
        om_edge_dim = (self._mesh_dimensions + 1) * 2
        # Initialize the gnn pipeline
        self._encode_process_decode = EncodeProcessDecode(
            mesh_n_dim_in=node_dim,
            mesh_n_dim_out=self._mesh_dimensions,
            obj_n_dim_in=node_dim,
            obj_n_dim_out=self._mesh_dimensions,
            mo_edge_dim=mo_edge_dim,
            om_edge_dim=om_edge_dim,
            latent_dim=latent_dim,
            message_passing_steps=nmessage_passing_steps,
            mlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self._node_dim = node_dim
        self._num_nodes = 0
        self._num_objs = 0
        self._index_offsets = {}

        # Setup normalizers
        self._node_normalizer = Normalizer(
            size=self._node_dim, name="node_normalizer", device=device
        )
        self._mo_edge_normalizer = Normalizer(
            size=norm_edge_dim, name="mo_edge_normalizer", device=device
        )
        self._om_edge_normalizer = Normalizer(
            size=norm_edge_dim, name="om_edge_normalizer", device=device
        )
        self._output_normalizer = Normalizer(
            size=self._mesh_dimensions, name="output_normalizer", device=device
        )
        self._edge_normalizer = {
            MOEdge: self._mo_edge_normalizer,
            OMEdge: self._om_edge_normalizer,
        }
        self._device = device

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
            input[k].x = self._node_normalizer(input[k].x)
        for k in input.edge_attr_dict.keys():
            if input[k].edge_index.shape[1] > 0:
                input[k].edge_attr = self._edge_normalizer[k](
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
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer = self._node_normalizer.get_variable()
        _mo_edge_normalizer = self._mo_edge_normalizer.get_variable()
        _om_edge_normalizer = self._om_edge_normalizer.get_variable()

        save_data = {
            "model": model,
            "_output_normalizer": _output_normalizer,
            "_node_normalizer": _node_normalizer,
            "_mo_edge_normalizer": _mo_edge_normalizer,
            "_om_edge_normalizer": _om_edge_normalizer,
        }

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
