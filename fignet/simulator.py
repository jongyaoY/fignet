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

from typing import Dict, Optional

import torch
import torch.nn as nn

from fignet.graph_networks import EncodeProcessDecode
from fignet.normalization import Normalizer
from fignet.utils import KinematicType, NodeType, to_tensor


class LearnedSimulator(nn.Module):
    def __init__(
        self,
        mesh_dimensions: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        property_dim: int = 5,
        noise_std: float = 1e-5,
        device="cpu",
    ):
        """Initializer

        Args:
            mesh_dimensions (int): Mesh position dimension, 3 for 3d meshes
            latent_dim (int): Latent feature dimension
            nmessage_passing_steps (int): Number of message passing steps
            nmlp_layers (int): Number of MLP layers mlp_hidden_dim (int):
            Hidden MLP dimension
            noise_std (float, optional): Velocity noise during training.
            Defaults to 1e-5. device (str, optional): Defaults to "cpu".
        """
        super(LearnedSimulator, self).__init__()

        self._mesh_dimensions = mesh_dimensions
        assert self._mesh_dimensions == 3

        # Initialize the EncodeProcessDecode
        self._num_node_types = len(NodeType)
        # !Not used
        self._node_type_embedding_size = 9
        self._noise_std = noise_std

        # node_dim =
        # self._mesh_dimensions + 2 + self._node_type_embedding_size
        # vel, kin, properties, node_type_embedding
        node_dim = self._mesh_dimensions + property_dim + KinematicType.SIZE
        norm_edge_dim = (
            self._mesh_dimensions + 1
        ) * 2  # [drs, |drs|, drs_ref, |drs_ref|]
        face_edge_dim = (
            7 * (self._mesh_dimensions + 1) + 2 * self._mesh_dimensions
        )  # [drs, |drs|, dsi, |dsi|, dri, |dri|, nr, ns]
        self._node_type_embedding = nn.Embedding(
            self._num_node_types, self._node_type_embedding_size, device=device
        )
        # Initialize the gnn pipeline
        self._encode_process_decode = EncodeProcessDecode(
            mesh_n_dim_in=node_dim,
            mesh_n_dim_out=self._mesh_dimensions,
            obj_n_dim_in=node_dim,
            obj_n_dim_out=self._mesh_dimensions,
            norm_edge_dim=norm_edge_dim,
            face_edge_dim=face_edge_dim,
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
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
        self._regular_edge_normalizer = Normalizer(
            size=norm_edge_dim, name="regular_edge_normalizer", device=device
        )
        self._face_edge_normalizer = Normalizer(
            size=face_edge_dim, name="face_edge_normalizer", device=device
        )
        self._output_normalizer = Normalizer(
            size=self._mesh_dimensions, name="output_normalizer", device=device
        )

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
        stats: Optional[dict] = None,
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
        if stats is None:
            return self._output_normalizer(accelerations, self.training)
        else:
            return (
                accelerations - to_tensor(stats["mean"], self._device)
            ) / to_tensor(stats["std"], self._device)

    def _encoder_preprocessor(
        self,
        input: Dict[str, torch.Tensor],
    ):
        """Preprocess input including normalization and adding noise

        Args:
            input (dict): Input graph

        Returns:
            dict: Preprocessed graph
        """

        input["node_features"]["mesh"] = input["node_features"][
            "mesh"
        ].squeeze()
        input["node_features"]["object"] = input["node_features"][
            "object"
        ].squeeze()

        input["index"]["mm"] = input["index"]["mm"].squeeze()
        input["index"]["mo"] = input["index"]["mo"].squeeze()
        input["index"]["om"] = input["index"]["om"].squeeze()
        input["index"]["ff"] = input["index"]["ff"].squeeze()
        m_node_types = torch.nn.functional.one_hot(
            input["kinematic"]["mesh"].squeeze(), KinematicType.SIZE
        )
        o_node_types = torch.nn.functional.one_hot(
            input["kinematic"]["object"].squeeze(), KinematicType.SIZE
        )
        input["node_features"]["mesh"] = torch.cat(
            [input["node_features"]["mesh"], m_node_types], dim=-1
        )
        input["node_features"]["object"] = torch.cat(
            [input["node_features"]["object"], o_node_types], dim=-1
        )
        # Normalize node features
        input["node_features"]["mesh"] = self._node_normalizer(
            input["node_features"]["mesh"]
        )
        input["node_features"]["object"] = self._node_normalizer(
            input["node_features"]["object"]
        )
        # Add noise to velocities
        m_x_noise = torch.normal(
            std=self._noise_std,
            mean=0.0,
            size=input["node_features"]["mesh"][:, :3].shape,
        ).to(self._device)
        o_x_noise = torch.normal(
            std=self._noise_std,
            mean=0.0,
            size=input["node_features"]["object"][:, :3].shape,
        ).to(self._device)
        input["node_features"]["mesh"][:, :3] + m_x_noise
        input["node_features"]["object"][:, :3] + o_x_noise

        input["edge_features"]["mm"] = self._regular_edge_normalizer(
            input["edge_features"]["mm"].squeeze()
        )
        input["edge_features"]["mo"] = self._regular_edge_normalizer(
            input["edge_features"]["mo"].squeeze()
        )
        input["edge_features"]["om"] = self._regular_edge_normalizer(
            input["edge_features"]["om"].squeeze()
        )
        if input["index"]["ff"].shape[1] > 0:
            input["edge_features"]["ff"] = self._face_edge_normalizer(
                input["edge_features"]["ff"].squeeze()
            )

        return input

    def predict_accelerations(
        self,
        input: Dict[str, torch.Tensor],
    ):
        """Predict mesh and object node next accelerations (velocity change)

        Args:
            input (Dict[str, torch.Tensor]): Input graph

        Returns:
            torch.Tensor: Mesh node accelerations
            torch.Tensor: Object node accelerations
        """
        input = self._encoder_preprocessor(input)

        m_pred_acc, o_pred_acc = self._encode_process_decode(
            mesh_n=input["node_features"]["mesh"],
            obj_n=input["node_features"]["object"],
            mm_index=input["index"]["mm"],
            mo_index=input["index"]["mo"],
            om_index=input["index"]["om"],
            ff_index=input["index"]["ff"],
            e_mm=input["edge_features"]["mm"],
            e_mo=input["edge_features"]["mo"],
            e_om=input["edge_features"]["om"],
            e_ff=input["edge_features"]["ff"],
        )
        return m_pred_acc, o_pred_acc

    def save(self, path: str = "model.pt"):
        """Save model state

        Args:
            path: Model path
        """
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer = self._node_normalizer.get_variable()
        _regular_edge_normalizer = self._regular_edge_normalizer.get_variable()
        _face_edge_normalizer = self._face_edge_normalizer.get_variable()

        save_data = {
            "model": model,
            "_output_normalizer": _output_normalizer,
            "_node_normalizer": _node_normalizer,
            "_regular_edge_normalizer": _regular_edge_normalizer,
            "_face_edge_normalizer": _face_edge_normalizer,
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
