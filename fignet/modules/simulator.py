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


from dataclasses import asdict, dataclass
from typing import Dict, Union

import torch
import torch.nn as nn
from dacite import from_dict
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import NodeOrEdgeType

from fignet.data.hetero_graph import HeteroGraph
from fignet.graph_builders import GraphBuildCfg
from fignet.modules.message_passing import EncodeProcessDecode
from fignet.modules.normalization import Normalizer


@dataclass
class SimCfg:
    input_sequence_length: int
    collision_radius: float
    build_cfg: GraphBuildCfg


class LearnedSimulator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        message_passing_steps: int,
        mlp_layers: int,
        mlp_hidden_dim: int,
        device="cpu",
    ):
        """Initializer

        Args:
            mesh_dimensions (int): Mesh position dimension, 3 for 3d meshes
            latent_dim (int): Latent feature dimension
            message_passing_steps (int): Number of message passing steps
            mlp_layers (int): Number of MLP layers mlp_hidden_dim (int):
            Hidden MLP dimension
            device (str, optional): Defaults to "cpu".
        """
        super(LearnedSimulator, self).__init__()

        self._mesh_dimensions = 3

        self._is_initialized = False
        self._encode_process_decode: EncodeProcessDecode = None
        self._node_normalizers: Dict[str, Normalizer] = ModuleDict()
        self._edge_normalizers: Dict[tuple, Normalizer] = ModuleDict()
        self._output_normalizer = None
        self._device = device
        self._gnn_params = {
            "latent_dim": latent_dim,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            "message_passing_steps": message_passing_steps,
        }
        self._init_info: Dict[str, Dict[NodeOrEdgeType, int]] = None
        self._cfg: SimCfg = None

    @property
    def initialized(self) -> bool:
        return self._is_initialized

    @property
    def cfg(self) -> SimCfg:
        return self._cfg

    @property
    def device(self):
        return self._device

    def init(
        self,
        init_info: Union[HeteroGraph, Dict[str, Dict[NodeOrEdgeType, int]]],
        cfg: SimCfg,
    ):
        """Lazy initializer"""
        if isinstance(init_info, HeteroGraph):
            node_dim_dict = init_info.num_node_features
            edge_dim_dict = init_info.num_edge_features
            self._init_info = {
                "node_dim_dict": node_dim_dict,
                "edge_dim_dict": edge_dim_dict,
            }
        elif isinstance(init_info, dict):
            node_dim_dict = init_info["node_dim_dict"]
            edge_dim_dict = init_info["edge_dim_dict"]
            self._init_info = init_info
        # Initialize GNN
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
        # Make sure normalizers and GNN params are on the same device
        device = next(self.parameters()).device
        # Initialize normalizers
        for node_type, node_dim in node_dim_dict.items():
            self._node_normalizers[node_type] = Normalizer(
                size=node_dim,
                name="_".join([node_type, "node_normalizer"]),
                device=device,
            )
        for edge_type, edge_dim in edge_dim_dict.items():
            edge_type_name = edge_type[1]
            self._edge_normalizers[edge_type] = Normalizer(
                size=edge_dim,
                name="_".join([edge_type_name, "edge_normalizer"]),
                device=device,
            )
        self._output_normalizer = Normalizer(
            size=self._mesh_dimensions,
            name="output_normalizer",
            device=device,
        )

        self._cfg = cfg
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

    def forward(self, input: HeteroGraph):
        out, _ = self.predict_accelerations(input)
        if not self.training:
            out = self.denormalize_accelerations(out)
        return out

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

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["cfg"] = asdict(self.cfg)
        state["init_info"] = self._init_info
        state["gnn_params"] = self._gnn_params
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        cfg = state_dict.pop("cfg")
        cfg = from_dict(SimCfg, cfg)
        init_info = state_dict.pop("init_info")
        if "gnn_params" in state_dict:
            state_dict.pop("gnn_params")
        if not self.initialized:
            self.init(init_info, cfg)
        else:
            if cfg != self._cfg or init_info != self._init_info:
                raise ValueError("Loaded states not match")
        self._is_initialized = True

        return super().load_state_dict(state_dict, *args, **kwargs)

    def save(self, path: str = "model.pt"):
        """Save model state

        Args:
            path: Model path
        """

        torch.save(self.state_dict(), path)

    @classmethod
    def load(self, path: str):
        """Load model state from file

        Args:
            path: Model path
        """
        dicts = torch.load(path)

        if "state_dict" in dicts:
            state_dict = dicts["state_dict"]
        else:
            state_dict = dicts
        model = LearnedSimulator(**state_dict["gnn_params"])
        model.load_state_dict(state_dict)

        print("Simulator model loaded checkpoint %s" % path)
        return model
