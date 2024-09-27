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


from collections import defaultdict
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType
from torch_scatter import scatter

from fignet.data import HeteroGraph
from fignet.graph_networks import Decoder, Encoder, build_mlp
from fignet.modules.hetero_conv import GenericHeteroConv
from fignet.types import FFEdge


class NodeConv(nn.Module):
    def __init__(
        self,
        node_in_dim,
        node_out_dim,
        edge_aggr_dim,
        mlp_layers,
        mlp_hidden_dim,
    ):
        super(NodeConv, self).__init__()

        # Node MLP
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    node_in_dim + edge_aggr_dim,
                    [mlp_hidden_dim for _ in range(mlp_layers)],
                    node_out_dim,
                ),
                nn.LayerNorm(node_out_dim),
            ]
        )

    def forward(self, x, aggr):
        return x + self.node_fn(torch.cat((x, aggr), dim=-1))


class EdgeConv(MessagePassing):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        edge_out_dim,
        mlp_layers,
        mlp_hidden_dim,
    ):
        super(EdgeConv, self).__init__(aggr="add")

        # Edge MLP
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    node_in_dim + node_in_dim + edge_in_dim,
                    [mlp_hidden_dim for _ in range(mlp_layers)],
                    edge_out_dim,
                ),
                nn.LayerNorm(edge_out_dim),
            ]
        )

    def forward(self, x, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """Returns e_updated and aggregation"""
        hyper_edge = False
        if edge_index.ndim == 3:
            edge_index = edge_index.view(
                2, edge_index.shape[1] * edge_index.shape[2]
            )
            hyper_edge = True
        if edge_index.shape[1] == 0:
            if isinstance(x, torch.Tensor):
                aggr = torch.zeros_like(x)
            elif isinstance(x, tuple):
                aggr = torch.zeros_like(x[1])
            else:
                raise NotImplementedError
            return edge_attr, aggr
        else:
            edge_attr_updated, aggr = self.propagate(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                hyper_edge=hyper_edge,
            )
            return (
                edge_attr + edge_attr_updated.view(edge_attr.shape[0], -1),
                aggr,
            )

    def message(
        self, x_i, x_j, edge_attr, hyper_edge: bool
    ):  # receiver  # sender
        if hyper_edge:
            num_edge = edge_attr.shape[0]
            assert num_edge != 0
            e_latent = torch.hstack(
                [x_i, x_j, edge_attr.view(x_i.shape[0], -1)]
            ).view(
                num_edge, -1
            )  # (num_edge, 3 * n_feature)
            return self.edge_fn(e_latent).view(
                x_i.shape[0], -1
            )  # (num_edge * 3, n_feature)
        else:
            e_latent = torch.cat((x_i, x_j, edge_attr), dim=-1)
            return self.edge_fn(e_latent)

    def aggregate(
        self, inputs: torch.Tensor, index: torch.Tensor, dim_size=None
    ):
        out = scatter(
            inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum"
        )
        return inputs, out


class Processor(nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        latent_dim,
        mlp_layers,
        mlp_hidden_dim,
        message_passing_steps,
    ):
        super(Processor, self).__init__()
        self.gnn_stacks = nn.ModuleList()
        params = {
            "node_in_dim": latent_dim,
            "node_out_dim": latent_dim,
            "edge_in_dim": latent_dim,
            "edge_out_dim": latent_dim,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
        }
        for _ in range(message_passing_steps):
            interaction_dict = {}
            edge_aggr_dim = defaultdict(list)
            for edge_type in edge_types:
                param_local = params.copy()
                param_local.pop("node_out_dim")
                if edge_type == FFEdge:
                    param_local["edge_out_dim"] = (
                        3 * param_local["edge_out_dim"]
                    )  # TODO: hardcoded 3
                    param_local["node_in_dim"] = (
                        3 * param_local["node_in_dim"]
                    )  # TODO: hardcoded 3
                    param_local["edge_in_dim"] = (
                        3 * param_local["edge_in_dim"]
                    )  # TODO: hardcoded 3
                dst_node_type = edge_type[2]
                edge_aggr_dim[dst_node_type].append(params["edge_out_dim"])
                interaction_dict.update({edge_type: EdgeConv(**param_local)})
            for node_type in node_types:
                param_local = params.copy()
                param_local.pop("edge_in_dim")
                param_local.pop("edge_out_dim")
                param_local["edge_aggr_dim"] = sum(edge_aggr_dim[node_type])
                interaction_dict.update({node_type: NodeConv(**param_local)})
            layer = GenericHeteroConv(
                convs=interaction_dict,
                aggr=None,
            )
            self.gnn_stacks.append(layer)
        self.node_types = node_types
        self.edge_types = edge_types

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for gnn in self.gnn_stacks:
            out = gnn(x_dict, edge_index_dict, edge_attr_dict)
            for key in out.keys():
                if isinstance(key, str):
                    x_dict[key] = out[key]
                elif isinstance(key, tuple):
                    edge_attr_dict[key] = out[key]

        return x_dict


class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        input_dim_dict: Dict[Union[NodeType, EdgeType], int],
        output_dim_dict: Dict[Union[NodeType, EdgeType], int],
        latent_dim: int,
        message_passing_steps: int,
        mlp_layers: int,
        mlp_hidden_dim: int,
    ):
        super(EncodeProcessDecode, self).__init__()
        self.params = {
            "latent_dim": latent_dim,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            "message_passing_steps": message_passing_steps,
        }
        self.encoders: Dict[str, Encoder] = ModuleDict()
        self.decoders: Dict[str, Decoder] = ModuleDict()
        edge_types = []
        node_types = []
        for input_type, input_dim in input_dim_dict.items():
            out_dim = latent_dim
            if isinstance(
                input_type, tuple
            ):  # TODO: check typing with EdgeType
                edge_types.append(input_type)
                if input_type == FFEdge:  # TODO: get rid of explicit edge type
                    out_dim = 3 * latent_dim  # TODO: hardcoded 3
                # input_type = input_type[1] # TODO: nn.ModuleDict takes only str keys
            else:
                node_types.append(input_type)

            self.encoders.update(
                {
                    input_type: Encoder(
                        input_dim, out_dim, mlp_layers, mlp_hidden_dim
                    )
                }
            )

        for output_type, output_dim in output_dim_dict.items():
            self.decoders.update(
                {
                    output_type: Decoder(
                        nnode_in=latent_dim,
                        nnode_out=output_dim,
                        nmlp_layers=mlp_layers,
                        mlp_hidden_dim=mlp_hidden_dim,
                    )
                }
            )
        self.processor = Processor(
            node_types=node_types, edge_types=edge_types, **self.params
        )

    def forward(self, data: HeteroGraph):
        for node_type in data.x_dict.keys():
            data[node_type].x = self.encoders[node_type](data[node_type].x)
        for edge_type in data.edge_index_dict.keys():
            data[edge_type].edge_attr = self.encoders[edge_type](
                data[edge_type].edge_attr
            )

        latent_out = self.processor(
            data.x_dict, data.edge_index_dict, data.edge_attr_dict
        )
        # TODO
        out = {}
        for node_type in latent_out.keys():
            out[node_type] = self.decoders[node_type](latent_out[node_type])
        return out["mesh"], out["object"]  # TODO
