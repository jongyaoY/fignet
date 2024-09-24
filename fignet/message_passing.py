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
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_scatter import scatter

from fignet.data import HeteroGraph
from fignet.graph_networks import Decoder, Encoder, build_mlp

MOEdge = ("mesh", "m-o", "object")
OMEdge = ("object", "o-m", "mesh")


class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        node_in_dim,
        node_out_dim,
        edge_in_dim,
        edge_out_dim,
        mlp_layers,
        mlp_hidden_dim,
    ):
        super(InteractionNetwork, self).__init__(aggr="add")
        # Node MLP
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    node_in_dim + edge_out_dim,
                    [mlp_hidden_dim for _ in range(mlp_layers)],
                    node_out_dim,
                ),
                nn.LayerNorm(node_out_dim),
            ]
        )
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

    def forward(self, x, edge_index, edge_attr):
        r_dim = 1
        if isinstance(edge_attr, tuple):
            edge_attr = edge_attr[r_dim]
        edge_attr_updated, aggr = self.propagate(
            x=x, edge_index=edge_index, edge_attr=edge_attr
        )
        x_updated = self.node_fn(torch.cat((x[r_dim], aggr), dim=-1))
        return x[r_dim] + x_updated, edge_attr + edge_attr_updated

    def message(self, x_i, x_j, edge_attr):  # receiver  # sender
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
        self, latent_dim, mlp_layers, mlp_hidden_dim, message_passing_steps
    ):
        super(Processor, self).__init__()
        self.gnn_stacks = torch.nn.ModuleList()
        params = {
            "node_in_dim": latent_dim,
            "node_out_dim": latent_dim,
            "edge_in_dim": latent_dim,
            "edge_out_dim": latent_dim,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
        }
        for _ in range(message_passing_steps):
            layer = HeteroConv(
                {
                    MOEdge: InteractionNetwork(**params),
                    OMEdge: InteractionNetwork(**params),
                },
                aggr="cat",
            )
            self.gnn_stacks.append(layer)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for gnn in self.gnn_stacks:
            out = gnn(x_dict, edge_index_dict, edge_attr_dict)
            x_dict["mesh"] = out["mesh"][0]
            edge_attr_dict[OMEdge] = out["mesh"][1]
            x_dict["object"] = out["object"][0]
            edge_attr_dict[MOEdge] = out["object"][1]

        return x_dict


class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        mesh_n_dim_in: int,
        mesh_n_dim_out: int,
        obj_n_dim_in: int,
        obj_n_dim_out: int,
        mo_edge_dim: int,
        om_edge_dim: int,
        latent_dim: int,
        message_passing_steps: int,
        mlp_layers: int,
        mlp_hidden_dim: int,
    ):
        super(EncodeProcessDecode, self).__init__()
        self.params = {
            # "node_dim_in": mesh_n_dim_in,
            # "node_dim_out": mesh_n_dim_out,
            "latent_dim": latent_dim,
            "mlp_layers": mlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim,
            "message_passing_steps": message_passing_steps,
        }

        self.m_encoder = Encoder(
            mesh_n_dim_in, latent_dim, mlp_layers, mlp_hidden_dim
        )
        self.o_encoder = Encoder(
            obj_n_dim_in, latent_dim, mlp_layers, mlp_hidden_dim
        )
        self.emo_encoder = Encoder(
            mo_edge_dim, latent_dim, mlp_layers, mlp_hidden_dim
        )
        self.eom_encoder = Encoder(
            om_edge_dim, latent_dim, mlp_layers, mlp_hidden_dim
        )
        self.m_decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=mesh_n_dim_out,
            nmlp_layers=mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.m_decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=obj_n_dim_out,
            nmlp_layers=mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.o_decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=obj_n_dim_out,
            nmlp_layers=mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.processor = Processor(**self.params)

    def forward(self, data: HeteroGraph):
        data["mesh"].x = self.m_encoder(data["mesh"].x)
        data["object"].x = self.o_encoder(data["object"].x)
        data[MOEdge].edge_attr = self.emo_encoder(data[MOEdge].edge_attr)
        data[OMEdge].edge_attr = self.eom_encoder(data[OMEdge].edge_attr)

        latent_out = self.processor(
            data.x_dict, data.edge_index_dict, data.edge_attr_dict
        )

        m_out = self.m_decoder(latent_out["mesh"])
        o_out = self.o_decoder(latent_out["object"])
        return m_out, o_out
