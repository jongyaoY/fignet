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

from typing import List, Optional

import torch
import torch.nn as nn
from torch_scatter import scatter

from fignet.utils import to_tensor


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


class InteractionNetwork(nn.Module):
    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        fedge_in: int,
        fedge_out: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        leave_out_mm: bool,
    ):
        """Single layer of the face interaction network

        Args:
            nnode_in (int): Input latent node feature dimension
            nnode_out (int): Output latent node feature dimension
            nedge_in (int): Input latent regular edge feature dimension
            nedge_out (int): Output latent regular edge feature dimension
            fedge_in (int): Input latent face-face edge feature dimension
            fedge_out (int): Output latent face-face edge feature dimension
            nmlp_layers (int):  Number of hidden layers in the MLP
            mlp_hidden_dim (int): Dimension of the MLP hidden layer
        """
        # Aggregate features from neighbors
        super(InteractionNetwork, self).__init__()
        # Node MLP
        self.mesh_node_fn = nn.Sequential(
            *[
                build_mlp(
                    (
                        nnode_in + 2 * nedge_in + fedge_in
                        if not leave_out_mm
                        else nnode_in + nedge_in + fedge_in
                    ),
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nnode_out,
                ),
                nn.LayerNorm(nnode_out),
            ]
        )
        self.obj_node_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in + nedge_in,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nnode_out,
                ),
                nn.LayerNorm(nnode_out),
            ]
        )
        # Edge MLP
        if not leave_out_mm:
            self.mm_edge_fn = nn.Sequential(
                *[
                    build_mlp(
                        nnode_in + nnode_in + nedge_in,
                        [mlp_hidden_dim for _ in range(nmlp_layers)],
                        nedge_out,
                    ),
                    nn.LayerNorm(nedge_out),
                ]
            )
        else:
            self.mm_edge_fn = None

        self.mo_edge_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in + nnode_in + nedge_in,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nedge_out,
                ),
                nn.LayerNorm(nedge_out),
            ]
        )
        self.om_edge_fn = nn.Sequential(
            *[
                build_mlp(
                    nnode_in + nnode_in + nedge_in,
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    nedge_out,
                ),
                nn.LayerNorm(nedge_out),
            ]
        )

        self.ff_edge_fn = nn.Sequential(
            *[
                build_mlp(
                    3 * (nnode_in + nnode_in + fedge_in),
                    [mlp_hidden_dim for _ in range(nmlp_layers)],
                    3 * fedge_out,
                ),
                nn.LayerNorm(3 * fedge_out),
            ]
        )

    def forward(
        self,
        mesh_n: torch.tensor,
        obj_n: torch.tensor,
        mm_index: torch.tensor,
        mo_index: torch.tensor,
        om_index: torch.tensor,
        ff_index: torch.tensor,
        e_mm: torch.tensor,
        e_mo: torch.tensor,
        e_om: torch.tensor,
        e_ff: Optional[torch.tensor] = None,
    ):
        """Runs the interaction network: calculate and aggregate messages,
        update node and edge features. Input and output are summed

        Args:
            mesh_n (torch.tensor): Mesh node latent features (num_mnode,
            mnode_dim)

            obj_n (torch.tensor): Obect node latent features (num_onode,
            onode_dim)


            mm_index (torch.tensor): Regular edge index between mesh nodes
            (num_mmedge, mmedge_dim)

            mo_index (torch.tensor): Regular edge index from mesh to object
            nodes (num_moedge, moedge_dim)

            om_index (torch.tensor): Regular edge index from object to mesh
            nodes (num_omedge, omedge_dim)

            ff_index (torch.tensor): Face-face edge index of shape
            (num_fedge, 3, fedge_dim)
            e_mm (torch.tensor): Mesh-mesh edge latent features
            e_mo (torch.tensor): Mesh-object edge latent features
            e_om (torch.tensor): Object-mesh edge latent features
            e_ff (Optional[torch.tensor], optional): Face-face edge latent
            features. None if there are no face-face edges

        Returns:
            torch.tensor: Updated mesh node latent
            torch.tensor: Updated object node latent
            torch.tensor: Updated mesh-mesh edge latent
            torch.tensor: Updated mesh-object edge latent
            torch.tensor: Updated object-mesh edge latent
            torch.tensor: Updated face-face edge latent
        """
        # Calculate messages
        s_dim = 0
        r_dim = 1
        if e_mm is not None:
            e_mm_updated = self.message(
                torch.index_select(mesh_n, 0, mm_index[s_dim]),
                torch.index_select(mesh_n, 0, mm_index[r_dim]),
                "mm",
                e_mm,
            )
        e_mo_updated = self.message(
            torch.index_select(mesh_n, 0, mo_index[s_dim]),
            torch.index_select(obj_n, 0, mo_index[r_dim]),
            "mo",
            e_mo,
        )

        e_om_updated = self.message(
            torch.index_select(obj_n, 0, om_index[s_dim]),
            torch.index_select(mesh_n, 0, om_index[r_dim]),
            "om",
            e_om,
        )
        if e_ff is not None:
            e_ff_updated = self.message(
                torch.index_select(mesh_n, 0, ff_index[s_dim].flatten()).view(
                    e_ff.shape[0], e_ff.shape[1], -1
                ),
                torch.index_select(mesh_n, 0, ff_index[r_dim].flatten()).view(
                    e_ff.shape[0], e_ff.shape[1], -1
                ),
                "ff",
                e_ff,
            )
        # Aggregate
        if e_mm is not None:
            aggr_mm = scatter(
                e_mm_updated,
                mm_index[r_dim],
                dim=0,
                dim_size=mesh_n.shape[0],
                reduce="sum",
            )
        aggr_mo = scatter(
            e_mo_updated,
            mo_index[r_dim],
            dim=0,
            dim_size=obj_n.shape[0],
            reduce="sum",
        )
        aggr_om = scatter(
            e_om_updated,
            om_index[r_dim],
            dim=0,
            dim_size=mesh_n.shape[0],
            reduce="sum",
        )

        if e_ff is not None:
            ff_r_index = ff_index[r_dim].view(
                ff_index.shape[1] * ff_index.shape[2]
            )  # receiver indices
            aggr_ff = scatter(
                e_ff_updated.view(
                    e_ff_updated.shape[0] * e_ff_updated.shape[1], -1
                ),
                ff_r_index,
                dim=0,
                dim_size=mesh_n.shape[0],
                reduce="sum",
            )
        else:
            aggr_ff = to_tensor(torch.zeros_like(mesh_n))
        # Update nodes
        obj_n_updated = self.obj_node_fn(torch.cat([obj_n, aggr_mo], dim=-1))
        if e_mm is not None:
            mesh_n_updated = self.mesh_node_fn(
                torch.cat([mesh_n, aggr_om, aggr_mm, aggr_ff], dim=-1)
            )
        else:
            mesh_n_updated = self.mesh_node_fn(
                torch.cat([mesh_n, aggr_om, aggr_ff], dim=-1)
            )

        return (
            mesh_n + mesh_n_updated,
            obj_n + obj_n_updated,
            e_mm + e_mm_updated if e_mm is not None else None,
            e_mo + e_mo_updated,
            e_om + e_om_updated,
            e_ff + e_ff_updated if e_ff is not None else None,
        )

    def message(
        self,
        x_j: torch.tensor,
        x_i: torch.tensor,
        edge_type: str,
        edge_features: torch.tensor,
    ):
        """Generalized message function that can also handle face-face message
        passing

        Args:
            x_j (torch.tensor): Sender node features
            x_i (torch.tensor): Receiver node features
            edge_type (str): Type of edges
            edge_features (torch.tensor): Edge features

        Raises:
            TypeError: If edge_type not in ['mm', 'om', 'mo']

        Returns:
            torch.tensor: Messages from senders to receivers
        """
        if x_i.ndim == 2:
            latent_n = torch.cat([edge_features, x_j, x_i], dim=-1)
            if edge_type not in ["mm", "om", "mo"]:
                raise TypeError(f"invalid edge type: {edge_type}")
            edge_function = getattr(self, edge_type + "_edge_fn")
            return edge_function(latent_n)
            # if edge_type == "mm":
            #     return self.mm_edge_fn(latent_n)
            # elif edge_type == "om":
            #     return self.om_edge_fn(latent_n)
            # elif edge_type == "mo":
            #     return self.mo_edge_fn(latent_n)
            # else:
            #     raise TypeError(f"invalid edge type: {edge_type}")

        elif x_i.ndim == 3:
            v_r = x_i.view(x_i.shape[0] * x_i.shape[1], -1)
            v_s = x_j.view(
                x_j.shape[0] * x_j.shape[1], -1
            )  # (n_edge * 3, n_feature)
            e_sr = edge_features.view(
                edge_features.shape[0] * edge_features.shape[1], -1
            )  # (n_edge * 3, n_feature)
            latent_face = torch.hstack([e_sr, v_s, v_r]).view(
                edge_features.shape[0], -1
            )  # (n_edge, n_feature*3)
            return self.ff_edge_fn(latent_face).view(
                edge_features.shape[0], 3, -1
            )


class Processor(nn.Module):
    """Contains the GNN stack"""

    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        fedge_in: int,
        fedge_out: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        leave_out_mm: bool,
    ):
        """Initializer

        Args:
            nnode_in (int): Input latent node feature dimension
            nnode_out (int): Output latent node feature dimension
            nedge_in (int): Input latent regular edge feature dimension
            nedge_out (int): Output latent regular edge feature dimension
            fedge_in (int): Input latent face-face edge feature dimension
            fedge_out (int): Output latent face-face edge feature dimension
            nmessage_passing_steps (int): Number of interaction networks in the
            stack
            nmlp_layers (int):  Number of hidden layers in the MLP
            mlp_hidden_dim (int): Dimension of the MLP hidden layer
        """
        super(Processor, self).__init__()
        # Create a stack of M Graph Networks GNs.
        self.gnn_stacks = nn.ModuleList(
            [
                InteractionNetwork(
                    nnode_in=nnode_in,
                    nnode_out=nnode_out,
                    nedge_in=nedge_in,
                    nedge_out=nedge_out,
                    fedge_in=fedge_in,
                    fedge_out=fedge_out,
                    nmlp_layers=nmlp_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                    leave_out_mm=leave_out_mm,
                )
                for _ in range(nmessage_passing_steps)
            ]
        )

    def forward(
        self,
        mesh_n: torch.tensor,
        obj_n: torch.tensor,
        mm_index: torch.tensor,
        mo_index: torch.tensor,
        om_index: torch.tensor,
        ff_index: torch.tensor,
        e_mm: torch.tensor,
        e_mo: torch.tensor,
        e_om: torch.tensor,
        e_ff: torch.tensor,
    ):
        """Run through the interaction network stack

        Args:
            mesh_n (torch.tensor): Mesh node latent features (num_mnode,
            mnode_dim)

            obj_n (torch.tensor): Obect node latent features (num_onode,
            onode_dim)


            mm_index (torch.tensor): Regular edge index between mesh nodes
            (num_mmedge, mmedge_dim)

            mo_index (torch.tensor): Regular edge index from mesh to object
            nodes (num_moedge, moedge_dim)

            om_index (torch.tensor): Regular edge index from object to mesh
            nodes (num_omedge, omedge_dim)

            ff_index (torch.tensor): Face-face edge index of shape
            (num_fedge, 3, fedge_dim)
            e_mm (torch.tensor): Mesh-mesh edge latent features
            e_mo (torch.tensor): Mesh-object edge latent features
            e_om (torch.tensor): Object-mesh edge latent features
            e_ff (Optional[torch.tensor], optional): Face-face edge latent
            features. None if there are no face-face edges

        Returns:
            torch.tensor: Mesh node latent features
            torch.tensor: Object node latent features
        """
        mesh_n_in = mesh_n
        obj_n_in = obj_n
        e_mm_in = e_mm
        e_mo_in = e_mo
        e_om_in = e_om
        e_ff_in = e_ff
        for gnn in self.gnn_stacks:
            (mesh_n_out, obj_n_out, e_mm_out, e_mo_out, e_om_out, e_ff_out) = (
                gnn(
                    mesh_n=mesh_n_in,
                    obj_n=obj_n_in,
                    mm_index=mm_index,
                    mo_index=mo_index,
                    om_index=om_index,
                    ff_index=ff_index,
                    e_mm=e_mm_in,
                    e_mo=e_mo_in,
                    e_om=e_om_in,
                    e_ff=e_ff_in,
                )
            )
            mesh_n_in = mesh_n_out
            obj_n_in = obj_n_out
            e_mm_in = e_mm_out
            e_mo_in = e_mo_out
            e_om_in = e_om_out
            e_ff_in = e_ff_out
        return mesh_n_out, obj_n_out


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


class EncodeProcessDecode(nn.Module):
    """Wraps encoder, processor and decoder in one module"""

    def __init__(
        self,
        mesh_n_dim_in: int,
        mesh_n_dim_out: int,
        obj_n_dim_in: int,
        obj_n_dim_out: int,
        norm_edge_dim: int,
        face_edge_dim: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        leave_out_mm: bool = False,
    ):
        """Initializer

        Args:
            mesh_n_dim_in (int): Input mesh node feature dimension
            mesh_n_dim_out (int): Output mesh node feature dimension
            obj_n_dim_in (int): Input object node feature dimension
            obj_n_dim_out (int): Output object node feature dimension
            norm_edge_dim (int): Regular edge feature dimension
            face_edge_dim (int): Face-face edge feature dimension
            latent_dim (int): Latent dimension of features in graph
            nmessage_passing_steps (int): Number of message passing steps
            nmlp_layers (int): Number of MLP hidden layers
            mlp_hidden_dim (int): MLP hidden dimension
            leave_out_mm (bool): leave out mesh node edges
        """
        super(EncodeProcessDecode, self).__init__()
        self._leave_out_mm = leave_out_mm
        self._m_encoder = Encoder(
            mesh_n_dim_in, latent_dim, nmlp_layers, mlp_hidden_dim
        )
        self._o_encoder = Encoder(
            obj_n_dim_in, latent_dim, nmlp_layers, mlp_hidden_dim
        )
        self._emo_encoder = Encoder(
            norm_edge_dim, latent_dim, nmlp_layers, mlp_hidden_dim
        )
        self._eom_encoder = Encoder(
            norm_edge_dim, latent_dim, nmlp_layers, mlp_hidden_dim
        )
        if not leave_out_mm:
            self._emm_encoder = Encoder(
                norm_edge_dim, latent_dim, nmlp_layers, mlp_hidden_dim
            )
        self._eff_encoder = Encoder(
            face_edge_dim, latent_dim * 3, nmlp_layers, mlp_hidden_dim
        )
        self._processor = Processor(
            nnode_in=latent_dim,
            nnode_out=latent_dim,
            nedge_in=latent_dim,
            nedge_out=latent_dim,
            fedge_in=latent_dim,
            fedge_out=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            leave_out_mm=leave_out_mm,
        )
        self._m_decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=mesh_n_dim_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self._o_decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=obj_n_dim_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(
        self,
        mesh_n: torch.Tensor,
        obj_n: torch.Tensor,
        om_index: torch.Tensor,
        mm_index: torch.Tensor,
        mo_index: torch.Tensor,
        ff_index: torch.Tensor,
        e_mm: torch.Tensor,
        e_mo: torch.Tensor,
        e_om: torch.Tensor,
        e_ff: torch.Tensor,
    ):
        """Forward hook

        Args:
            mesh_n (torch.Tensor): Input mesh node features (n_mnode, nfeature_dim)
            obj_n (torch.Tensor): Input object node features (n_onode, nfeature_dim)
            om_index (torch.Tensor): Object-mesh edge index (2, n_omedge)
            mm_index (torch.Tensor): Mesh-mesh edge index (2, n_mmedge)
            mo_index (torch.Tensor): Mesh-object edge index (2, n_moedge)
            ff_index (torch.Tensor): Face-face edge index (2, n_ffedge, 3)
            e_mm (torch.Tensor): Mesh-mesh edge features (n_mmedge, efeature_dim)
            e_mo (torch.Tensor): Mesh-object edge features (n_moedge, efeature_dim)
            e_om (torch.Tensor): Object-mesh edge features (n_omedge, efeature_dim)
            e_ff (torch.Tensor): Face-face edge features (n_fgedge, fefeature_dim)

        Returns:
            torch.Tensor: Decoded mesh node features
            torch.Tensor: Decoded object node features
        """

        mesh_n_latent = self._m_encoder(mesh_n)
        obj_n_latent = self._o_encoder(obj_n)
        if self._leave_out_mm:
            e_mm_latent = None
        else:
            e_mm_latent = self._emm_encoder(e_mm)
        e_mo_latent = self._emo_encoder(e_mo)
        e_om_latent = self._eom_encoder(e_om)
        if ff_index.shape[1] > 0:
            e_ff_latent = self._eff_encoder(e_ff).view(e_ff.shape[0], 3, -1)
        else:  # No face-face edges
            e_ff_latent = None

        mesh_n_latent_out, obj_n_latent_out = self._processor(
            mesh_n=mesh_n_latent,
            obj_n=obj_n_latent,
            mm_index=mm_index,
            mo_index=mo_index,
            om_index=om_index,
            ff_index=ff_index,
            e_mm=e_mm_latent,
            e_mo=e_mo_latent,
            e_om=e_om_latent,
            e_ff=e_ff_latent,
        )
        return self._m_decoder(mesh_n_latent_out), self._o_decoder(
            obj_n_latent_out
        )
