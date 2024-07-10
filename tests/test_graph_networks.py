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

import pytest
import torch

from fignet.graph_networks import EncodeProcessDecode

n_mnode = 20
n_onode = 5
mesh_dim = 3


@pytest.fixture
def model_data():
    node_feature_dim = 10
    edge_feature_dim = 12
    latent_dim = 128
    nmessage_passing_steps = 2
    nmlp_layers = 2
    mlp_hidden_dim = 128
    model = EncodeProcessDecode(
        mesh_n_dim_in=node_feature_dim,
        mesh_n_dim_out=mesh_dim,
        obj_n_dim_in=node_feature_dim,
        obj_n_dim_out=mesh_dim,
        norm_edge_dim=edge_feature_dim,
        face_edge_dim=edge_feature_dim,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )

    n_mmedge = 10
    n_moedge = 12
    n_ffedge = 5
    mesh_x = torch.rand(n_mnode, node_feature_dim)
    obj_x = torch.rand(n_onode, node_feature_dim)
    mm_index = torch.randint(0, n_mnode, (2, n_mmedge))
    mo_index = torch.vstack(
        [
            torch.randint(0, n_mnode, (1, n_moedge)),
            torch.randint(0, n_onode, (1, n_moedge)),
        ]
    )
    om_index = mo_index[[1, 0]]
    ff_index = torch.randint(0, n_mnode, (2, n_ffedge, 3))

    e_mm = torch.rand(n_mmedge, edge_feature_dim)
    e_om = torch.rand(n_moedge, edge_feature_dim)
    e_mo = torch.rand(n_moedge, edge_feature_dim)
    e_ff = torch.rand(n_ffedge, edge_feature_dim)

    data = {
        "mesh_n": mesh_x,
        "obj_n": obj_x,
        "mm_index": mm_index,
        "mo_index": mo_index,
        "om_index": om_index,
        "ff_index": ff_index,
        "e_mm": e_mm,
        "e_om": e_om,
        "e_mo": e_mo,
        "e_ff": e_ff,
    }
    return model, data


def test_encode_process_decode(model_data):
    model, data = model_data
    output = model(**data)
    assert output[0].shape == (n_mnode, mesh_dim)
    assert output[1].shape == (n_onode, mesh_dim)
