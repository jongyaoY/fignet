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


import numpy as np

from fignet.data import HeteroGraph
from fignet.graph_builders import GraphBuildType
from fignet.scene import SceneInfoDict, SceneInfoKey
from fignet.utils.conversion import to_tensor


def noised_position_sequence(
    position_sequence: np.ndarray, noise_std, mask=None
):
    if mask is None:
        return position_sequence + np.random.normal(
            0, noise_std, position_sequence.shape
        )
    else:
        return position_sequence[:, mask, ...] + np.random.normal(
            0, noise_std, position_sequence[:, mask, ...].shape
        )


def build_graph(scn_info: SceneInfoDict, config: dict = None):
    if config is not None:
        if "noise_std" in config:
            noise_std = config.get("noise_std")
            scn_info[SceneInfoKey.VERT_SEQ] = noised_position_sequence(
                scn_info[SceneInfoKey.VERT_SEQ], noise_std=noise_std
            )
            scn_info[SceneInfoKey.COM_SEQ] = noised_position_sequence(
                scn_info[SceneInfoKey.COM_SEQ], noise_std=noise_std
            )
    graph_struct_type = GraphBuildType.FIGNET  # TODO

    # TODO: create graph builder class
    if graph_struct_type == GraphBuildType.FIGNET_ACT:
        from fignet.graph_builders.fignet_act import (
            cal_connectivity,
            cal_node_attr,
        )
    elif graph_struct_type == GraphBuildType.FIGNET:
        from fignet.graph_builders.fignet import (
            cal_connectivity,
            cal_node_attr,
        )
    else:
        raise NotImplementedError

    edge_index_dict, edge_attr_dict = cal_connectivity(scn_info)
    node_attr_dict = cal_node_attr(scn_info)

    data = HeteroGraph()
    for node_type, node_attr in node_attr_dict.items():
        for k, v in node_attr.items():
            if v is not None:
                if isinstance(v, np.ndarray):
                    setattr(data[node_type], k, to_tensor(v))
                else:
                    raise NotImplementedError
    for edge_type, edge_attr in edge_attr_dict.items():
        data[edge_type].edge_index = to_tensor(edge_index_dict[edge_type])
        data[edge_type].edge_attr = to_tensor(edge_attr)

    return data
