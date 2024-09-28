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

from dataclasses import dataclass
from typing import Dict

import numpy as np
from torch_geometric.typing import EdgeType, NodeType

from fignet.scene import SceneInfoKey
from fignet.types import MetaEnum


class FIGNodeType(MetaEnum):
    VERT = "vert"
    OBJECT = "object"


class FIGRelationType(MetaEnum):
    COLLIDE = "collide"
    CONNECT = "connect"


class FIGEdgeType(MetaEnum):
    VCollideO = (FIGNodeType.VERT, FIGRelationType.COLLIDE, FIGNodeType.OBJECT)
    OConnectV = (FIGNodeType.OBJECT, FIGRelationType.CONNECT, FIGNodeType.VERT)
    VConnectO = (FIGNodeType.VERT, FIGRelationType.CONNECT, FIGNodeType.OBJECT)
    VCollideV = (FIGNodeType.VERT, FIGRelationType.COLLIDE, FIGNodeType.VERT)


@dataclass
class GraphMetaData:
    node_dim: int
    edge_dim: int


@dataclass
class GraphBuildResult:
    node_attr_dict: Dict[NodeType, np.ndarray]
    edge_index_dict: Dict[EdgeType, np.ndarray]
    edge_attr_dict: Dict[EdgeType, np.ndarray]


def cal_inner_connectivity(scn_info, direction: str):
    edge_info = {}
    senders = []
    receivers = []
    for name, obj_id in scn_info[SceneInfoKey.OBJ_OFFSETS_DICT].items():
        v_offset = scn_info[SceneInfoKey.VERT_OFFSETS_DICT].get(name)
        o_offset = obj_id
        o_idx = np.repeat(0, scn_info[SceneInfoKey.NUM_VERTS_DICT].get(name))
        v_idx = np.arange(
            0,
            scn_info[SceneInfoKey.NUM_VERTS_DICT].get(name),
            dtype=np.int64,
        )
        o_idx = o_idx + o_offset
        v_idx = v_idx + v_offset

        if direction == "v-o":
            senders.append(v_idx)
            receivers.append(o_idx)
        elif direction == "o-v":
            senders.append(o_idx)
            receivers.append(v_idx)
        else:
            raise TypeError("Direction must be v-o or o-v")

    edge_info["verts_pos"] = scn_info[SceneInfoKey.VERT_SEQ][-1, ...]
    edge_info["com_pos"] = scn_info[SceneInfoKey.COM_SEQ][-1, ...]
    edge_info["verts_ref_pos"] = scn_info[SceneInfoKey.VERT_REF_POS]
    edge_info["com_ref_pos"] = scn_info[SceneInfoKey.COM_REF_POS]

    return senders, receivers, edge_info
