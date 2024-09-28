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

import enum
from dataclasses import dataclass
from typing import Dict

import numpy as np
from torch_geometric.typing import EdgeType, NodeType

from fignet.types import MetaEnum


class GraphBuildType(enum.Enum, metaclass=MetaEnum):
    """Ways of constructing input graph"""

    FIGNET = "fignet"
    FIGNET_ACT = "fignet_act"


class FIGNetNodeType(MetaEnum):
    MESH = "mesh"
    OBJECT = "object"


@dataclass
class GraphMetaData:
    node_dim: int
    edge_dim: int


@dataclass
class GraphBuildResult:
    node_attr_dict: Dict[NodeType, np.ndarray]
    edge_index_dict: Dict[EdgeType, np.ndarray]
    edge_attr_dict: Dict[EdgeType, np.ndarray]
