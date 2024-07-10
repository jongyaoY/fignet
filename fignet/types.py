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
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Dict, Union

import numpy as np
from dacite import from_dict
from torch import Tensor

TensorType = Union[np.ndarray, Tensor]


class MetaEnum(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class KinematicType(enum.IntEnum):
    STATIC = 0
    DYNAMIC = 1
    SIZE = 1


class NodeType(enum.Enum, metaclass=MetaEnum):
    MESH = "mesh"
    OBJECT = "object"


class EdgeType(enum.Enum, metaclass=MetaEnum):
    MESH_MESH = "mesh-mesh"
    MESH_OBJ = "mesh-object"
    OBJ_MESH = "object-mesh"
    FACE_FACE = "face-face"


def key_to_string(key):
    if isinstance(key, str):
        return key
    elif isinstance(key, enum.Enum):
        return key.value
    else:
        raise ValueError(f"{type(key)} not supported")


def string_to_enum(k_str, enum_list):
    for e in enum_list:
        if k_str in e:
            return e(k_str)


def to_dict(item):
    d = {}
    if isinstance(item, Dict):
        for k, v in item.items():
            d.update({key_to_string(k): to_dict(v)})
        return d
    elif is_dataclass(item):
        return to_dict(asdict(item))
    else:
        return item


@dataclass
class NodeFeature:
    position: TensorType
    kinematic: TensorType
    properties: TensorType
    target: TensorType = None


@dataclass
class Edge:
    attribute: TensorType
    index: TensorType


@dataclass
class Graph:
    node_sets: Dict[NodeType, NodeFeature] = field(default_factory=lambda: {})
    edge_sets: Dict[EdgeType, Edge] = field(default_factory=lambda: {})

    def to_dict(self):
        return to_dict(self)

    def from_dict(self, d):
        for f in fields(self):
            field_name = f.name
            if field_name not in d:
                continue
            if field_name == "node_sets":
                d_cls = NodeFeature
            elif field_name == "edge_sets":
                d_cls = Edge
            for k, v in d[field_name].items():
                getattr(self, field_name).update(
                    {
                        string_to_enum(k, [NodeType, EdgeType]): from_dict(
                            d_cls, v
                        )
                    }
                )
