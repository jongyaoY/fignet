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

import dataclasses
from typing import Union

import numpy as np
import torch

from fignet.data import HeteroGraph
from fignet.types import EdgeType, Graph, NodeType


def to_tensor(array: Union[np.ndarray, torch.Tensor], device: str = None):
    if isinstance(array, torch.Tensor):
        if array.dtype == torch.float64:
            tensor = array.float()
        else:
            tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.int64:
            tensor = torch.from_numpy(array).long()
        else:
            tensor = torch.from_numpy(array).float()
    elif isinstance(array, dict):
        return dict_to_tensor(array, device)
    else:
        raise TypeError(f"Cannot conver {type(array)} to tensor")

    if device:
        return tensor.to(device)
    else:
        return tensor


def dict_to_tensor(d: dict, device=None):
    new_dict = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = dict_to_tensor(v, device)
        else:
            if device is None:
                new_dict[k] = torch.FloatTensor(v)
            else:
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.long:
                        new_dict[k] = v.to(device)
                    else:
                        new_dict[k] = v.float().to(device)
                elif isinstance(v, np.ndarray):
                    if v.dtype == np.int64:
                        new_dict[k] = torch.from_numpy(v).long().to(device)
                    else:
                        new_dict[k] = torch.from_numpy(v).float().to(device)
                elif isinstance(v, int):
                    new_dict[k] = torch.LongTensor([v]).to(device)
                else:
                    raise TypeError(f"Unexpected data type: {type(v)}")

    return new_dict


def dataclass_to_tensor(d, device=None):

    if isinstance(d, np.ndarray) or isinstance(d, torch.Tensor):
        return to_tensor(d, device=device)
    elif dataclasses.is_dataclass(d):
        for f in dataclasses.fields(d):
            setattr(
                d,
                f.name,
                dataclass_to_tensor(getattr(d, f.name), device=device),
            )
        return d
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = dataclass_to_tensor(v, device=device)
        return d


class ToHeteroData(object):

    def __init__(self):
        pass

    def __call__(self, graph: Graph):
        seq_len = graph.node_sets[NodeType.MESH].position.shape[0]
        m_features = []
        o_features = []
        for i in range(seq_len):
            if i + 1 < seq_len:
                m_vel = (
                    graph.node_sets[NodeType.MESH].position[i + 1, ...]
                    - graph.node_sets[NodeType.MESH].position[i, ...]
                )
                o_vel = (
                    graph.node_sets[NodeType.OBJECT].position[i + 1, ...]
                    - graph.node_sets[NodeType.OBJECT].position[i, ...]
                )
                m_features.append(m_vel)
                o_features.append(o_vel)
        m_features.append(graph.node_sets[NodeType.MESH].properties)
        m_features.append(graph.node_sets[NodeType.MESH].kinematic)
        o_features.append(graph.node_sets[NodeType.OBJECT].properties)
        o_features.append(graph.node_sets[NodeType.OBJECT].kinematic)
        m_features = torch.cat(m_features, dim=-1)
        o_features = torch.cat(o_features, dim=-1)
        data = HeteroGraph()
        data["mesh"].x = m_features
        data["mesh"].kinematic = graph.node_sets[NodeType.MESH].kinematic
        data["mesh"].y = graph.node_sets[NodeType.MESH].target
        data["object"].x = o_features
        data["object"].y = graph.node_sets[NodeType.OBJECT].target

        data["mesh", "m-o", "object"].edge_index = graph.edge_sets[
            EdgeType.MESH_OBJ
        ].index
        data["mesh", "m-o", "object"].edge_attr = graph.edge_sets[
            EdgeType.MESH_OBJ
        ].attribute
        data["object", "o-m", "mesh"].edge_index = graph.edge_sets[
            EdgeType.OBJ_MESH
        ].index
        data["object", "o-m", "mesh"].edge_attr = graph.edge_sets[
            EdgeType.OBJ_MESH
        ].attribute

        return data


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        # convert numpy arrays to pytorch tensors
        if isinstance(sample, dict):
            return dict_to_tensor(sample, self.device)
        elif isinstance(sample, Graph):
            return dataclass_to_tensor(sample, self.device)
        else:
            raise TypeError(f"Cannot convert {type(sample)} to tensor.")
