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


import os
import pickle
from dataclasses import fields
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils

from fignet.scene import Scene
from fignet.types import EdgeType, Graph, NodeType


def collate_fn(batch: List[Graph]):
    """Merge batch of graphs into one graph"""
    if len(batch) == 1:
        return batch[0]
    else:
        batch_graph = batch.pop(0)
        m_node_offset = batch_graph.node_sets[NodeType.MESH].kinematic.shape[0]
        o_node_offset = batch_graph.node_sets[NodeType.OBJECT].kinematic.shape[
            0
        ]
        for graph in batch:
            for node_typ in graph.node_sets.keys():
                for field in fields(graph.node_sets[node_typ]):
                    if field.name == "position":
                        cat_dim = 1
                    else:
                        cat_dim = 0
                    setattr(
                        batch_graph.node_sets[node_typ],
                        field.name,
                        torch.cat(
                            [
                                getattr(
                                    batch_graph.node_sets[node_typ], field.name
                                ),
                                getattr(graph.node_sets[node_typ], field.name),
                            ],
                            dim=cat_dim,
                        ),
                    )
            for edge_typ in graph.edge_sets.keys():
                if (
                    edge_typ == EdgeType.MESH_MESH
                    or edge_typ == EdgeType.FACE_FACE
                ):
                    graph.edge_sets[edge_typ].index += m_node_offset
                elif edge_typ == EdgeType.OBJ_MESH:
                    graph.edge_sets[edge_typ].index[0, :] += o_node_offset
                    graph.edge_sets[edge_typ].index[1, :] += m_node_offset
                elif edge_typ == EdgeType.MESH_OBJ:
                    graph.edge_sets[edge_typ].index[0, :] += m_node_offset
                    graph.edge_sets[edge_typ].index[1, :] += o_node_offset
                else:
                    raise TypeError(f"Unknown edge type {edge_typ}")
                # Concatenate
                batch_graph.edge_sets[edge_typ].index = torch.cat(
                    [
                        batch_graph.edge_sets[edge_typ].index,
                        graph.edge_sets[edge_typ].index,
                    ],
                    dim=1,
                )
                batch_graph.edge_sets[edge_typ].attribute = torch.cat(
                    [
                        batch_graph.edge_sets[edge_typ].attribute,
                        graph.edge_sets[edge_typ].attribute,
                    ],
                    dim=0,
                )
            m_node_offset += graph.node_sets[NodeType.MESH].kinematic.shape[0]
            o_node_offset += graph.node_sets[NodeType.OBJECT].kinematic.shape[
                0
            ]

        return batch_graph


class MujocoDataset(torch.utils.data.Dataset):
    """Load Mujoco dataset"""

    def __init__(
        self,
        path: str,
        input_sequence_length: int,
        mode: str,
        config: dict = None,
        transform=None,
    ):
        # If raw data is given, need to calculate graph connectivity on the fly
        if os.path.isfile(path):
            self._load_raw_data = True

            self._data = list(np.load(path, allow_pickle=True).values())[0]
            self._dimension = self._data[0]["pos"].shape[-1]
            self._target_length = 1
            self._input_sequence_length = input_sequence_length

            self._data_lengths = [
                x["pos"].shape[0] - input_sequence_length - self._target_length
                for x in self._data
            ]
            self._length = sum(self._data_lengths)

            # pre-compute cumulative lengths
            # to allow fast indexing in __getitem__
            self._precompute_cumlengths = [
                sum(self._data_lengths[:x])
                for x in range(1, len(self._data_lengths) + 1)
            ]
            self._precompute_cumlengths = np.array(
                self._precompute_cumlengths, dtype=int
            )
        # Directly load pre-calculated graphs and save time
        elif os.path.isdir(path):
            self._load_raw_data = False

            self._graph_ext = "pkl"
            self._file_list = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
                and f.endswith(self._graph_ext)
            ]
            self._file_list.sort(key=lambda f: int(Path(f).stem.split("_")[1]))
            self._length = len(self._file_list)
        else:
            raise FileNotFoundError(f"{path} not found")

        self._transform = transform
        self._mode = mode
        if config is not None:
            self._config = config
        else:
            self._config = {}

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._mode == "sample":
            return self._get_sample(idx)
        elif self._mode == "trajectory":
            return self._get_trajectory(idx)

    def _load_graph(self, graph_file):
        try:
            with open(graph_file, "rb") as f:
                sample_dict = pickle.load(f)
                graph = Graph()
                graph.from_dict(sample_dict)
            return graph

        except FileNotFoundError as e:
            print(e)
            return None

    def _get_sample(self, idx):
        """Sample one step"""
        if self._load_raw_data:
            trajectory_idx = np.searchsorted(
                self._precompute_cumlengths - 1, idx, side="left"
            )
            # Compute index of pick along time-dimension of trajectory.
            start_of_selected_trajectory = (
                self._precompute_cumlengths[trajectory_idx - 1]
                if trajectory_idx != 0
                else 0
            )
            time_idx = self._input_sequence_length + (
                idx - start_of_selected_trajectory
            )

            start = time_idx - self._input_sequence_length
            end = time_idx
            obj_ids = dict(self._data[trajectory_idx]["obj_id"].item())
            positions = self._data[trajectory_idx]["pos"][
                start:end
            ]  # (seq_len, n_obj, 3) input sequence
            quats = self._data[trajectory_idx]["quat"][
                start:end
            ]  # (seq_len, n_obj, 4) input sequence
            target_posisitons = self._data[trajectory_idx]["pos"][time_idx]
            target_quats = self._data[trajectory_idx]["quat"][time_idx]
            poses = np.concatenate([positions, quats], axis=-1)
            target_poses = np.concatenate(
                [target_posisitons, target_quats], axis=-1
            )

            scene_config = dict(self._data[trajectory_idx]["meta_data"].item())

            connectivity_radius = self._config.get("connectivity_radius")
            if connectivity_radius is not None:
                scene_config.update(
                    {"connectivity_radius": connectivity_radius}
                )

            noise_std = self._config.get("noise_std")
            if noise_std is not None:
                scene_config.update({"noise_std": noise_std})

            scn = Scene(scene_config)
            scn.synchronize_states(
                obj_poses=poses,
                obj_ids=obj_ids,
            )
            graph = scn.to_graph(
                target_poses=target_poses,
                obj_ids=obj_ids,
                noise=True,
            )

            if self._transform is not None:
                graph = self._transform(graph)
            return graph
        else:
            if os.path.exists(self._file_list[idx]):
                return self._transform(self._load_graph(self._file_list[idx]))
            else:
                raise FileNotFoundError

    def _get_trajectory(self, idx):
        """Sample continuous steps"""
        trajectory_idx = np.searchsorted(
            self._precompute_cumlengths - 1, idx, side="left"
        )
        start_of_selected_trajectory = (
            self._precompute_cumlengths[trajectory_idx - 1]
            if trajectory_idx != 0
            else 0
        )
        time_idx = self._input_sequence_length + (
            idx - start_of_selected_trajectory
        )

        start = time_idx - self._input_sequence_length
        end = time_idx
        obj_ids = self._data[trajectory_idx]["obj_id"]
        obj_ids = dict(obj_ids.item())
        mujoco_xml = str(self._data[trajectory_idx]["mujoco_xml"])
        scene_config = dict(self._data[trajectory_idx]["meta_data"].item())

        try:
            connectivity_radius = self._config["connectivity_radius"]
            scene_config.update({"connectivity_radius": connectivity_radius})
        except KeyError:
            pass
        positions = self._data[trajectory_idx]["pos"][
            start:end
        ]  # (seq_len, n_obj, 3) input sequence
        quats = self._data[trajectory_idx]["quat"][
            start:end
        ]  # (seq_len, n_obj, 4) input sequence
        pose_seq = np.concatenate([positions, quats], axis=-1)

        traj = {"obj_ids": obj_ids, "pose_seq": pose_seq}
        if self._transform is not None:
            traj = self._transform(traj)
        return traj, mujoco_xml, scene_config


# def get_data_loader(
#     dataset: MujocoDataset, batch_size: int, device, num_workers: int = 0
# ):
#     if device == torch.device("cuda"):
#         sampler = DistributedSampler(dataset, shuffle=True)
#         return torch.utils.data.DataLoader(
#             dataset=dataset,
#             sampler=sampler,
#             batch_size=batch_size,
#             pin_memory=True,
#             collate_fn=collate_fn,
#             num_workers=num_workers,
#         )
#     else:
#         return torch.utils.data.DataLoader(
#             dataset=dataset,
#             batch_size=batch_size,
#             pin_memory=True,
#             collate_fn=collate_fn,
#             num_workers=num_workers,
#         )
