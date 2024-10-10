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
from pathlib import Path

import numpy as np
import torch
import torch.utils

from fignet.mujoco_extensions.mj_sim_learned import MjSimLearned
from fignet.mujoco_extensions.physics_state_tracker import PhysicsStateTracker
from fignet.mujoco_extensions.preprocess import get_scene_info


def collate_fn(batch: list):
    """Merge batch of graphs into one graph"""
    if len(batch) == 1:
        return batch[0]
    else:
        raise NotImplementedError(
            "This collate function is not meant for batch"
        )


class MujocoDataset(torch.utils.data.Dataset):
    """Load Mujoco dataset"""

    def __init__(
        self,
        path: str,
        input_sequence_length: int,
        mode: str,
        collision_radius: float,
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
        self._collision_radius = collision_radius

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._mode == "sample":
            return self._get_sample(idx)
        elif self._mode == "trajectory":
            return self._get_trajectory(idx)

    def _load_scn_info(self, scn_info_file):
        try:
            with open(scn_info_file, "rb") as f:
                scn_info_dict = pickle.load(f)
            return scn_info_dict

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

            sim = MjSimLearned.from_xml_string(
                str(self._data[trajectory_idx]["mujoco_xml"])
            )
            sim.set_state(
                positions=positions[-1, ...],
                quaternions=quats[-1, ...],
                obj_ids=obj_ids,
            )

            tracker = PhysicsStateTracker(
                sim=sim, security_margin=self._collision_radius
            )

            collisions = tracker.detect_collisions(bidirectional=True)

            scn_info = get_scene_info(
                model=sim.model,
                body_meshes=tracker.body_meshes,
                properties=tracker.properties,
                obj_positions=np.vstack(
                    [positions, target_posisitons[np.newaxis, :]]
                ),
                obj_quaternions=np.vstack(
                    [quats, target_quats[np.newaxis, :]]
                ),
                pose_addr=obj_ids,
                collisions=collisions,
                contains_targets=True,
            )
        else:
            if os.path.exists(self._file_list[idx]):
                scn_info = self._load_scn_info(self._file_list[idx])
            else:
                raise FileNotFoundError

        if self._transform is not None:
            out = self._transform(scn_info)
        else:
            out = scn_info

        return out

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
        return traj, mujoco_xml
