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
import torch
import torch.utils
import tqdm

from fignet.scene import Scene
from fignet.utils import dict_to_tensor


def collate_fn(batch_dict):
    if len(batch_dict) == 1:
        return batch_dict[0]
    else:
        raise NotImplementedError()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        # convert numpy arrays to pytorch tensors
        return dict_to_tensor(sample, self.device)


class MujocoDataset(torch.utils.data.Dataset):
    """Load Mujoco dataset"""

    def __init__(self, path, input_sequence_length, mode: str, transform=None):
        # meta_data_path = os.path.join(path, "metadata.json")
        # with open(meta_data_path) as f:
        #     self.metadata = json.load(f)
        # self._scene = Scene(self.metadata["scene_config"])
        # if mode == "sample":
        #     data_name = "dataset.npz"
        # elif mode == "trajectory":
        #     data_name = "test_dataset.npz"
        # data_path = os.path.join(path, data_name)
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
        self._transform = transform
        self._mode = mode

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._mode == "sample":
            return self._get_sample(idx)
        elif self._mode == "trajectory":
            return self._get_trajectory(idx)

    def _get_sample(self, idx):
        """Sample one step"""
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

        scn = Scene(scene_config)
        scn.synchronize_states(
            obj_poses=poses,
            obj_ids=obj_ids,
        )
        graph = scn.to_graph(target_poses=target_poses, obj_ids=obj_ids)
        # graph = ToTensor()(graph) data = HeteroData() data['mesh'].x =
        # graph['m_x'] data['mesh'].y = graph['m_y'] data['object'].x =
        # graph['o_x'] data['object'].y = graph['o_y'] data['mesh', 'regular',
        # 'mesh'].edge_index = graph['index']['mm'] data['mesh', 'regular',
        # 'object'].edge_index = graph['index']['mo'] data['object', 'regular',
        # 'mesh'].edge_index = graph['index']['om']
        # # data['mesh', 'face', 'mesh'].edge_index = graph['index']['ff']

        # data['mesh', 'regular', 'mesh'].edge_attr =
        # graph['edge_features']['mm'] data['mesh', 'regular',
        # 'object'].edge_attr = graph['edge_features']['mo'] data['object',
        # 'regular', 'mesh'].edge_attr = graph['edge_features']['om']
        # data['mesh', 'face', 'mesh'].edge_attr = graph['edge_features']['ff']
        if self._transform is not None:
            graph = self._transform(graph)
        return graph

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


def get_normalization_stats(dataset):
    stats = {
        "vel": {"mean": np.zeros(3), "std": np.zeros(3)},
        "acc": {"mean": np.zeros(3), "std": np.zeros(3)},
    }
    for sample in tqdm.tqdm(dataset, desc="Calculating normalization stats"):
        stats["vel"]["mean"] += torch.mean(sample["m_x"][:, :3], axis=0)
        stats["acc"] += torch.mean(sample["m_y"][:, :3], axis=0)
    stats["vel"] = stats["vel"] / len(dataset)
    stats["acc"] = stats["acc"] / len(dataset)
