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
from typing import Optional

import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from fignet.data.scene_info import deserialize_scene_info
from fignet.data.transform import ToHeteroGraph
from fignet.graph_builders import GraphBuildCfg


class InteractiveGraphDataset(Dataset):
    def __init__(self, root, build_config: GraphBuildCfg, transform=None):
        file_ext = "pkl"
        self._raw_file_names = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f)) and f.endswith(file_ext)
        ]
        pre_transform = Compose(
            [deserialize_scene_info, ToHeteroGraph(build_config)]
        )
        super(InteractiveGraphDataset, self).__init__(
            root, transform, pre_transform
        )

    @property
    def raw_file_names(self):
        return self._raw_file_names

    def load_data(self, raw_path: str):
        try:
            with open(raw_path, "rb") as f:
                raw_data = pickle.load(f)

                return raw_data
        except FileNotFoundError:
            return None

    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        raw_path = self.raw_file_names[idx]
        data = self.load_data(raw_path)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data


class RolloutDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        file_path: str,
        rollout_length: int,
    ):

        assert os.path.isfile(file_path)

        self._data = list(np.load(file_path, allow_pickle=True).values())[0]

        self._rollout_length = rollout_length

        sample = next(iter(self._data))
        if rollout_length > sample["pos"].shape[0]:
            raise ValueError(
                f"rollout length has to be smaller than {sample['pos'].shape[0]}"
            )
        self._data_lengths = [
            x["pos"].shape[0] - rollout_length for x in self._data
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

        self._length = sum(self._data_lengths)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        trajectory_idx = np.searchsorted(
            self._precompute_cumlengths - 1, idx, side="left"
        )
        start_of_selected_trajectory = (
            self._precompute_cumlengths[trajectory_idx - 1]
            if trajectory_idx != 0
            else 0
        )
        time_idx = self._rollout_length + (idx - start_of_selected_trajectory)

        start = time_idx - self._rollout_length
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

        return {
            "pose_addr": obj_ids,
            "pose_seq": pose_seq,
            "mujoco_xml": mujoco_xml,
        }


def extract_sample_collate_fn(batch):
    return batch[0]


def create_datasets(
    rollout_datapath: Optional[str] = None,
    rollout_length: Optional[int] = None,
    val_ratio: Optional[float] = 0.2,
    **kwargs,
):
    dataset = InteractiveGraphDataset(**kwargs)
    num_val = int(len(dataset) * val_ratio)
    num_train = len(dataset) - num_val
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    rollout_dataset = RolloutDataset(
        file_path=rollout_datapath, rollout_length=rollout_length
    )

    return {
        "train": train_dataset,
        "val": val_dataset,
        "rollout": rollout_dataset,
    }


def create_dataloaders(
    batch_size: int,
    rollout_datapath: Optional[str] = None,
    rollout_length: Optional[int] = None,
    val_ratio: Optional[float] = 0.2,
    num_workers: int = 0,
    **kwargs,
):
    dataset = InteractiveGraphDataset(**kwargs)
    rollout_dataloader = None
    if rollout_datapath is not None and rollout_length is not None:

        rollout_dataset = RolloutDataset(
            file_path=rollout_datapath, rollout_length=rollout_length
        )
        rollout_dataloader = torch.utils.data.DataLoader(
            dataset=rollout_dataset,
            batch_size=1,
            collate_fn=extract_sample_collate_fn,
            num_workers=1,
            shuffle=True,
        )
    num_val = int(len(dataset) * val_ratio)
    num_train = len(dataset) - num_val

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    out = {"train": train_loader, "val": val_loader}
    if rollout_dataloader is not None:
        out.update({"rollout": rollout_dataloader})
    return out
