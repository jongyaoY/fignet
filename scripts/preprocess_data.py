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

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
import tqdm
import yaml

from fignet.data_loader import MujocoDataset

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--num_workers", type=int, required=True, default=1)

args = parser.parse_args()

data_path = args.data_path
config_file = args.config_file
num_workers = max(args.num_workers, 1)
batch_size = min(2 * num_workers, 64)
output_path = os.path.join(Path(data_path).parent, Path(data_path).stem)
device = torch.device("cpu")


def collate_fn(batch):
    return batch


def save_graph(graph, graph_i, save_path):
    if isinstance(graph, list):
        batch_size = len(graph)
        for g_i, g in enumerate(graph):
            i = graph_i * batch_size + g_i
            save_graph(g, i, save_path)
    else:
        if not isinstance(graph, dict):
            graph_dict = graph.to_dict()
        else:
            graph_dict = graph
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        file_name = os.path.join(save_path, f"graph_{graph_i}.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(graph_dict, f)


if __name__ == "__main__":
    try:
        with open(os.path.join(os.getcwd(), args.config_file)) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    print(
        f"Parsing {data_path}. Preprocessed graphs will be stored in {output_path}"
    )
    try:
        sim_cfg = config.pop("simulator")

        dataset = MujocoDataset(
            path=data_path,
            mode="sample",
            input_sequence_length=sim_cfg["input_seq_length"],
            # transform=T.Compose([ToTensor(device)]),
            collision_radius=sim_cfg["collision_radius"],
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        for i, sample in enumerate(
            tqdm.tqdm(data_loader, desc="Preprocessing data")
        ):
            save_graph(sample, i, output_path)
    except FileNotFoundError as e:
        print(e)
