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
import concurrent.futures
import logging
import os
import pickle
import sys
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import torch
import tqdm
import yaml

from fignet.data.scene_info import serialize_scene_info
from fignet.mujoco_extensions.mj_sim_learned import MjSimLearned
from fignet.mujoco_extensions.physics_state_tracker import PhysicsStateTracker
from fignet.mujoco_extensions.preprocess import get_scene_info

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--num_workers", type=int, required=True, default=1)
parser.add_argument("--out_path", type=str, required=False)
args = parser.parse_args()

data_path = args.data_path
config_file = args.config_file
num_workers = min(args.num_workers, os.cpu_count())
# batch_size = min(2 * num_workers, 64)
if args.out_path is None:
    output_path = os.path.join(Path(data_path).parent, Path(data_path).stem)
else:
    output_path = args.out_path
device = torch.device("cpu")

logger = logging.getLogger(__name__)
# lock = Lock()


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
            pickle.dump(serialize_scene_info(graph_dict), f)


def process_episode(traj, sim_cfg, ep_i, lock, counter):
    logger.info(f"Started processing episode {ep_i}")

    obj_ids = traj["obj_id"]
    obj_ids = dict(obj_ids.item())

    mujoco_xml = traj["mujoco_xml"]
    all_pos = traj["pos"]
    all_quats = traj["quat"]
    input_seq_len = sim_cfg["input_seq_length"]
    seq_len = all_pos.shape[0]
    ep_len = seq_len - input_seq_len - 1
    sim = MjSimLearned.from_xml_string(str(mujoco_xml))
    sim.forward()
    tracker = PhysicsStateTracker(
        sim=sim, security_margin=sim_cfg["collision_radius"]
    )
    for t in range(seq_len - input_seq_len - 1):
        # for t in tqdm.tqdm(range(seq_len - input_seq_len - 1)):
        # Include targets
        positions = all_pos[t : t + input_seq_len + 1, ...]
        quaternions = all_quats[t : t + input_seq_len + 1, ...]
        sim.set_state(
            positions=positions[-2, ...],
            quaternions=quaternions[-2, ...],
            obj_ids=obj_ids,
        )
        collisions = tracker.detect_collisions(bidirectional=True)
        scn_info = get_scene_info(
            model=sim.model,
            body_meshes=tracker.body_meshes,
            properties=tracker.properties,
            obj_positions=positions,
            obj_quaternions=quaternions,
            pose_addr=obj_ids,
            collisions=collisions,
            contains_targets=True,
        )
        save_graph(scn_info, ep_len * ep_i + t, output_path)
        with lock:
            counter.value += 1


def process_wrapper(args):
    """Unpack arguments and call process_episode."""
    # traj, sim_cfg, ep_i, lock, pbar = args
    process_episode(*args)


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
        data = list(np.load(data_path, allow_pickle=True).values())[0]
    except FileNotFoundError as e:
        print(e)
    sim_cfg = config.pop("simulator")
    data_len = len(data) * (
        data[0]["pos"].shape[0] - sim_cfg["input_seq_length"] - 1
    )
    pbar = tqdm.tqdm(total=data_len, desc="Processing")

    with Manager() as manager:
        counter = manager.Value("i", 1)
        lock = manager.Lock()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            tasks = [
                (traj, sim_cfg, i, lock, counter)
                for i, traj in enumerate(data)
            ]

            # Submit tasks to the executor without storing futures since no
            # return is needed
            for task in tasks:
                executor.submit(process_wrapper, task)

            while True:
                pbar.n = counter.value
                pbar.refresh()
                if counter.value >= data_len:
                    break
