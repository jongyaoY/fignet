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

import numpy as np
import tqdm

import rigid_fall

parser = argparse.ArgumentParser()
parser.add_argument("--total_steps", type=int, required=True)
parser.add_argument("--ep_len", type=int, required=False, default=200)
parser.add_argument(
    "--render", action="store_true", required=False, default=False
)
parser.add_argument("--data_path", required=False, default="datasets")
parser.add_argument("--dataset_prefix", required=False, default="mujoco_moviA")
args = parser.parse_args()

# Params
num_object_range = [3, 5]
ep_len = args.ep_len
total_steps = args.total_steps
render = args.render
data_path = args.data_path
data_path = os.path.join(os.getcwd(), data_path)
dataset_name = args.dataset_prefix + "_" + str(total_steps)
if not os.path.exists(data_path):
    os.mkdir(data_path)
data_path = os.path.join(data_path, dataset_name)


if __name__ == "__main__":
    data_storage = []
    steps = 0

    pbar = tqdm.tqdm(total=total_steps, desc="Sampling rollouts")
    while steps < total_steps:
        # Randomize object properties
        rand_objs = rigid_fall.choose_objects(num_range=num_object_range)
        scene = rigid_fall.Scene(random_floor=True)
        scene.add_objects(rand_objs)
        try:
            sim, viewer = rigid_fall.init_sim(scene)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
        data = rigid_fall.init_data(scene, sim)
        # Generate one episode
        success = rigid_fall.rollout(
            sim=sim,
            data=data,
            ep_len=ep_len,
            current_steps=steps,
            total_steps=total_steps,
            viewer=viewer,
            pbar=pbar,
            render=render,
        )
        if success:
            steps += data["pos"].shape[0]
            data_storage.append(data)
    pbar.close()
    print(
        f"collected {len(data_storage)} episodes, \
            {sum([data['timestep'].shape[0] for data in data_storage])} steps"
    )
    np.savez(data_path, data_storage)
