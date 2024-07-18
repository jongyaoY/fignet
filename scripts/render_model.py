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
import torch
from moviepy import editor as mpy

import fignet
import rigid_fall

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", required=True)
parser.add_argument("--video_path", required=False, default="log/video")
parser.add_argument("--off_screen", required=False, action="store_true")
parser.add_argument("--ep_length", required=False, type=int, default=200)
parser.add_argument("--num_ep", required=False, type=int, default=5)

args = parser.parse_args()

model_path = args.model_path
video_path = args.video_path
off_screen = args.off_screen
ep_length = args.ep_length
num_ep = args.num_ep

video_path = os.path.join(os.getcwd(), video_path)

if off_screen:
    if not os.path.exists(video_path):
        os.mkdir(video_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_simulator(learned_sim: fignet.LearnedSimulator, off_screen=True):

    input_seq_length = 2

    mujoco_scene = None
    sim = None
    for _ in range(10):
        mujoco_scene = rigid_fall.Scene(random_floor=True)
        mujoco_scene.add_objects(rigid_fall.choose_objects())
        try:
            sim, _ = rigid_fall.init_sim(mujoco_scene)
            break
        except ValueError:
            mujoco_scene = None
            sim = None
            continue
    if sim is None:
        raise RuntimeError("Failed to initialize mujoco simulation")

    gt_data = rigid_fall.init_data(scene=mujoco_scene, sim=sim)
    rigid_fall.rollout(sim, gt_data, ep_length)

    obj_id = dict(gt_data["obj_id"].item())
    meta_data = dict(gt_data["meta_data"].item())
    gt_traj = np.concatenate([gt_data["pos"], gt_data["quat"]], axis=2)
    pred_traj = fignet.rollout(
        learned_sim,
        gt_traj[:input_seq_length, ...],
        obj_id,
        fignet.Scene(meta_data),
        device,
        ep_length,
    )
    screen_gt = fignet.visualize_trajectory(
        mujoco_scene.to_xml(),
        gt_traj[input_seq_length - 1 :, ...],
        obj_id,
        off_screen=off_screen,
    )
    screen_prd = fignet.visualize_trajectory(
        mujoco_scene.to_xml(), pred_traj, obj_id, off_screen=off_screen
    )

    return screen_gt, screen_prd


if __name__ == "__main__":
    latent_dim = 128
    learned_sim = fignet.LearnedSimulator(
        mesh_dimensions=3,
        latent_dim=latent_dim,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=latent_dim,
        noise_std=1e-4,
        device=device,
    )
    learned_sim.load(model_path)
    learned_sim.to(device)
    learned_sim.eval()
    for ep_i in range(num_ep):
        screens = render_simulator(learned_sim, off_screen)
        if off_screen:
            screen = np.concatenate([screens[0], screens[1]], axis=1)
            clip = mpy.ImageSequenceClip(list(screen), fps=60)
            filename = os.path.join(video_path, f"simulated_{ep_i}.gif")
            clip.write_gif(filename)
