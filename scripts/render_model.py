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
from PIL import Image

import fignet
import rigid_fall

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", required=True)
parser.add_argument("--leave_out_mm", action="store_true")
parser.add_argument(
    "--split_video",
    action="store_true",
    help="store ground truth and simulation videos separately",
)
parser.add_argument(
    "--only_ground_truth",
    action="store_true",
    help="generate only ground truth videos",
)
parser.add_argument("--load_from", required=False, default="")
parser.add_argument("--input_seq_length", required=False, type=int, default=3)
parser.add_argument("--video_path", required=False, default="log/video")
parser.add_argument("--internal_steps", required=False, type=int, default=10)
parser.add_argument("--skip_frame", required=False, type=int, default=10)
parser.add_argument("--max_num_object", required=False, type=int, default=10)
parser.add_argument("--min_num_object", required=False, type=int, default=3)
parser.add_argument("--off_screen", required=False, action="store_true")
parser.add_argument("--use_cuda", required=False, action="store_true")
parser.add_argument("--ep_length", required=False, type=int, default=200)
parser.add_argument("--num_ep", required=False, type=int, default=5)
parser.add_argument("--height", required=False, type=int, default=240)
parser.add_argument("--width", required=False, type=int, default=320)
args = parser.parse_args()

leave_out_mm = args.leave_out_mm
load_from = args.load_from
model_path = args.model_path
video_path = args.video_path
off_screen = args.off_screen
ep_length = args.ep_length
use_cuda = args.use_cuda
only_ground_truth = args.only_ground_truth
num_ep = args.num_ep
height = args.height
width = args.width
split_video = args.split_video
input_seq_length = args.input_seq_length
video_path = os.path.join(os.getcwd(), video_path)
duration = 1

num_object_range = [args.min_num_object, args.max_num_object]
# TODO: should be same as the dataset on which the simulator has been trained
internal_steps = args.internal_steps
SPAWN_REGION = [(-0.15, -0.15, 0.1), (0.15, 0.15, 0.5)]

skip_frame = args.skip_frame
if off_screen:
    if not os.path.exists(video_path):
        os.mkdir(video_path)

if only_ground_truth:
    split_video = True

device = torch.device(
    "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
)


def sample_ground_truth():
    mujoco_scene = None
    sim = None
    for _ in range(10):
        mujoco_scene = rigid_fall.Scene(random_floor=True)
        mujoco_scene.add_objects(
            rigid_fall.random_objects(num_range=num_object_range)
        )
        try:
            sim, _ = rigid_fall.init_sim(mujoco_scene, has_renderer=False)
            break
        except ValueError:
            mujoco_scene = None
            sim = None
            continue
    if sim is None:
        raise RuntimeError("Failed to initialize mujoco simulation")
    gt_data = rigid_fall.init_data(scene=mujoco_scene, sim=sim)

    rigid_fall.rollout(
        sim,
        gt_data,
        ep_length,
        internal_steps=internal_steps,
        spawn_region=SPAWN_REGION,
    )
    return gt_data


def render_simulator(
    learned_sim: fignet.LearnedSimulator, ep_i, off_screen=True
):

    # mujoco_scene = None
    # sim = None
    # for _ in range(10):
    #     mujoco_scene = rigid_fall.Scene(random_floor=True)
    #     mujoco_scene.add_objects(
    #         rigid_fall.random_objects(num_range=num_object_range)
    #     )
    #     try:
    #         sim, _ = rigid_fall.init_sim(mujoco_scene, has_renderer=False)
    #         break
    #     except ValueError:
    #         mujoco_scene = None
    #         sim = None
    #         continue
    # if sim is None:
    #     raise RuntimeError("Failed to initialize mujoco simulation")

    # successful_rollout = False
    # for _ in range(10):
    #     gt_data = rigid_fall.init_data(scene=mujoco_scene, sim=sim)

    #     successful_rollout = rigid_fall.rollout(
    #         sim,
    #         gt_data,
    #         ep_length,
    #         internal_steps=internal_steps,
    #         spawn_region=SPAWN_REGION,
    #     )
    #     if successful_rollout:
    #         break
    # if not successful_rollout:
    #     raise RuntimeError("Failed to sample rollout")
    if load_from != "":
        data = list(np.load(load_from, allow_pickle=True).values())[0]
        gt_data = data[ep_i]
    else:
        gt_data = sample_ground_truth()

    mujoco_xml = str(gt_data["mujoco_xml"])
    obj_id = dict(gt_data["obj_id"].item())
    meta_data = dict(gt_data["meta_data"].item())
    meta_data["connectivity_radius"] = 0.01  # TODO: load from config
    # meta_data["noise_std"] = 3e-5
    gt_traj = np.concatenate([gt_data["pos"], gt_data["quat"]], axis=2)
    screen_gt = fignet.visualize_trajectory(
        mujoco_xml,
        gt_traj[input_seq_length - 1 :, ...],
        obj_id,
        height=height,
        width=width,
        off_screen=off_screen,
    )
    if not only_ground_truth:
        pred_traj = fignet.rollout(
            learned_sim,
            gt_traj[:input_seq_length, ...],
            obj_id,
            fignet.Scene(meta_data),
            device,
            ep_length,
        )

        screen_prd = fignet.visualize_trajectory(
            mujoco_xml,
            pred_traj,
            obj_id,
            height=height,
            width=width,
            off_screen=off_screen,
        )
    else:
        screen_prd = None

    return screen_gt, screen_prd


if __name__ == "__main__":
    latent_dim = 128
    learned_sim = fignet.LearnedSimulator(
        mesh_dimensions=3,
        latent_dim=latent_dim,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        input_seq_length=input_seq_length,
        mlp_hidden_dim=latent_dim,
        device=device,
        leave_out_mm=leave_out_mm,
    )
    learned_sim.load(model_path)
    learned_sim.to(device)
    learned_sim.eval()
    for ep_i in range(num_ep):
        screens = render_simulator(learned_sim, ep_i, off_screen)
        if off_screen:
            if split_video:
                filename_gt = os.path.join(
                    video_path, f"ground_truth_{ep_i}.gif"
                )
                filename_sim = os.path.join(
                    video_path, f"simulation_{ep_i}.gif"
                )
                imgs = [
                    Image.fromarray(img)
                    for img in list(screens[0])[::skip_frame]
                ]
                imgs[0].save(
                    filename_gt,
                    save_all=True,
                    append_images=imgs[1:],
                    duration=duration,
                    loop=0,
                )
                if not only_ground_truth:
                    imgs = [
                        Image.fromarray(img)
                        for img in list(screens[1])[::skip_frame]
                    ]
                    imgs[0].save(
                        filename_sim,
                        save_all=True,
                        append_images=imgs[1:],
                        duration=duration,
                        loop=0,
                    )
            else:
                screen = np.concatenate([screens[0], screens[1]], axis=1)
                filename = os.path.join(
                    video_path, f"ground_truth_simulation_{ep_i}.gif"
                )
                imgs = [
                    Image.fromarray(img) for img in list(screen)[::skip_frame]
                ]
                imgs[0].save(
                    filename,
                    save_all=True,
                    append_images=imgs[1:],
                    duration=duration,
                    loop=0,
                )
