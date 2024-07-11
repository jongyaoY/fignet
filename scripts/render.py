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

import json
import os

import numpy as np
import torch
from moviepy import editor as mpy
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
from scipy.spatial.transform import Rotation as R

from fignet.scene import Scene
from fignet.simulator import LearnedSimulator
from fignet.utils import rollout, visualize_trajectory

data_path = "datasets/three_bodies_1m"
meta_data_path = os.path.join(data_path, "metadata.json")
with open(meta_data_path) as f:
    metadata = json.load(f)
scene = Scene(metadata["scene_config"])

model_path = "log/202407091835/models/weights_itr_306000.ckpt"
video_path = "log/video"
mujoco_model_path = "test_models/mujoco_scene.xml"
dyn_bodies = ["cube", "bread", "bottle"]
sim = MjSim.from_xml_file(mujoco_model_path)
sim.add_render_context(MjRenderContextOffscreen(sim, 0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = 0.002

ep_length = 400
num_ep = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_to_numpy(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = list_to_numpy(v)
        else:
            d[k] = np.asarray(v)
    return d


def test_simulator():
    latent_dim = 128
    learned_sim = LearnedSimulator(
        mesh_dimensions=3,
        latent_dim=latent_dim,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        noise_std=1e-3,
        device=device,
    )
    learned_sim.load(model_path)
    learned_sim.to(device)
    learned_sim.eval()
    input_seq_length = 2
    for ep_i in range(num_ep):
        # Randomize objs
        init_obj_poses = np.empty((input_seq_length, len(dyn_bodies), 7))
        obj_ids = {}
        for i, name in enumerate(dyn_bodies):
            obj_ids[name] = i
        for i, name in enumerate(dyn_bodies):
            rand_pos = np.zeros(3)
            rand_pos[:2] = np.random.uniform(-0.2, 0.2, 2)
            rand_pos[2] = np.random.uniform(0.1, 0.2)
            rand_quat = R.random().as_quat()[[3, 0, 1, 2]]
            bid = sim.model.body_name2id(name)
            q_id = sim.model.body_jntadr[bid]
            sim.model.body_pos[bid] = rand_pos
            sim.data.qpos[q_id * 7 : q_id * 7 + 3] = rand_pos
            sim.data.qpos[q_id * 7 + 3 : q_id * 7 + 7] = rand_quat
        rand_vel = np.random.randn(len(dyn_bodies) * 6) * 0.8
        sim.data.qvel = rand_vel

        for t in range(input_seq_length):
            sim.forward()
            sim.step()
            for i, name in enumerate(dyn_bodies):
                bid = sim.model.body_name2id(name)
                q_id = sim.model.body_jntadr[bid]
                pos = sim.data.body_xpos[bid]
                quat = sim.data.body_xquat[bid][[1, 2, 3, 0]]
                # velp = sim.data.get_body_xvelp(name)
                # velr = sim.data.get_body_xvelr(name)
                init_obj_poses[t, i, :] = np.concatenate([pos, quat])

        print("Collecting ground truth rollout")
        gt_traj = []
        for t in range(ep_length - input_seq_length + 1):
            sim.forward()
            sim.step()
            obj_poses = np.empty((len(dyn_bodies), 7))
            for obj_id, name in enumerate(dyn_bodies):
                bid = sim.model.body_name2id(name)
                pos = sim.data.body_xpos[bid]
                quat = sim.data.body_xquat[bid][[1, 2, 3, 0]]
                obj_poses[obj_id, :] = np.concatenate([pos, quat])
            gt_traj.append(obj_poses)

        gt_traj = np.asarray(gt_traj)

        pred_traj = rollout(
            learned_sim, init_obj_poses, obj_ids, scene, device, ep_length
        )
        screen_gt = visualize_trajectory(sim, gt_traj, obj_ids, True)
        screen_prd = visualize_trajectory(sim, pred_traj, obj_ids, True)
        screen = np.concatenate([screen_gt, screen_prd], axis=1)
        clip = mpy.ImageSequenceClip(list(screen), fps=60)
        filename = os.path.join(video_path, f"simulated_{ep_i}.gif")
        clip.write_gif(filename)


if __name__ == "__main__":
    test_simulator()
