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

import time
from typing import Dict

import numpy as np
import torch
import tqdm
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContext, MjSim

from fignet.mujoco_extensions.mj_sim_learned import MjSimLearned
from fignet.simulator import LearnedSimulator
from fignet.utils.conversion import to_numpy


def rollout(
    gnn_model: LearnedSimulator,
    init_obj_poses: np.ndarray,
    obj_ids: Dict[str, int],
    mujoco_xml: dict,
    nsteps: int,
):
    if isinstance(init_obj_poses, torch.Tensor):
        init_obj_poses = to_numpy(init_obj_poses)
    trajectory = np.vstack([init_obj_poses[-1, ...][None, :]])
    sim = MjSimLearned.from_xml_string(mujoco_xml)
    sim.set_gnn_backend(gnn_model)
    sim.set_state(
        positions=init_obj_poses[:, :, :3],
        quaternions=init_obj_poses[:, :, 3:],
        obj_ids=obj_ids,
    )
    for _ in tqdm.tqdm(
        range(nsteps - init_obj_poses.shape[0]), desc="sampling rollout"
    ):
        sim.step(backend="gnn")
        pos, quat = sim.get_body_states()
        obj_poses = np.concatenate([pos, quat], axis=-1)
        trajectory = np.vstack([trajectory, obj_poses[None, :]])

    return np.asarray(trajectory)


def visualize_trajectory(
    mujoco_xml: str,
    traj: np.ndarray,
    obj_ids: dict,
    height: int = 480,
    width: int = 640,
    off_screen: bool = False,
):
    sim = MjSim.from_xml_string(mujoco_xml)

    render_context = MjRenderContext(sim)
    sim.add_render_context(render_context)
    viewer = OpenCVRenderer(sim)
    dt = 0.002  # TODO
    seq_length = traj.shape[0]
    if off_screen:
        screens = []
    for t in range(seq_length):
        for name, ob_id in obj_ids.items():
            pose = traj[t, ob_id, :]
            # bid = sim.model.body_name2id(name)
            q_id = ob_id
            # q_id = sim.model.body_jntadr[bid]
            sim.data.qpos[q_id * 7 : q_id * 7 + 3] = pose[:3]
            sim.data.qpos[q_id * 7 + 3 : q_id * 7 + 7] = pose[3:][
                [3, 0, 1, 2]
            ]  # xyzw -> wxyz
        sim.forward()
        if not off_screen:
            viewer.render()
            time.sleep(dt)
        else:
            im = sim.render(
                camera_name=viewer.camera_name,
                height=height,
                width=width,
            )
            # write frame to window
            im = np.flip(im, axis=0)
            screens.append(im)
    if off_screen:
        return np.array(screens)
