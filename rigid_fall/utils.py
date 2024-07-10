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
import tqdm
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContext, MjSim
from scipy.spatial.transform import Rotation as R

from rigid_fall.mesh_object import get_all_objects, get_n_objects
from rigid_fall.scene import Scene


def process_object_properties(prop):
    prop_out = {}
    for k, v in prop.items():
        if k == "priority" or k == "solimp":
            continue
        elif k == "solref":
            restitution = v[1]
            prop_out["restitution"] = restitution
        else:
            prop_out[k] = v
    return prop_out


def choose_objects(all_objects=None, num_range=[3, 5]):
    if all_objects is None:
        all_objects = get_all_objects(random_properties=True)
    num_obj = np.random.randint(low=num_range[0], high=num_range[1])
    num_obj = min(num_obj, len(all_objects))
    objects = np.random.choice(all_objects, num_obj, replace=False)
    return objects.tolist()


def random_objects(num_range=[3, 5]):
    num_obj = np.random.randint(low=num_range[0], high=num_range[1])
    return get_n_objects(num_obj, random_properties=True)


def get_object_names(sim):
    names = []
    for name in sim.model.body_names:
        if name == "world":
            continue
        if "_" in name:
            name = name.split("_")[0]
        names.append(name)
    return names


def init_data(scene: Scene, sim: MjSim):
    object_properties = scene.get_properties()
    object_names = get_object_names(sim)
    data = {
        "obj_id": {},
        "timestep": [],
        "pos": [],
        "quat": [],
        "velp": [],
        "velr": [],
    }

    floor_prop = process_object_properties(object_properties["floor"])
    meta_data = {
        "env": {
            "floor": {
                "type": "box",
                "extents": [3.0, 3.0, 0.5],
                "properties": floor_prop,
                "initial_pose": [0, 0, -0.5 / 2.0],
            }
        },
        "objects": {},
    }
    for i, name in enumerate(object_names):
        data["obj_id"][name] = i
        obj_prop = process_object_properties(object_properties[name])
        mesh_file = obj_prop["mesh"]
        del obj_prop["mesh"]
        meta_data["objects"].update(
            {name: {"mesh": mesh_file, "properties": obj_prop}}
        )
    data["meta_data"] = meta_data
    data["mujoco_xml"] = scene.to_xml()
    return data


def record_step(sim, data):
    data["timestep"].append(sim.data.time)
    pos = []
    quat = []
    velp = []
    velr = []
    for name in sim.model.body_names:
        if name == "world":
            continue
        bid = sim.model.body_name2id(name)
        pos.append(sim.data.body_xpos[bid])
        quat.append(sim.data.body_xquat[bid][[1, 2, 3, 0]])
        velp.append(sim.data.get_body_xvelp(name))
        velr.append(sim.data.get_body_xvelr(name))
    data["pos"].append(np.array(pos))
    data["quat"].append(np.array(quat))
    data["velp"].append(np.array(velp))
    data["velr"].append(np.array(velr))


def init_sim(scene: Scene, has_renderer: bool):

    sim = MjSim.from_xml_string(scene.to_xml())
    if has_renderer:
        render_context = MjRenderContext(sim)
        sim.add_render_context(render_context)

        viewer = OpenCVRenderer(sim)
        viewer.set_camera(sim.model.camera_name2id(scene.camera_name))
    else:
        viewer = None
    return sim, viewer


def rollout(
    sim: MjSim,
    data: dict,
    ep_len: int,
    current_steps: int = None,
    total_steps: int = None,
    internal_steps: int = 1,
    spawn_region: list = None,
    vel_scale: float = 0.5,
    num_sample_trial=100,
    viewer: OpenCVRenderer = None,
    pbar: tqdm.tqdm = None,
    render=False,
):
    body_names = get_object_names(sim)
    # Randomize positions
    successful_sample = False
    for _ in range(num_sample_trial):
        for _, name in enumerate(body_names):
            if spawn_region is None:
                spawn_region = [(-0.17, -0.17, 0.1), (0.17, 0.17, 0.5)]
            rand_pos = np.concatenate(
                [
                    np.random.uniform(
                        spawn_region[0][0], spawn_region[1][0], 1
                    ),  # x pos
                    np.random.uniform(
                        spawn_region[0][1], spawn_region[1][1], 1
                    ),  # y pos
                    np.random.uniform(
                        spawn_region[0][2], spawn_region[1][2], 1
                    ),  # z pos
                ]
            )
            rand_quat = R.random().as_quat()[[3, 0, 1, 2]]
            rand_pose = np.concatenate([rand_pos, rand_quat], axis=-1)
            joint_name = name + "_joint0"
            sim.data.set_joint_qpos(joint_name, rand_pose)
        sim.forward()
        if sim.data.ncon == 0:
            successful_sample = True
            break
    if not successful_sample:
        return False
    # Randomize velocities
    rand_vel = np.random.randn(sim.data.qvel.shape[0]) * vel_scale
    sim.data.qvel = rand_vel
    # Rollout loop
    for t in range(ep_len):
        if current_steps is not None and total_steps is not None:
            if current_steps + t >= total_steps:
                break  # Early stopping
        for _ in range(internal_steps):
            sim.forward()
            sim.step()
        record_step(sim, data)
        if render and viewer is not None:
            viewer.render()

        if pbar is not None:
            pbar.update(1)
    # Transform results
    for k, v in data.items():
        data[k] = np.asarray(v)
    if data[k].dtype == np.float64:
        if np.isnan(data[k]).all():
            raise RuntimeError("nan")
    return True
