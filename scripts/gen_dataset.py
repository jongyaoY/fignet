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
import tqdm
from dm_control import mjcf
from lxml import etree
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
from scipy.spatial.transform import Rotation as R

from fignet.scene import Scene

mujoco_model_path = "test_models/mujoco_scene.xml"

data_path = "datasets/three_bodies_1m"
meta_data_path = os.path.join(data_path, "metadata.json")
with open(meta_data_path) as f:
    metadata = json.load(f)
data_path = "datasets/three_bodies"
scene = Scene(metadata["scene_config"])
mjcf_model = mjcf.RootElement()
# Setup arena
mjcf_model.worldbody.add("light", name="top", pos=[3, 0, 1])
mjcf_model.worldbody.add(
    "camera",
    mode="fixed",
    name="frontview",
    pos=[3, 0, 1],
    quat=[0.56, 0.43, 0.43, 0.56],
)
mjcf_model.worldbody.add(
    "geom",
    name="floor",
    type="plane",
    pos=[0, 0, 0],
    size=[0, 0, 0.1],
    condim=3,
)
asset_prefix = os.getcwd()
dyn_bodies = []
# dyn_bodies=['cube', 'bread', 'bottle']
for name, obj in metadata["scene_config"]["objects"].items():
    dyn_bodies.append(name)
    mesh_name = name + "_mesh"
    mjcf_obj = mjcf_model.worldbody.add("body", name=name)
    mesh_file = os.path.join(asset_prefix, obj["mesh"])
    mjcf_model.asset.add("mesh", name=mesh_name, file=mesh_file)
    mjcf_obj.add("joint", type="free")
    mjcf_obj.add("geom", name=mesh_name, type="mesh", condim=3)

# Remove hashes from filenames
root = mjcf_model.to_xml()
meshes = [mesh for mesh in root.find("asset").iter() if mesh.tag == "mesh"]
for mesh in meshes:
    name, extension = mesh.get("file").split(".")
    filename = ".".join((name[:-41], extension))
    filename = os.path.join(asset_prefix, "test_models", filename)
    mesh.set("file", filename)
root.remove(root.find("default"))

xml_string = etree.tostring(root)
xml_string = xml_string.replace(b' class="/"', b"")
# sim = MjSim.from_xml_string(xml_string)
sim = MjSim.from_xml_file(mujoco_model_path)
sim.add_render_context(MjRenderContextOffscreen(sim, 0))
viewer = OpenCVRenderer(sim)

all_data = []
total_steps = 1000000
ep_length = 200
data_path = data_path + "_" + str(total_steps)
steps = 0
pbar = tqdm.tqdm(total=total_steps, desc="Sampling rollouts")

while steps < total_steps:
    data = {
        "obj_id": {},
        "timestep": [],
        "pos": [],
        "quat": [],
        "velp": [],
        "velr": [],
    }

    for i, name in enumerate(dyn_bodies):
        data["obj_id"][name] = i
    sim.reset()
    # Sample random object poses
    # TODO: sample until no overlap
    trial = 0
    num_trial = 10
    while trial < num_trial:
        for name in dyn_bodies:
            rand_pos = np.zeros(3)
            rand_pos[:2] = np.random.uniform(-0.15, 0.15, 2)
            rand_pos[2] = np.random.uniform(0.1, 0.15)
            rand_quat = R.random().as_quat()[[3, 0, 1, 2]]
            bid = sim.model.body_name2id(name)
            q_id = sim.model.body_jntadr[bid]
            sim.model.body_pos[bid] = rand_pos
            sim.data.qpos[q_id * 7 : q_id * 7 + 3] = rand_pos
            sim.data.qpos[q_id * 7 + 3 : q_id * 7 + 7] = rand_quat
        sim.forward()
        if sim.data.ncon == 0:
            break

    rand_vel = np.random.randn(len(dyn_bodies) * 6) * 0.8
    sim.data.qvel = rand_vel
    for _ in range(ep_length):
        if steps >= total_steps:
            break
        sim.forward()
        sim.step()
        data["timestep"].append(sim.data.time)
        pos = []
        quat = []
        velp = []
        velr = []
        for name in dyn_bodies:
            bid = sim.model.body_name2id(name)
            pos.append(sim.data.body_xpos[bid])
            quat.append(sim.data.body_xquat[bid][[1, 2, 3, 0]])
            velp.append(sim.data.get_body_xvelp(name))
            velr.append(sim.data.get_body_xvelr(name))
        data["pos"].append(np.array(pos))
        data["quat"].append(np.array(quat))
        data["velp"].append(np.array(velp))
        data["velr"].append(np.array(velr))
        # viewer.render()
        steps += 1
        pbar.update(1)
    for k, v in data.items():
        data[k] = np.asarray(v)
        if data[k].dtype == np.float64:
            if np.isnan(data[k]).all():
                raise RuntimeError("nan")
    all_data.append(data)
pbar.close()
print(
    f"collected {len(all_data)} episodes, \
        {sum([data['timestep'].shape[0] for data in all_data])} steps"
)
np.savez(data_path, all_data)
