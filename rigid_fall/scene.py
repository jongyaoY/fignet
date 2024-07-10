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

import copy
from typing import List

import numpy as np
from robosuite.models.arenas import Arena, EmptyArena
from robosuite.models.world import MujocoWorldBase
from robosuite.utils.mjcf_utils import array_to_string, find_elements
from robosuite.utils.transform_utils import convert_quat
from scipy.spatial.transform import Rotation as R

from rigid_fall.mesh_object import MeshObject, get_object_properties


def random_floor_properties():
    friction = np.random.uniform(low=0.8, high=1.0, size=(1,)).item()
    restitution = np.random.uniform(low=0.99, high=1.01, size=(1,)).item()
    floor_properties = {
        "solref": [0.02, restitution],
        "solimp": [0.998, 0.998, 0.001],
        "friction": [friction, 0.3 * friction, 0.1 * friction],
        "priority": [1],
    }
    return floor_properties


def set_floor_properties(arena, properties):
    floor = find_elements(
        root=arena.worldbody, tags="geom", attribs={"name": "floor"}
    )
    for name, p in properties.items():
        floor.set(name, array_to_string(p))


class Scene:
    def __init__(self, random_floor: bool = True):
        self.objects: List[MeshObject] = None
        self.arena: Arena = EmptyArena()
        # Random floor
        if random_floor:
            self.floor_properties = random_floor_properties()
            set_floor_properties(self.arena, self.floor_properties)
        # Set camera
        self.camera_name = "frontview"
        matrix = [
            [0.0, -0.369, 0.929],
            [1.0, 0.001, -0.001],
            [0.0, 0.929, 0.369],
        ]
        # matrix = np.array(matrix).T
        cam_quat = R.from_matrix(matrix).as_quat()
        self.arena.set_camera(
            self.camera_name,
            pos=[1.1, 0.0, 0.47],
            quat=convert_quat(cam_quat, "wxyz"),
        )

    def add_objects(self, objects: List[MeshObject]):
        self.objects = copy.deepcopy(objects)

    def get_properties(self):
        properties = get_object_properties(self.objects)
        properties.update({"floor": self.floor_properties})
        return properties

    def to_xml(self):
        mujoco_model = MujocoWorldBase()
        mujoco_model.merge(self.arena)
        for o in self.objects:
            mujoco_model.merge_assets(o)
            mujoco_model.worldbody.append(o.get_obj())
        return mujoco_model.get_xml()
