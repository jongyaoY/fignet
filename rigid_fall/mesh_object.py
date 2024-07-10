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

import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import trimesh
from robosuite.models.objects import PrimitiveObject
from robosuite.utils.mjcf_utils import (
    ALL_TEXTURES,
    CustomMaterial,
    array_to_string,
    find_elements,
    new_body,
    new_element,
    new_joint,
    new_site,
    string_to_array,
)

assets_root = os.path.join(os.path.dirname(__file__), "assets")


def asset_path_completion(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(assets_root, path)


def get_objects_from_files(
    mesh_file_list: List[str],
    random_properties: bool = True,
    restitution_range=[0.95, 1.1],
    friction_range=[0.0, 1.0],
    density_range=[500.0, 3000.0],
):
    object_list = []

    for mesh_path in mesh_file_list:
        if random_properties:
            friction = np.random.uniform(
                low=friction_range[0], high=friction_range[1], size=(1,)
            ).item()
            restitution = np.random.uniform(
                low=restitution_range[0], high=restitution_range[1], size=(1,)
            ).item()
            density = np.random.uniform(
                low=density_range[0], high=density_range[0], size=(1,)
            ).item()
        else:
            friction = 0.95
            restitution = 1.0
            density = 1000.0

        tex = np.random.choice(list(ALL_TEXTURES), size=(1,)).item()
        material = CustomMaterial(
            texture=tex,
            tex_name="tex",
            mat_name="mat",
        )
        obj_prop = {
            "solref": [0.002, restitution],
            "solimp": [0.998, 0.998, 0.001],
            "friction": [friction, 0.3 * friction, 0.1 * friction],
            "density": density,
            "material": material,
        }
        object_list.append(
            MeshObject(
                mesh_path=mesh_path,
                **obj_prop,
            )
        )
    return object_list


def get_n_objects(
    num_objects: int,
    random_properties: bool = True,
    restitution_range=[0.95, 1.1],
    friction_range=[0.0, 1.0],
    density_range=[500.0, 3000.0],
):
    all_file_list = []
    mesh_path = os.path.join(assets_root, "meshes")
    for f in os.listdir(mesh_path):
        if os.path.isfile(os.path.join(mesh_path, f)):
            all_file_list.append(os.path.join(mesh_path, f))
    mesh_file_list = np.random.choice(
        all_file_list, num_objects, replace=True
    ).tolist()
    return get_objects_from_files(
        mesh_file_list=mesh_file_list,
        random_properties=random_properties,
        restitution_range=restitution_range,
        friction_range=friction_range,
        density_range=density_range,
    )


def get_all_objects(
    random_properties: bool = True,
    restitution_range=[0.95, 1.1],
    friction_range=[0.0, 1.0],
    density_range=[500.0, 3000.0],
):
    mesh_file_list = []
    mesh_path = os.path.join(assets_root, "meshes")
    for f in os.listdir(mesh_path):
        if os.path.isfile(os.path.join(mesh_path, f)):
            mesh_file_list.append(os.path.join(mesh_path, f))

    return get_objects_from_files(
        mesh_file_list=mesh_file_list,
        random_properties=random_properties,
        restitution_range=restitution_range,
        friction_range=friction_range,
        density_range=density_range,
    )


class MeshObject(PrimitiveObject):
    def __init__(
        self,
        mesh_path,
        name=None,
        unique_name=True,
        rgba=None,
        density=None,
        friction=None,
        solref=None,
        solimp=None,
        material=None,
        joints="default",
        obj_type="all",
        duplicate_collision_geoms=True,
    ):
        if name is None:
            self._name = Path(mesh_path).stem
        else:
            self._name = name
        if unique_name:
            self._name = "-".join([self._name, str(id(self))])
        self._mesh_name = self._name + "_mesh"
        self._mesh_path = asset_path_completion(mesh_path)
        mesh_obj = trimesh.load(self._mesh_path)
        size = mesh_obj.bounding_box.extents
        mesh = ET.Element(
            "mesh", attrib={"file": self._mesh_path, "name": self._mesh_name}
        )

        super().__init__(
            name=self._name,
            size=size,
            rgba=rgba,
            density=density,
            friction=friction,
            solref=solref,
            solimp=solimp,
            material=material,
            joints=joints,
            obj_type=obj_type,
            duplicate_collision_geoms=duplicate_collision_geoms,
        )

        if not hasattr(self, "asset"):
            self.asset = ET.Element("asset")
        self.asset.append(mesh)

    def _get_object_subtree(self):
        obj = new_body(name="main")
        # Get base element attributes
        element_attr = {"name": "g0", "type": "mesh", "mesh": self._mesh_name}
        # Add collision geom if necessary
        if self.obj_type in {"collision", "all"}:
            col_element_attr = deepcopy(element_attr)
            col_element_attr.update(self.get_collision_attrib_template())
            col_element_attr["density"] = str(self.density)
            col_element_attr["friction"] = array_to_string(self.friction)
            col_element_attr["solref"] = array_to_string(self.solref)
            col_element_attr["solimp"] = array_to_string(self.solimp)
            obj.append(new_element(tag="geom", **col_element_attr))

        # Add visual geom if necessary
        if self.obj_type in {"visual", "all"}:
            vis_element_attr = deepcopy(element_attr)
            vis_element_attr.update(self.get_visual_attrib_template())
            vis_element_attr["name"] += "_vis"
            if self.material == "default":
                vis_element_attr["rgba"] = "0.5 0.5 0.5 1"  # mujoco default
                vis_element_attr["material"] = "mat"
            elif self.material is not None:
                vis_element_attr["material"] = self.material.mat_attrib["name"]
            else:
                vis_element_attr["rgba"] = array_to_string(self.rgba)
            obj.append(new_element(tag="geom", **vis_element_attr))
        # add joint(s)
        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))
        # add a site as well
        site_element_attr = self.get_site_attrib_template()
        site_element_attr["name"] = "default_site"
        obj.append(new_site(**site_element_attr))
        return obj


def get_object_properties(objects: List[MeshObject]):
    properties = {}
    for obj in objects:
        geom = find_elements(
            root=obj.get_obj(), tags="geom", attribs={"group": "0"}
        )
        mesh_name = geom.get("mesh")
        mesh = find_elements(
            root=obj.asset, tags="mesh", attribs={"name": mesh_name}
        )
        mesh_file = mesh.get("file")
        solref = string_to_array(geom.get("solref"))
        solimp = string_to_array(geom.get("solimp"))
        friction = string_to_array(geom.get("friction"))
        density = string_to_array(geom.get("density"))
        properties[obj.name] = {
            "solref": solref,
            "solimp": solimp,
            "friction": friction,
            "density": density,
            "mesh": mesh_file,
        }
    return properties
