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

from typing import Dict, List, Optional, Tuple, Union

import mujoco
import numpy as np
import trimesh
from robosuite.utils.binding_utils import MjModel, MjSim
from scipy.spatial.transform import Rotation as R

from fignet.mujoco_extensions.constants import JointType


def set_mjdata(
    sim: MjSim,
    positions: Union[np.ndarray, List[List[float]]],
    quaternions: Union[np.ndarray, List[List[float]]],
    obj_ids: Dict[str, int],
):
    """
    Set the positions and orientations of objects in mujoco.MjData.

    Parameters:
    sim (mujoco.MjSim): The MuJoCo simulation instance containing the MjData to update.
    positions (np.ndarray or List[List[float]]): Array or list of object
    positions, shape (n_obj, 3).
    quaternions (np.ndarray or List[List[float]]): Array or list of object
    orientations represented as quaternions, shape (n_obj, 4).
    obj_ids (Dict[str, int]): Dictionary mapping object names to MuJoCo body IDs.
    """
    if isinstance(positions, list):
        positions = np.array(positions)
    if isinstance(quaternions, list):
        quaternions = np.array(quaternions)

    assert positions.shape[1] == 3, "Positions should have shape (n_obj, 3)"
    assert (
        quaternions.shape[1] == 4
    ), "Quaternions should have shape (n_obj, 4)"

    for obj_name, obj_id in obj_ids.items():
        if obj_name not in sim.model.body_names:
            # ! TODO: adhoc solution
            for name in sim.model.body_names:
                if name.startswith(obj_name):
                    obj_name = name
                    break
        bid = sim.model.body_name2id(obj_name)
        q_id = sim.model.body_jntadr[bid]
        sim.data.qpos[q_id * 7 : q_id * 7 + 3] = positions[obj_id]
        sim.data.qpos[q_id * 7 + 3 : q_id * 7 + 7] = quaternions[obj_id][
            [3, 0, 1, 2]
        ]  # xyzw -> wxyz

    sim.forward()


def get_mjdata(
    sim: MjSim,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Get the positions and orientations of all objects in mujoco.MjData.

    Parameters:
    sim (mujoco.MjSim): The MuJoCo simulation instance containing the MjData to read.

    Returns:
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        - Dictionary of object positions, where keys are object names and
          values are positions (shape (3,))
        - Dictionary of object orientations represented as quaternions, where
          keys are object names and values are quaternions (shape (4,))
    """
    obj_ids = {sim.model.body_id2name(i): i for i in range(sim.model.nbody)}

    positions = {}
    quaternions = {}

    for obj_name, obj_id in obj_ids.items():
        positions[obj_name] = sim.data.body_xpos[obj_id]
        quaternions[obj_name] = sim.data.body_xquat[obj_id][[1, 2, 3, 0]]

    return positions, quaternions


def get_body_transform(sim: MjSim, body_id: Union[int, str]) -> np.ndarray:

    if isinstance(body_id, str):
        body_id = sim.model.body_name2id(body_id)

    body_pos = sim.data.body_xpos[body_id]
    body_quat = sim.data.body_xquat[body_id][[1, 2, 3, 0]]
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R.from_quat(body_quat).as_matrix()
    # ! this gives wrong results:
    # transform_matrix = trimesh.transformations.quaternion_matrix(body_quat)
    transform_matrix[:3, 3] = body_pos
    return transform_matrix


def create_mesh_from_geom(
    model: MjModel, geom_id: int
) -> Optional[trimesh.Trimesh]:
    """
    Create a trimesh for the given geometry, focusing on collision-related geometries.

    Parameters:
    model : MjModel
        The MuJoCo simulation instance.
    geom_id : int
        The ID of the geometry in MuJoCo.

    Returns:
    Optional[trimesh.Trimesh]
        The created trimesh if it is a collision-related geometry, None otherwise.
    """
    # Ignore geoms that are not used for collision

    if (
        model.geom_contype[geom_id] == 0
        or model.geom_conaffinity[geom_id] == 0
    ):
        return None  # Skip geometries not involved in collision detection.
    geom_type = model.geom_type[geom_id]

    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        size = np.array([3.0, 3.0, 0.1])
        mesh = trimesh.creation.box(extents=size)
        # Plane should be centered, so we shift the mesh down by half its thickness
        mesh.apply_translation([0, 0, -size[2] / 2.0])
    elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        vertices = model.mesh_vert[
            model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id]
            + model.mesh_vertnum[mesh_id]
        ].reshape(-1, 3)
        faces = model.mesh_face[
            model.mesh_faceadr[mesh_id] : model.mesh_faceadr[mesh_id]
            + model.mesh_facenum[mesh_id]
        ].reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices, faces, process=False)
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        size = model.geom_size[geom_id]
        mesh = trimesh.creation.box(extents=size * 2)
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = model.geom_size[geom_id][0]
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = model.geom_size[geom_id][0]
        height = (
            model.geom_size[geom_id][1] * 2
        )  # MuJoCo capsule height is hemispherical
        return trimesh.creation.capsule(radius=radius, height=height)
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        # Create ellipsoid by scaling an icosphere
        mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=4)
        scale = model.geom_size[geom_id] * 2  # MuJoCo uses half-extents
        mesh.apply_scale(scale)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, height = (
            model.geom_size[geom_id][0],
            model.geom_size[geom_id][1] * 2,
        )
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
    else:
        raise ValueError(
            f"Unknown geom_type {geom_type} for geom_id {geom_id}"
        )
    return mesh


def get_geom_transform(model: MjModel, geom_id: int) -> np.ndarray:
    """
    Get the local transformation matrix for a geom using MjModel.

    Parameters:
    model : MjModel
        The MuJoCo model instance.
    geom_id : int
        The ID of the geometry in MuJoCo.

    Returns:
    np.ndarray
        The 4x4 transformation matrix for the geometry.
    """

    geom_pos = model.geom_pos[geom_id]
    # Convert MuJoCo quaternion [w, x, y, z] to trimesh quaternion [x, y, z, w]
    geom_quat = model.geom_quat[geom_id][[1, 2, 3, 0]]
    geom_transform = np.eye(4)
    geom_transform[:3, :3] = R.from_quat(geom_quat).as_matrix()
    geom_transform[:3, 3] = geom_pos

    return geom_transform


def parse_meshes_initial(
    model: MjModel, excluded_bodies: Optional[List[str]] = None
) -> Dict[str, Dict]:
    body_meshes = {}
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = model.body_id2name(body_id)
        if excluded_bodies and body_name in excluded_bodies:
            continue
        if body_name not in body_meshes:
            body_meshes[body_name] = {
                "meshes": [],
                "transforms": [],
                "geom_ids": [],
                "vert_offsets": [],
            }

        try:
            mesh = create_mesh_from_geom(model, geom_id)
        except ValueError as e:
            print(e)
            continue
        if mesh is None:
            continue  # Skip geometries not involved in collision detection.
        geom_transform = get_geom_transform(model, geom_id)
        body_meshes[body_name]["meshes"].append(mesh)
        body_meshes[body_name]["transforms"].append(geom_transform)
        body_meshes[body_name]["geom_ids"].append(geom_id)

    out_body_meshes = {}
    for body_name, body_info in body_meshes.items():
        acc_vert_num = 0
        if len(body_info["meshes"]) > 0:
            for mesh in body_info["meshes"]:
                body_info["vert_offsets"].append(acc_vert_num)
                acc_vert_num += len(mesh.vertices)
            out_body_meshes[body_name] = body_info

    return out_body_meshes


def parse_physical_properties(model: MjModel) -> Dict[str, Dict]:
    """
    Parse and store static physical properties like friction and mass for each body.

    Parameters:
    model : MjModel
        The MuJoCo simulation instance.

    Returns:
    Dict[str, Dict]: Dictionary mapping body names to their physical properties.
    """
    properties = {}
    for body_id in range(model.nbody):
        body_name = model.body_id2name(body_id)
        # Get the starting index and number of geometries for the body
        start_geom_id = model.body_geomadr[body_id]
        geom_count = model.body_geomnum[body_id]

        # Slice the geom_friction array to get the friction values for all
        # geoms of the body, now assume one body has only one geom!
        friction = []
        restitution = []
        for geom_index in range(start_geom_id, start_geom_id + geom_count):
            if (
                model.geom_contype[geom_index] != 0
                and model.geom_conaffinity[geom_index] != 0
            ):
                friction.append(model.geom_friction[geom_index])
                restitution.append(model.geom_solref[geom_index][1])
        # TODO: Take only the properties from the first geom
        friction = friction[0]
        restitution = restitution[0]

        mass = model.body_mass[body_id]
        inertia = model.body_inertia[body_id]
        is_dynamic = model.body_dofnum[body_id] > 0
        properties[body_name] = {
            "friction": friction,
            "restitution": restitution,
            "mass": mass,
            "inertia": inertia,
            "is_dynamic": is_dynamic,
        }
    return properties


def parse_kinematic_chain(model: MjModel):
    kinematic_chain = []
    for joint_id in range(model.njnt):
        # Get joint information
        joint_name = model.joint_id2name(joint_id)
        joint_type = JointType(model.jnt_type[joint_id])
        if joint_type == JointType.FREE:
            joint_axis = np.zeros(3, dtype=np.float64)
        else:
            joint_axis = model.jnt_axis[joint_id]
        joint_qposadr = model.jnt_qposadr[joint_id]
        joint_dofadr = model.jnt_dofadr[joint_id]
        body_id = model.jnt_bodyid[joint_id]

        # Get body information
        body_name = model.body_id2name(body_id)
        parent_id = model.body_parentid[body_id]
        parent_name = model.body_id2name(parent_id)

        # Append kinematic chain information
        kinematic_chain.append(
            {
                "joint_name": joint_name,
                "joint_axis": joint_axis,
                "joint_type": joint_type,
                "qpos_address": joint_qposadr,
                "dof_address": joint_dofadr,
                "body_name": body_name,
                "parent_name": parent_name,
            }
        )

    return kinematic_chain
