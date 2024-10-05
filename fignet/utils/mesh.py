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

from collections import defaultdict
from typing import Dict

import numpy as np
import trimesh

from fignet.utils.geometric import (
    mesh_verts_sequence,
    pose_to_transform,
    transform_to_pose,
)


def get_vertices_offsets(
    body_meshes: Dict[str, Dict],
):
    vertex_offsets = defaultdict(int)
    id_offset = 0
    for body_name, body_info in body_meshes.items():
        vertex_offsets[body_name] = id_offset
        for mesh in body_info["meshes"]:
            id_offset += len(mesh.vertices)
    return vertex_offsets


def get_vertices_num(
    body_meshes: Dict[str, Dict],
):
    vert_num_dict = defaultdict(int)
    for body_name, body_info in body_meshes.items():
        num_verts = 0
        for mesh in body_info["meshes"]:
            num_verts += len(mesh.vertices)
        vert_num_dict[body_name] = num_verts
    return vert_num_dict


def get_vertices_from_history(
    obj_positions: np.ndarray,
    obj_quaternions: np.ndarray,
    body_meshes: Dict[str, Dict],
    obj_ids: Dict[str, int],
) -> np.ndarray:
    """
    Given history of object positions and quaternions, return the stacked
    vertices of all objects.

    Parameters:
    obj_positions (np.ndarray): Array of shape (history_length, num_obj, 3)
                                containing the positions history for each object.
    obj_quaternions (np.ndarray): Array of shape (history_length, num_obj, 4)
                            containing the quaternions history for each object.
    body_meshes (Dict[str, Dict]): Dictionary of body meshes.
    obj_ids (Dict[str, int]): Dictionary mapping object names to their IDs.

    Returns:
    np.ndarray: Stacked vertices of all objects, shape (history_length, num_verts, 3).
    """
    history_length = obj_positions.shape[0]
    all_transformed_vertices = []
    num_outliers = 0
    for body_name in body_meshes.keys():
        body_info = body_meshes[body_name]
        # Get body poses
        if body_name in obj_ids:
            obj_id = obj_ids[body_name]
            pos = obj_positions[:, obj_id]
            quat = obj_quaternions[:, obj_id]
            poses = np.concatenate([pos, quat], axis=-1)
        else:
            num_outliers += 1
            poses = transform_to_pose(np.eye(4))
            poses = np.tile(poses, (history_length, 1))
        # Get verts sequence
        for mesh, transform in zip(
            body_info["meshes"], body_info["transforms"]
        ):
            # Get mesh absolute pose sequence
            if not np.allclose(transform, np.eye(4), atol=1e-6):
                for t in range(poses.shape[0]):
                    poses[t] = transform_to_pose(
                        pose_to_transform(poses[t]) @ transform
                    )
            # Calculate transformed vert sequence
            vert_seq = mesh_verts_sequence(mesh, poses)
            all_transformed_vertices.append(vert_seq)

    if num_outliers >= obj_quaternions.shape[1]:
        raise RuntimeError("Too many unmatched names")
    # Stack all vertices over the history length
    stacked_vertices = np.concatenate(all_transformed_vertices, axis=1)
    return stacked_vertices


def get_transformed_copy(
    mesh: trimesh.Trimesh, transform: np.ndarray
) -> trimesh.Trimesh:
    """
    Get a transformed copy of the given mesh.

    Parameters:
    mesh : trimesh.Trimesh
        The original mesh to be transformed.
    transform : np.ndarray
        The 4x4 transformation matrix to be applied.

    Returns:
    trimesh.Trimesh
        The transformed mesh.
    """
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(transform)
    return mesh_copy
