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

from typing import Union

import numpy as np
import torch
import trimesh
from pytorch3d.ops import corresponding_points_alignment
from scipy.spatial.transform import Rotation as R

from fignet.utils.conversion import to_numpy, to_tensor


def rot_diff(quat1, quat2):
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    diff = r1.inv() * r2
    diff = diff.as_rotvec()
    if diff.ndim == 2:
        return np.linalg.norm(diff, axis=1)
    else:
        return np.linalg.norm(diff)


def transform_to_pose(transform):
    quat = R.from_matrix(transform[:3, :3]).as_quat()
    pos = transform[:3, 3]
    return np.concatenate([pos, quat])


def pose_to_transform(pose: Union[np.ndarray, torch.Tensor]):
    """
    pose: [pos_xyz, quat_xyzw]
    """
    seq_mode = False
    if isinstance(pose, torch.Tensor):
        pose = to_numpy(pose)
    if pose.ndim == 1:
        pose = pose[None, :]
    if pose.ndim == 3:
        seq_mode = True
        batch_size = pose.shape[0]
        seq_len = pose.shape[1]
        pose = pose.reshape((batch_size * seq_len, pose.shape[2]))
    transform = np.repeat(np.eye(4), pose.shape[0], axis=0).reshape(
        (pose.shape[0], 4, 4)
    )
    transform[:, :3, 3] = pose[:, :3]
    if pose.shape[-1] == 7:
        r = R.from_quat(pose[:, 3:])
        transform[:, :3, :3] = r.as_matrix()
    if seq_mode:
        transform = transform.reshape((batch_size, seq_len, 4, 4))
    # Squeeze if batch_size == 1
    if transform.shape[0] == 1:
        return transform.squeeze(axis=0)
    else:
        return transform


def match_meshes(
    trg_mesh: trimesh.Trimesh,
    src_mesh: trimesh.Trimesh,
    device,
):
    src_verts = to_tensor(src_mesh.vertices, device)[None, :]
    trg_verts = to_tensor(trg_mesh.vertices, device)[None, :]
    ret = corresponding_points_alignment(src_verts, trg_verts)
    R = (
        to_numpy(ret.R).squeeze().transpose()
    )  # X was multiplied from the right side: X[i] R[i] + T[i] = Y[i]
    T = to_numpy(ret.T).squeeze()
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T
    return transform


def mesh_node_velocities(
    mesh: trimesh.Trimesh, vel: Union[np.ndarray, torch.Tensor]
):
    if vel.ndim == 1:
        vel = vel[None, :]
    velp = vel[:, :3]
    velr = vel[:, 3:]
    if isinstance(vel, torch.Tensor):
        # assume mesh is not transformed
        r = to_tensor(mesh.vertices - mesh.center_mass, vel.device)
        v = torch.linalg.cross(velr, r) + velp
    elif isinstance(vel, np.ndarray):
        r = mesh.vertices
        v = np.cross(velr, r) + velp
    return v


def mesh_verts(mesh: trimesh.Trimesh, pose: np.ndarray = None):
    if pose is None:
        return mesh.vertices.copy()
    else:
        if pose.size == 7:
            matrix = pose_to_transform(pose)
        elif pose.size == 16:
            matrix = pose
        else:
            raise TypeError("invalid pose")
        verts = trimesh.transform_points(mesh.vertices, matrix)
        return verts


def mesh_verts_sequence(mesh: trimesh.Trimesh, poses: np.ndarray):
    """
    Args:
        mesh:
        poses: (seq_len, 7)
    Return:
        verts_seq (seq_len, n_verts, 3)
    """
    assert poses.ndim == 2
    verts_seq = []
    for t in range(poses.shape[0]):
        verts_seq.append(mesh_verts(mesh, poses[t, :]))
    return np.asarray(verts_seq)


def mesh_com(mesh: trimesh.Trimesh, pose: np.ndarray = None):
    if pose is None:
        return mesh.center_mass.copy()
    else:
        if pose.size == 7:
            matrix = pose_to_transform(pose)
        elif pose.size == 16:
            matrix = pose
        else:
            raise TypeError("invalid pose")
        com = trimesh.transform_points(mesh.center_mass[None, :], matrix)[0]
        return com


def mesh_com_sequence(mesh: trimesh.Trimesh, poses: np.ndarray):
    assert poses.ndim == 2
    com_seq = []
    for t in range(poses.shape[0]):
        com_seq.append(mesh_com(mesh, poses[t, :]))
    return np.asarray(com_seq)
