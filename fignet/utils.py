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

import enum
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import trimesh
from pytorch3d.ops import corresponding_points_alignment
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContext, MjSim
from scipy.spatial.transform import Rotation as R


class KinematicType(enum.IntEnum):
    STATIC = 0
    DYNAMIC = 1
    SIZE = 2


class NodeType(enum.IntEnum):
    MESH = 0
    OBJECT = 1
    SIZE = 9


def check_nan(data):
    if isinstance(data, dict):
        for k, v in data.items():
            check_nan(v)
    elif isinstance(data, torch.Tensor):
        if data.nelement() and torch.isnan(data).all().item():
            raise RuntimeError("nan")


def to_numpy(tensor: torch.Tensor):
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


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


def dict_to_tensor(d: dict, device=None):
    new_dict = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = dict_to_tensor(v, device)
        else:
            if device is None:
                new_dict[k] = torch.FloatTensor(v)
            else:
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.long:
                        new_dict[k] = v.to(device)
                    else:
                        new_dict[k] = v.float().to(device)
                elif isinstance(v, np.ndarray):
                    if v.dtype == np.int64:
                        new_dict[k] = torch.from_numpy(v).long().to(device)
                    else:
                        new_dict[k] = torch.from_numpy(v).float().to(device)
                elif isinstance(v, int):
                    new_dict[k] = torch.LongTensor([v]).to(device)
                else:
                    raise TypeError(f"Unexpected data type: {type(v)}")

    return new_dict


def to_tensor(array: Union[np.ndarray, torch.Tensor], device: str = None):
    if isinstance(array, torch.Tensor):
        if array.dtype == torch.float64:
            tensor = array.float()
        else:
            tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.int64:
            tensor = torch.from_numpy(array).long()
        else:
            tensor = torch.from_numpy(array).float()
    elif isinstance(array, dict):
        return dict_to_tensor(array, device)
    else:
        raise TypeError(f"Cannot conver {type(array)} to tensor")

    if device:
        return tensor.to(device)
    else:
        return tensor


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def rollout(
    sim,
    init_obj_poses,
    obj_ids,
    scene,
    device,
    nsteps,
):
    if isinstance(init_obj_poses, torch.Tensor):
        init_obj_poses = to_numpy(init_obj_poses)
    scene.synchronize_states(init_obj_poses, obj_ids)
    obj_poses = init_obj_poses[-1, ...]
    trajectory = np.vstack([obj_poses[None, :]])
    for _ in tqdm.tqdm(
        range(nsteps - init_obj_poses.shape[0]), desc="sampling rollout"
    ):
        graph = scene.to_graph()
        graph = to_tensor(graph, device)
        m_pred_acc, o_pred_acc = sim.predict_accelerations(graph)
        m_pred_acc = sim.denormalize_accelerations(m_pred_acc)
        o_pred_acc = sim.denormalize_accelerations(o_pred_acc)
        obj_rel_poses = scene.update(
            m_acc=to_numpy(m_pred_acc),
            o_acc=to_numpy(o_pred_acc),
            obj_ids=obj_ids,
            device=device,
        )
        for i in range(obj_poses.shape[0]):
            prev_transform = pose_to_transform(obj_poses[i, :])
            rel_transform = pose_to_transform(obj_rel_poses[i, :])
            transform = rel_transform @ prev_transform
            obj_poses[i, :] = transform_to_pose(transform)
        trajectory = np.vstack([trajectory, obj_poses[None, :]])

    return np.asarray(trajectory)


def plot_graph(
    graph,
):
    mesh_nodes = graph["pos"]["mesh"][-1, ...]
    obj_nodes = graph["pos"]["object"][-1, ...]
    mm_edges = graph["index"]["mm"]
    mo_edges = graph["index"]["mo"]
    ff_edges = graph["index"]["ff"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.scatter(*mesh_nodes.T, alpha=0.2, s=100, color="blue")
    ax.scatter(*obj_nodes.T, alpha=0.2, s=100, color="red")
    for mm_index in mm_edges.T:
        edge = np.array([mesh_nodes[mm_index[0]], mesh_nodes[mm_index[1]]])
        ax.plot(*edge.T, color="gray")
    for mo_index in mo_edges.T:
        edge = np.array([mesh_nodes[mo_index[0]], obj_nodes[mo_index[1]]])
        ax.plot(*edge.T, color="green")
    for edge_id in range(ff_edges.shape[1]):
        ff_index = ff_edges[:, edge_id, ...].T
        ax.plot_trisurf(
            *mesh_nodes[ff_index[:, 0]].T, linewidth=0.2, antialiased=True
        )
        ax.plot_trisurf(
            *mesh_nodes[ff_index[:, 1]].T, linewidth=0.2, antialiased=True
        )
        for i in range(3):
            edge = np.array(
                [mesh_nodes[ff_index[i, 0]], mesh_nodes[ff_index[i, 1]]]
            )
            ax.plot(*edge.T, color="red")

    ax.grid(False)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def visualize_trajectory(
    mujoco_xml: str, traj: np.ndarray, obj_ids: dict, off_screen: bool = False
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
                height=viewer.height,
                width=viewer.width,
            )
            # write frame to window
            im = np.flip(im, axis=0)
            screens.append(im)
    if off_screen:
        return np.array(screens)
