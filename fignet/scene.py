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

from typing import Any, Dict, Optional

import numpy as np
import trimesh
import trimesh.creation

from fignet.collision import CollisionManager
from fignet.types import (
    Edge,
    EdgeType,
    Graph,
    KinematicType,
    NodeFeature,
    NodeType,
)
from fignet.utils import (
    match_meshes,
    mesh_com,
    mesh_com_sequence,
    mesh_verts,
    mesh_verts_sequence,
    pose_to_transform,
    transform_to_pose,
)


class Scene:
    """Describe/manage mesh objects and calculate node features and
    connectivity
    """

    def __init__(self, config: dict):
        self._mesh_dim = 3
        self._dim_properties = None
        self._num_vertices = 0
        self._num_obj = 0
        self._meshes = {}
        self._init_poses = {}
        self._obj_properties = {}
        self._obj_kin = {}
        self._vert_offsets = {}
        self._obj_ids = {}
        self._connectivity_radius = config.get("connectivity_radius", 0.005)
        self._noise_std = config.get("noise_std", 0.0)
        self._collision_manager = CollisionManager(
            security_margin=self._connectivity_radius
        )
        self._verts_ref_pos = None
        # Initialize scene

        # Create env objects
        for name, env_obj in config.get("env").items():
            if name == "floor":
                mesh = trimesh.creation.box(extents=env_obj.get("extents"))
            else:
                raise RuntimeError(f"Unknown environment object type {name}")
            if env_obj.get("initial_pose"):
                mesh.apply_transform(
                    pose_to_transform(np.asarray(env_obj["initial_pose"]))
                )
                self._init_poses[name] = env_obj["initial_pose"]
            self.add_object(
                name=name, mesh=mesh, obj_kinematic=KinematicType.STATIC
            )
            self._set_obj_properties(name, env_obj["properties"])
        # Load meshes
        for name, obj in config.get("objects").items():
            mesh = trimesh.load(obj["mesh"])
            trimesh.repair.fix_normals(mesh)
            self.add_object(
                name=name, mesh=mesh, obj_kinematic=KinematicType.DYNAMIC
            )
            self._set_obj_properties(name, obj["properties"])
            if self._dim_properties is None:
                self._dim_properties = self._object_property_array(name).shape[
                    1
                ]

        self._setup()

    def add_object(
        self,
        name: str,
        obj_kinematic: KinematicType,
        mesh: trimesh.Trimesh,
        id: int = None,
    ):
        """Append mesh object to collision manager and update ids respectively

        Args:
            name (str): Object name, has to be unique
            obj_kinematic (KinematicType): Object kinematic type, static or dynamic
            mesh (trimesh.Trimesh): Object mesh
            id (int, optional): Specify object id

        Raises:
            RuntimeError: If object name already exist
        """
        if name in self._meshes:
            raise RuntimeError(f"{name} already exists!")
            # self._num_vertices -= self._meshes[name].vertices.shape[0]
            # self._num_obj -= 1
        self._meshes.update({name: mesh.copy()})
        self._collision_manager.add_object(name, mesh)
        if id is not None:
            self._obj_ids[name] = id
        else:
            self._obj_ids[name] = self._num_obj
        self._obj_kin[name] = obj_kinematic
        self._num_vertices += self._meshes[name].vertices.shape[0]
        self._num_obj += 1

    def _setup(self):
        """Set up mesh node id offsets and their positions"""
        offset = 0
        for name in self._obj_ids.keys():
            self._vert_offsets[name] = offset
            offset += self._meshes[name].vertices.shape[0]
        self._verts_ref_pos = np.empty((self._num_vertices, self._mesh_dim))
        self._obj_ref_com = np.empty((self._num_obj, self._mesh_dim))
        for name, ob_id in self._obj_ids.items():
            start, end = self._vert_index(name)
            self._verts_ref_pos[start:end, ...] = mesh_verts(
                self._meshes[name], None
            )
            self._obj_ref_com[ob_id, ...] = self._meshes[name].center_mass

    def _set_obj_properties(self, name: str, properties: Dict[str, Any]):
        """Set object properties"""

        try:
            if properties["mass"] == "undefined":
                properties["mass"] = 1.0 / self._meshes[name].mass
        except KeyError:
            if "density" in properties:
                self._meshes[name].density = properties["density"]
                # properties["mass"] = properties["density"] * self._meshes[name].volume
                properties["mass"] = 1.0 / self._meshes[name].mass
                del properties["density"]
            else:
                properties["mass"] = 0.0

        self._obj_properties.update({name: properties})

    def set_obj_pose(self, name: str, pose: np.ndarray):
        """Set object absolute pose in the collision manager"""
        self._collision_manager.set_transform(
            name, pose_to_transform(pose), relative=False
        )

    def update_obj_pose(self, name: str, pose: np.ndarray):
        """Update object pose in the collision manager"""
        if pose.size == 16:
            transform = pose
        elif pose.size == 7:
            transform = pose_to_transform(pose)
        self._collision_manager.set_transform(name, transform, relative=True)

    def synchronize_states(
        self, obj_poses: np.ndarray, obj_ids: Dict[str, int]
    ):
        """Synchronize object poses and calculate mesh vertices positions"""
        seq_len = obj_poses.shape[0]
        verts_seq = np.empty((seq_len, self._num_vertices, self._mesh_dim))
        obj_com_seq = np.empty((seq_len, self._num_obj, self._mesh_dim))
        for ob_name, ob_id in self._obj_ids.items():
            start, end = self._vert_index(ob_name)
            if self.is_dynamic_object(ob_name):
                ob_pose_id = obj_ids[ob_name]
                obj_pose = obj_poses[-1, ob_pose_id, ...]
                self.set_obj_pose(ob_name, obj_pose)
                verts_seq[:, start:end, ...] = mesh_verts_sequence(
                    self._meshes[ob_name], obj_poses[:, ob_pose_id, ...]
                )
                obj_com_seq[:, ob_id, ...] = mesh_com_sequence(
                    self._meshes[ob_name], obj_poses[:, ob_pose_id, ...]
                )

            else:
                obj_pose = None
                verts_seq[:, start:end, ...] = np.tile(
                    mesh_verts(self._meshes[ob_name], None), (seq_len, 1, 1)
                )
                obj_com_seq[:, ob_id, ...] = np.tile(
                    self._init_poses[ob_name][:3], (seq_len, 1)
                )
        self._verts_seq = verts_seq
        self._obj_com_seq = obj_com_seq

    def refresh_sequence(self):
        """update verts pos buffer after poses updates"""
        latest_verts_pos = np.empty((self._num_vertices, self._mesh_dim))
        latest_com_pos = np.empty((self._num_obj, self._mesh_dim))
        for name, ob_id in self._obj_ids.items():
            start, end = self._vert_index(name)
            if self.is_dynamic_object(name):
                obj_transform = self._collision_manager.get_transform(name)
            else:
                obj_transform = None
            latest_verts_pos[start:end, :] = mesh_verts(
                self._meshes[name], obj_transform
            )
            latest_com_pos[ob_id, :] = mesh_com(
                self._meshes[name], obj_transform
            )
        self._verts_seq = np.vstack(
            [self._verts_seq, latest_verts_pos[None, :]]
        )
        self._obj_com_seq = np.vstack(
            [self._obj_com_seq, latest_com_pos[None, :]]
        )
        self._verts_seq = np.delete(self._verts_seq, 0, axis=0)
        self._obj_com_seq = np.delete(self._obj_com_seq, 0, axis=0)

    def to_graph(
        self,
        target_poses: Optional[np.ndarray] = None,
        obj_ids: Optional[Dict[str, int]] = None,
        noise: bool = False,
    ):
        """Encode scene state into graph

        Args:
            target_poses (Optional[np.ndarray], optional): (seq_len, n_obj, 7).
            Defaults to None.
            obj_ids (Optional[Dict[str, int]], optional):
            Object indices. Defaults to None.
            noise (bool): Whether to add noise
        Returns:
            Graph: A graph with node/edge features and edge index
        """
        # Get the synchronized verts and com
        noised_verts_seq = self._verts_seq.copy()
        noised_obj_com_seq = self._obj_com_seq.copy()
        if noise and self._noise_std:
            node_type = self._node_types()
            m_mask = (node_type[0] == KinematicType.DYNAMIC).squeeze()
            o_mask = (node_type[1] == KinematicType.DYNAMIC).squeeze()
            noised_verts_seq[:, m_mask, ...] = self._verts_seq[
                :, m_mask, ...
            ] + np.random.normal(
                0, self._noise_std, self._verts_seq[:, m_mask, ...].shape
            )
            noised_obj_com_seq[:, o_mask, ...] = self._obj_com_seq[
                :, o_mask, ...
            ] + np.random.normal(
                0, self._noise_std, self._obj_com_seq[:, o_mask, ...].shape
            )
            # Calculate node/edge features and connectivity

        index, ff_features = self._cal_connectivity()
        edge_features = self._edge_features(
            obj_com=noised_obj_com_seq[-1, ...],
            obj_ref_com=self._obj_ref_com,
            vert_pos=noised_verts_seq[-1, ...],
            vert_ref_pos=self._verts_ref_pos,
            index=index,
        )
        edge_features[EdgeType.FACE_FACE] = ff_features
        # Calculate target without noise
        cal_target = False
        if target_poses is not None:
            cal_target = True
            vert_target_pos = np.empty((self._num_vertices, self._mesh_dim))
            obj_target_pos = np.empty((self._num_obj, self._mesh_dim))
            for ob_name, ob_id in self._obj_ids.items():
                start, end = self._vert_index(ob_name)
                if self.is_dynamic_object(ob_name):
                    ob_pose_id = obj_ids[ob_name]
                    obj_target_pose = target_poses[ob_pose_id, ...]
                    vert_target_pos[start:end, ...] = mesh_verts(
                        self._meshes[ob_name], obj_target_pose
                    )
                    obj_target_pos[ob_id, ...] = mesh_com(
                        self._meshes[ob_name], obj_target_pose
                    )
                else:
                    vert_target_pos[start:end, ...] = self._verts_seq[
                        -1, start:end, :
                    ]
                    obj_target_pos[ob_id, ...] = self._obj_com_seq[
                        -1, ob_id, :
                    ]
            vert_target_acc = (
                vert_target_pos
                - 2 * self._verts_seq[-1, ...]
                + self._verts_seq[-2, ...]
            )
            obj_target_acc = (
                obj_target_pos
                - 2 * self._obj_com_seq[-1, ...]
                + self._obj_com_seq[-2, ...]
            )

        node_pos = (noised_verts_seq, noised_obj_com_seq)
        node_kin = self._node_types()
        node_prop = self._node_properties()
        if cal_target:
            node_target = (vert_target_acc, obj_target_acc)
        m_features = NodeFeature(
            position=node_pos[0],
            kinematic=node_kin[0],
            properties=node_prop[0],
            target=node_target[0] if cal_target else None,
        )
        o_features = NodeFeature(
            position=node_pos[1],
            kinematic=node_kin[1],
            properties=node_prop[1],
            target=node_target[1] if cal_target else None,
        )
        graph = Graph()
        graph.node_sets = {
            NodeType.MESH: m_features,
            NodeType.OBJECT: o_features,
        }
        for edge_type in EdgeType:
            graph.edge_sets.update(
                {
                    edge_type: Edge(
                        attribute=edge_features[edge_type],
                        index=index[edge_type],
                    )
                }
            )

        return graph

    def update(
        self,
        m_acc: np.ndarray,
        o_acc: np.ndarray,
        obj_ids: Dict[str, int],
        device,
    ):
        """Update scene state with node accelerations. First, mesh vertex
        positions are updated using the predicted accelerations: v_{t} = a_{t}
        + 2v_{t-1} - v_{t-2}. Next object poses are calculated using mesh
        matching.

        Args:
            m_acc (np.ndarray): Mesh node accelerations (n_mnode, 3)
            o_acc (np.ndarray): Object node accelerations (n_onode, 3)
            obj_ids (Dict[str, int]): Specifies the order of object poses
            device (str): cuda or cpu

        Returns:
            np.ndarray: Object relative poses (pos,quat), ordered according to
            obj_ids (n_obj, 7)
        """
        m_kin, o_kin = self._node_types()
        m_kin = m_kin.squeeze()
        o_kin = o_kin.squeeze()
        m_mask = m_kin == KinematicType.DYNAMIC
        o_mask = o_kin == KinematicType.DYNAMIC
        pred_verts = np.empty_like(self._verts_seq[-1, ...])
        pred_com = np.empty_like(self._obj_com_seq[-1, ...])
        pred_verts[m_kin == KinematicType.STATIC, :] = self._verts_seq[
            -1, m_kin == KinematicType.STATIC, :
        ]
        pred_com[o_kin == KinematicType.STATIC, :] = self._obj_com_seq[
            -1, o_kin == KinematicType.STATIC, :
        ]
        pred_verts[m_mask, :] = (
            m_acc[m_mask, :]
            + 2 * self._verts_seq[-1, m_mask, :]
            - self._verts_seq[-2, m_mask, :]
        )
        pred_com[o_mask, :] = (
            o_acc[o_mask, :]
            + 2 * self._obj_com_seq[-1, o_mask, :]
            - self._obj_com_seq[-2, o_mask, :]
        )
        obj_rel_poses = np.empty((len(obj_ids), 7))
        for name, _ in self._obj_ids.items():
            # Update only dynamic objects
            if self.is_dynamic_object(name):
                trg_mesh = self._meshes[name].copy()
                src_mesh = self._collision_manager.get_object(name)
                start, end = self._vert_index(name)
                trg_mesh.vertices = pred_verts[start:end, ...]
                try:
                    rel_transform = match_meshes(trg_mesh, src_mesh, device)
                except Exception:
                    rel_transform = np.eye(4)
                self.update_obj_pose(name, rel_transform)
                obj_rel_poses[obj_ids[name]] = transform_to_pose(rel_transform)

        self.refresh_sequence()
        return obj_rel_poses

    def _object_property_array(self, name: str):
        """Get object properties as an array

        Args:
            name (str): Object name

        Returns:
            np.ndarray: Object property array (1, prop_dim)
        """
        prop = self._obj_properties.get(name)
        prop_array = []
        if prop:
            for v in prop.values():
                if isinstance(v, list):
                    prop_array += v
                elif isinstance(v, np.ndarray):
                    prop_array += v.tolist()
                else:
                    prop_array += [v]
            return np.asarray(prop_array)[None, :]  # (1, prop_dim)
        else:
            return None

    def _node_types(self):
        """Return node kinematic types

        Returns:
            np.ndarray: Mesh node types
            np.ndarray: Object node types
        """
        m_node_types = np.empty((self._num_vertices, 1), dtype=np.int64)
        o_node_types = np.empty((self._num_obj, 1), dtype=np.int64)
        for name, obj_id in self._obj_ids.items():
            start, end = self._vert_index(name)
            if self.is_dynamic_object(name):
                m_node_types[start:end, :] = KinematicType.DYNAMIC
                o_node_types[obj_id, :] = KinematicType.DYNAMIC
            else:
                m_node_types[start:end, :] = KinematicType.STATIC
                o_node_types[obj_id, :] = KinematicType.STATIC
        return m_node_types, o_node_types

    def _node_properties(self):
        mesh_prop = np.empty((self._num_vertices, self._dim_properties))
        obj_prop = np.empty((self._num_obj, self._dim_properties))
        for name, obj_id in self._obj_ids.items():
            start, end = self._vert_index(name)
            mesh_prop[start:end, :] = np.repeat(
                self._object_property_array(name), end - start, axis=0
            )
            obj_prop[obj_id, :] = self._object_property_array(name)
        return mesh_prop, obj_prop

    def _node_features(self, verts_seq: np.ndarray, com_seq: np.ndarray):
        """Calculate and return node features

        Args:
            verts_seq (np.ndarray): Sequence of mesh node positions (n_mnode,
            3)
            com_seq (np.ndarray): Sequence of object node positions (n_onode,
            3)

        Returns:
            np.ndarray: Mesh node features
            np.ndarray: Object node features
        """
        # Mesh node components
        mesh_vel = np.empty((self._num_vertices, self._mesh_dim))
        mesh_prop = np.empty((self._num_vertices, self._dim_properties))
        # Object node components
        obj_vel = np.empty((self._num_obj, self._mesh_dim))
        obj_prop = np.empty((self._num_obj, self._dim_properties))
        for name, obj_id in self._obj_ids.items():
            start, end = self._vert_index(name)
            mesh_prop[start:end, :] = np.repeat(
                self._object_property_array(name), end - start, axis=0
            )
            obj_prop[obj_id, :] = self._object_property_array(name)
        # Node velocities
        mesh_vel = verts_seq[-1, ...] - verts_seq[-2, ...]
        obj_vel = com_seq[-1, ...] - obj_vel[-2, ...]
        mesh_features = np.concatenate([mesh_vel, mesh_prop], axis=-1)
        obj_features = np.concatenate([obj_vel, obj_prop], axis=-1)
        return mesh_features, obj_features

    def _cal_connectivity(self):
        """Calculate graph connectivity, including mesh to mesh (mm) edges,
        mesh to object (mo) edges, and face to face (ff) edges and face edge
        features

        Returns:
            Dict[str, np.ndarray]: Edge indices
            np.ndarray: Face edge features
        """
        # regular edges
        mm_index = np.empty((2, 0), dtype=np.int64)
        mo_index = np.empty((2, 0), dtype=np.int64)
        for name, obj_id in self._obj_ids.items():
            mm_edges = self._meshes[name].edges_unique.T  # (2, n_edges)
            # bidirectional edges
            mm_edges = np.concatenate([mm_edges, mm_edges[[1, 0]]], axis=-1)
            m_offset, _ = self._vert_index(name)
            o_offset = obj_id
            mm_edges = mm_edges + m_offset

            mo_senders = np.arange(
                0, len(self._meshes[name].vertices), dtype=np.int64
            )
            mo_receivers = np.repeat(0, len(self._meshes[name].vertices))
            mo_senders = mo_senders + m_offset
            mo_receivers = mo_receivers + o_offset
            mo_edges = np.vstack([mo_senders, mo_receivers])

            mm_index = np.concatenate([mm_index, mm_edges], axis=-1)
            mo_index = np.concatenate([mo_index, mo_edges], axis=-1)

        # face-face edges
        contacts = self._collision_manager.in_collision()
        # self._collision_manager.visualize_contacts(contacts) #! for debug
        pairs = self._collision_manager.get_collision_pairs(
            contacts, bidirectional=True
        )
        senders = []
        receivers = []
        ff_features = []

        for (name1, name2, face_id1, face_id2), (
            point1,
            point2,
        ) in pairs.items():
            mesh1 = self._collision_manager.get_object(name1)
            mesh2 = self._collision_manager.get_object(name2)
            verts1 = mesh1.vertices[
                mesh1.faces[face_id1]
            ]  # or mesh1.triangles[face_id1]?
            verts2 = mesh2.vertices[mesh2.faces[face_id2]]
            norm_s = mesh1.face_normals[face_id1]
            norm_r = mesh2.face_normals[face_id2]
            d_rs = point2 - point1  # p_r - p_s from receiver to sender
            id_s = mesh1.faces[face_id1]
            id_r = mesh2.faces[face_id2]
            d_si = verts1 - point1  # sender
            d_ri = verts2 - point2  # receiver
            d_si_norms = np.linalg.norm(d_si, axis=1)
            d_ri_norms = np.linalg.norm(d_ri, axis=1)

            # Order according to the distance to retain permutation equivariance
            ord_s = np.argsort(d_si_norms)
            ord_r = np.argsort(d_ri_norms)
            id_s = id_s[ord_s]
            id_r = id_r[ord_r]
            d_si = d_si[ord_s]
            d_ri = d_ri[ord_r]
            d_si_norms = d_si_norms[ord_s]
            d_ri_norms = d_ri_norms[ord_r]

            # Concatenate the norms too
            d_rs = np.concatenate(
                [d_rs, np.linalg.norm(d_rs, keepdims=True)], axis=-1
            )
            d_si = np.concatenate([d_si, d_si_norms[:, None]], axis=-1)
            d_ri = np.concatenate([d_ri, d_ri_norms[:, None]], axis=-1)
            # assert np.allclose(mesh1.vertices[id_s]-point1, d_si)
            # assert np.allclose(mesh2.vertices[id_r]-point2, d_ri)

            features = np.concatenate(
                [
                    d_rs,  # (4,)
                    d_si.flatten(),  # (3*4,)
                    d_ri.flatten(),  # (3*4,)
                    norm_s.flatten(),  # (3,)
                    norm_r.flatten(),  # (3)
                ],
                axis=-1,
            )
            id_s += self._vert_index(name1)[0]
            id_r += self._vert_index(name2)[0]

            senders.append(id_s)
            receivers.append(id_r)
            ff_features.append(features)

        ff_index = np.empty((2, len(ff_features), 3), dtype=np.int64)
        if senders:
            ff_index[0] = np.asarray(senders, dtype=np.int64)
        if receivers:
            ff_index[1] = np.asarray(receivers, dtype=np.int64)

        index = {
            EdgeType.MESH_MESH: mm_index,
            EdgeType.MESH_OBJ: mo_index,
            EdgeType.OBJ_MESH: mo_index[[1, 0]],
            EdgeType.FACE_FACE: ff_index,
        }
        return index, np.asarray(ff_features)

    def _edge_features(
        self,
        vert_pos: np.ndarray,
        vert_ref_pos: np.ndarray,
        obj_com: np.ndarray,
        obj_ref_com: np.ndarray,
        index: Dict[str, np.ndarray],
    ):
        """Calculate edge features given edge indices, and node positions

        Args:
            vert_pos (np.ndarray): Mesh vertex positions
            vert_ref_pos (np.ndarray): Reference mesh vertex positions
            obj_com (np.ndarray): Mesh center of mass positions
            obj_ref_com (np.ndarray): Reference center of mass positions
            index (Dict[str, np.ndarray]): Edge indices

        Returns:
            Dict[str, np.ndarray]: Edge features
        """
        edge_features = {}
        for edge_type, edge_index in index.items():
            s_index = edge_index[0]
            r_index = edge_index[1]
            if edge_type == EdgeType.MESH_MESH:
                d_rs = vert_pos[s_index, ...] - vert_pos[r_index, ...]
                d_rs_ref = (
                    vert_ref_pos[s_index, ...] - vert_ref_pos[r_index, ...]
                )
            elif edge_type == EdgeType.MESH_OBJ:
                d_rs = vert_pos[s_index] - obj_com[r_index]
                d_rs_ref = vert_ref_pos[s_index] - obj_ref_com[r_index]
            elif edge_type == EdgeType.OBJ_MESH:
                d_rs = obj_com[s_index] - vert_pos[r_index]
                d_rs_ref = obj_ref_com[s_index] - vert_ref_pos[r_index]
            else:
                continue
            # assert np.allclose(
            #     np.linalg.norm(d_rs, axis=1), np.linalg.norm(d_rs_ref, axis=1)
            # )
            d_rs = np.concatenate(
                [d_rs, np.linalg.norm(d_rs, axis=1, keepdims=True)], axis=-1
            )
            d_rs_ref = np.concatenate(
                [d_rs_ref, np.linalg.norm(d_rs_ref, axis=1, keepdims=True)],
                axis=-1,
            )
            edge_features[edge_type] = np.hstack([d_rs, d_rs_ref])

        return edge_features

    def _vert_index(self, name: str):
        """Get the start and end id of the specified object. Helper function to
        access mesh nodes of a specific object

        Args:
            name (str): Object name

        Raises:
            RuntimeError: If name not exists

        Returns:
            tuple(int, int): Start and end id
        """
        if name not in self._meshes:
            raise RuntimeError(f"{name} not exists!")
        start_id = self._vert_offsets[name]
        end_id = start_id + self._meshes[name].vertices.shape[0]
        return start_id, end_id

    def is_dynamic_object(self, name: str):
        """Return if an object is dynamic

        Args:
            name (str): Object name

        Raises:
            RuntimeError: If name not exists

        Returns:
            bool: Return true if object is dynamic
        """
        if name in self._obj_kin:
            return self._obj_kin[name] == KinematicType.DYNAMIC
        else:
            raise RuntimeError(f"{name} not exists!")
