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
from typing import Dict, List, Optional, Tuple

import hppfcl
import numpy as np
import trimesh

from fignet.mujoco_extensions.mj_classes import MjSimLearned
from fignet.mujoco_extensions.mj_utils import (
    get_body_transform,
    parse_meshes_initial,
)


class CollisionDetector:
    def __init__(
        self,
        sim: MjSimLearned,
        security_margin: float = 0.0,
        excluded_bodies: Optional[List[str]] = None,
    ):
        """
        Initialize the collision detector.

        Parameters:
        sim (MjSimLearned): The MuJoCo simulation instance.
        security_margin (float): Security margin for collision detection.
        excluded_bodies (List[str], optional): List of bodies to exclude from
        detection.
        """
        self.sim = sim
        self.security_margin = security_margin
        self.excluded_bodies = (
            excluded_bodies if excluded_bodies is not None else []
        )
        self.body_meshes = parse_meshes_initial(self.sim, self.excluded_bodies)
        self.col_obj_map = {}
        self.mesh_map = {}
        self._initialize_collision_objects()

    def _initialize_collision_objects(self):
        """Initialize hppfcl collision objects based on parsed meshes."""
        for body_name, body_info in self.body_meshes.items():
            for i, mesh in enumerate(body_info["meshes"]):
                fcl_mesh = hppfcl.BVHModelOBB()
                verts = hppfcl.StdVec_Vec3f()
                faces = hppfcl.StdVec_Triangle()
                verts.extend(mesh.vertices)
                faces.extend(
                    [
                        hppfcl.Triangle(
                            int(face[0]), int(face[1]), int(face[2])
                        )
                        for face in mesh.faces
                    ]
                )
                transform = np.eye(4)
                transform = hppfcl.Transform3f(
                    transform[:3, :3], transform[:3, 3]
                )
                fcl_mesh.beginModel(len(mesh.faces), len(mesh.vertices))
                fcl_mesh.addSubModel(verts, faces)
                fcl_mesh.endModel()
                fcl_collision_object = hppfcl.CollisionObject(fcl_mesh)
                self.col_obj_map[fcl_mesh.id()] = fcl_collision_object
                self.mesh_map[fcl_mesh.id()] = (
                    body_name,
                    i,
                )  # the i-th mesh of the body of body_name

    def _update_collision_object_transforms(self):
        """Update the transforms of hppfcl collision objects based on
        simulation state."""
        for fcl_mesh_id, (body_name, mesh_idx) in self.mesh_map.items():
            # body_id = self.sim.model.body_name2id(body_name)
            body_transform = get_body_transform(self.sim, body_name)
            geom_transform = self.body_meshes[body_name]["transforms"][
                mesh_idx
            ]
            combined_transform = body_transform @ geom_transform
            combined_transform = hppfcl.Transform3f(
                combined_transform[:3, :3], combined_transform[:3, 3]
            )
            self.col_obj_map[fcl_mesh_id].setTransform(combined_transform)

    def _get_body_mesh(self, body_name, mesh_idx) -> trimesh.Trimesh:
        if body_name in self.body_meshes:
            if mesh_idx < len(self.body_meshes[body_name]["meshes"]):
                return self.body_meshes[body_name]["meshes"][mesh_idx]
            else:
                raise ValueError(f"Mesh_idx: {mesh_idx} exceeds len")
        else:
            raise ValueError(f"{body_name} not exists")

    def detect_collisions(
        self, bidirectional: bool
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Detect collisions between bodies' meshes with a security margin.
        Args:
            bidirectional (bool, optional): If also returns the reverse contact
            pairs. Defaults to False.
        Returns:
        List[Tuple[str, int, str, int]]: List of tuples indicating collision pairs.
        """
        self._update_collision_object_transforms()

        # Initialize the collision manager and add collision objects
        manager = hppfcl.DynamicAABBTreeCollisionManager()
        for col_obj in self.col_obj_map.values():
            manager.registerObject(col_obj)
        manager.update()

        # Perform collision detection
        callback = hppfcl.CollisionCallBackDefault()
        callback.data.request.security_margin = self.security_margin
        callback.data.request.num_max_contacts = 100000
        callback.data.request.enable_contacts = True

        manager.collide(callback)

        contacts = list(callback.data.result.getContacts())
        collisions = defaultdict(lambda: defaultdict(list))
        for contact in contacts:
            idx1 = contact.o1.id()
            idx2 = contact.o2.id()
            body_name_1, mesh_idx_1 = self.mesh_map[idx1]
            body_name_2, mesh_idx_2 = self.mesh_map[idx2]
            mesh_1 = self._get_body_mesh(body_name_1, mesh_idx_1)
            mesh_2 = self._get_body_mesh(body_name_2, mesh_idx_2)
            point_1 = contact.getNearestPoint1()
            point_2 = contact.getNearestPoint2()
            face_idx_1 = contact.b1
            face_idx_2 = contact.b2
            vert_idx_1 = mesh_1.faces[face_idx_1]
            vert_idx_2 = mesh_2.faces[face_idx_2]
            contact_info = collisions[(body_name_1, body_name_2)]

            contact_info["contact_points"].append((point_1, point_2)),
            contact_info["contact_normals"].append(
                (
                    mesh_1.face_normals[face_idx_1],
                    mesh_2.face_normals[face_idx_2],
                )
            )
            contact_info["vert_ids_local"].append((vert_idx_1, vert_idx_2))
            contact_info["face_ids_local"].append((face_idx_1, face_idx_2))

            if bidirectional:
                # Reverse the order for the bidirectional entry
                reverse_contact_info = collisions[(body_name_2, body_name_1)]
                reverse_contact_info["contact_points"].append(
                    (point_2, point_1)
                )
                reverse_contact_info["contact_normals"].append(
                    (
                        mesh_2.face_normals[face_idx_2],
                        mesh_1.face_normals[face_idx_1],
                    )
                )
                reverse_contact_info["vert_ids_local"].append(
                    (vert_idx_2, vert_idx_1)
                )
                reverse_contact_info["face_ids_local"].append(
                    (face_idx_2, face_idx_1)
                )

        return collisions

    def visualize(self):
        """
        Visualize the current state of the collision meshes with trimesh,
        highlighting the faces in collision.
        """
        collisions = self.detect_collisions(bidirectional=False)

        # Create a scene
        scene = trimesh.Scene()

        # Track which meshes have collisions
        collision_faces = defaultdict(list)
        contact_points = []
        for (body_name_1, body_name_2), contact_info in collisions.items():
            for face_idx_1, face_idx_2 in contact_info["face_ids_local"]:
                collision_faces[body_name_1].append(face_idx_1)
                collision_faces[body_name_2].append(face_idx_2)
            for point_1, point_2 in contact_info["contact_points"]:
                contact_points.append(point_1)
                contact_points.append(point_2)

        # Visualize meshes and colliding faces
        for fcl_mesh_id, (body_name, mesh_idx) in self.mesh_map.items():
            transform = self.col_obj_map[fcl_mesh_id].getTransform()
            mat = np.eye(4)
            mat[:3, :3] = transform.getRotation()
            mat[:3, 3] = transform.getTranslation()
            mesh = self._get_body_mesh(body_name, mesh_idx).copy()
            mesh.apply_transform(mat)
            collided_faces = collision_faces[body_name]
            for face_idx in collided_faces:
                mesh.visual.face_colors[face_idx] = np.array(
                    [255, 0, 0, 100], dtype=np.uint8
                )  # Red color for collision faces
            scene.add_geometry(mesh)

        # Visualize points
        if contact_points:
            points = np.array(contact_points)
            point_cloud = trimesh.points.PointCloud(
                points, colors=[0, 255, 0, 255]
            )  # Green for contact points
            scene.add_geometry(point_cloud)
        # Show the scene
        scene.show(smooth=False)
