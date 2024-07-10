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

import collections
from typing import List, Optional

import hppfcl as fcl
import numpy as np
import trimesh


class CollisionManager:
    """Similar to CollisionManager from trimesh"""

    def __init__(self, security_margin: float = 0.005):
        """Initializer

        Args:
            security_margin (float, optional): Collision radius. Defaults to
            0.005.
        """
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._security_margin = security_margin
        self._objs = {}
        self._applied_transforms = {}
        # unpopulated values will return None {id(bvh) : str, name}
        self._names = collections.defaultdict(lambda: None)
        self._manager.setup()

    def get_objects(self):
        """Get mesh object

        Returns:
            Dict[str, trimesh.Trimesh]: Dictionary of the meshes in the manager
        """
        ret = {}
        for name, obj in self._objs.items():
            ret.update(
                {
                    name: obj["mesh"],
                }
            )

        return ret

    def get_object(self, name: str):
        """Get single mesh

        Args:
            name (str): mesh name

        Returns:
            trimesh.Trimesh: Mesh
        """

        if name in self._objs:
            return self._objs[name]["mesh"].copy()
        else:
            return None

    def add_object(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        transform: Optional[np.ndarray] = None,
    ):
        """Append mesh to the manager

        Args:
            name (str): Object name mesh (trimesh.Trimesh): Object mesh
            transform (Optional[np.ndarray], optional): Initial transform.
            Defaults to None.

        Raises:
            ValueError: If transform matrix is not 4x4
        """
        verts = fcl.StdVec_Vec3f()
        faces = fcl.StdVec_Triangle()
        verts.extend(mesh.vertices)
        faces.extend(
            [
                fcl.Triangle(int(face[0]), int(face[1]), int(face[2]))
                for face in mesh.faces
            ]
        )
        if transform is None:
            transform = np.eye(4)
        if transform.shape != (4, 4):
            raise ValueError("transform must be (4,4)!")

        # geom = fcl.BVHModelOBBRSS() #hppfcl.BVHModelOBB
        geom = fcl.BVHModelOBB()
        geom.beginModel(len(mesh.faces), len(mesh.vertices))
        geom.addSubModel(verts, faces)
        geom.endModel()

        t = fcl.Transform3f(transform[:3, :3], transform[:3, 3])
        o = fcl.CollisionObject(geom, t)
        # o.getAABB().expand(self._security_margin/2.)
        # #https://github.com/humanoid-path-planner/hpp-fcl/issues/346

        # Add collision object to set
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name])
        self._objs[name] = {"obj": o, "geom": geom, "mesh": mesh}
        # store the name of the geometry
        self._names[geom.id()] = name

        self._manager.registerObject(o)
        self._manager.update()
        # self._obj2mesh.update( { o: mesh, } )

    def set_transform(
        self,
        name: str,
        transform: np.ndarray,
        relative: bool = False,
    ):
        """Set absolute/relative transform of an object in the manager

        Args:
            name (str): Object name
            transform (np.ndarray): 4x4 transform matrix
            relative (bool, optional): Is the transform relative to the current
            pose. Defaults to False.

        Raises:
            ValueError: If object name not exists
        """
        if name in self._objs:
            if relative:
                prev_t = self._applied_transforms[name]
                t = transform @ prev_t
            else:
                t = transform
            o = self._objs[name]["obj"]
            o.setRotation(t[:3, :3])
            o.setTranslation(t[:3, 3])
            self._manager.update(o)

            mesh = self._objs[name]["mesh"]
            if name in self._applied_transforms:
                mesh.apply_transform(
                    np.linalg.inv(self._applied_transforms[name])
                )
            mesh.apply_transform(t)
            self._applied_transforms[name] = t
        else:
            raise ValueError("{} not in collision manager!".format(name))

    def get_transform(
        self,
        name: str,
    ):
        """Get current absolute transform of an object

        Args:
            name (str): Object name

        Raises:
            ValueError: If object name not exists

        Returns:
            np.ndarray: 4x4 transform matrix
        """
        if name in self._objs:
            return self._applied_transforms[name]
        else:
            raise ValueError("{} not in collision manager!".format(name))

    def in_collision(self):
        """Run collision detection

        Returns:
            List[hppfcl.Contact]: A list of contacts
        """
        callback = fcl.CollisionCallBackDefault()
        callback.data.request.security_margin = self._security_margin
        callback.data.request.num_max_contacts = 100000
        callback.data.request.enable_contacts = True
        self._manager.update()
        self._manager.collide(callback)
        contacts = list(callback.data.result.getContacts())
        return contacts

    def get_collision_pairs(
        self, contacts: List[fcl.Contact], bidirectional: bool = False
    ):
        """Parse a list of fcl.Contact into dictionary of tuples and
        contact points

        Args:
            contacts (List[fcl.Contact]): List of contacts
            bidirectional (bool, optional): If also returns the reverse contact
            pairs. Defaults to False.

        Raises:
            RuntimeWarning: If there are repeating contacts

        Returns:
            Dict[tuple, tuple]: (obj_name1, obj_name2, face_id1, face_id2) ->
            (contact_point1, contact_point2)
        """
        contact_pairs = {}
        for contact in contacts:
            name1 = self._extract_name(contact.o1)
            name2 = self._extract_name(contact.o2)
            point1 = contact.getNearestPoint1()
            point2 = contact.getNearestPoint2()
            assert name1 is not None
            assert name2 is not None
            # t = -contact.penetration_depth/2. point1 = contact.pos + t *
            # contact.normal point2 = contact.pos - t * contact.normal
            if (name1, name2, contact.b1, contact.b2) in contact_pairs:
                raise RuntimeWarning("conflicting contact pairs!")
            contact_pairs[(name1, name2, contact.b1, contact.b2)] = (
                point1,
                point2,
            )
            if bidirectional:
                contact_pairs[(name2, name1, contact.b2, contact.b1)] = (
                    point2,
                    point1,
                )

        return contact_pairs

    def visualize_contacts(self, contacts: List[fcl.Contact]):
        """Helper function to visualize contacts

        Args:
            contacts (List[fcl.Contact]): List of contacts to visualize
        """
        collision_face_color = np.array([255, 0, 0, 100], dtype=np.uint8)
        faces_in_collision = {}
        for name in self._objs.keys():
            faces_in_collision[name] = []
        pts = []
        ray_origins = []
        ray_directions = []
        for contact in contacts:
            names = (
                self._extract_name(contact.o1),
                self._extract_name(contact.o2),
            )

            if contact.b1 not in faces_in_collision[names[0]]:
                faces_in_collision[names[0]].append(contact.b1)
            if contact.b2 not in faces_in_collision[names[1]]:
                faces_in_collision[names[1]].append(contact.b2)
            # t = -contact.penetration_depth / 2.0
            # if t > self._security_margin: print(f'{t} is larger then security
            #     margin') point1 = contact.pos + t * contact.normal point2 =
            # contact.pos - t * contact.normal
            point1 = contact.getNearestPoint1()
            point2 = contact.getNearestPoint2()
            pts.append(point1)
            pts.append(point2)
            # pts.append(contact.pos) pts.append(contact.pos)
            ray_origins.append(contact.pos)
            ray_directions.append(contact.normal)
        # stack rays into line segments for visualization as Path3D
        meshes = []
        for name, faces in faces_in_collision.items():
            mesh = self._objs[name]["mesh"].copy()
            mesh.visual.face_colors[faces] = collision_face_color
            meshes.append(mesh)
        scene = trimesh.Scene()
        scene.add_geometry(meshes)

        if len(pts) > 0:
            pcl = trimesh.points.PointCloud(vertices=pts)
            scene.add_geometry(pcl)

        if len(ray_origins) > 0:
            ray_origins = np.asarray(ray_origins)
            ray_directions = np.asarray(ray_directions)
            # ray_visualize = trimesh.load_path( np.hstack((ray_origins,
            #     ray_origins + ray_directions)).reshape(-1, 2, 3) )
            # scene.add_geometry(ray_visualize)

        scene.show(smooth=False)

    def _extract_name(self, geom):
        """
        Retrieve the name of an object from the manager by its CollisionObject,
        or return None if not found.

        Parameters
        -----------
        geom : CollisionObject or BVHModel
          Input model

        Returns
        ------------
        names : hashable
          Name of input geometry
        """
        return self._names.get(
            geom.id()
        )  # !id(geom) changes every time, should be
        # a bug of the hppfcl:
        # https://github.com/humanoid-path-planner/hpp-fcl/issues/590
        # for name, obj in self._objs.items():
        #     if obj["geom"] == geom:
        #         return name

        # return None
