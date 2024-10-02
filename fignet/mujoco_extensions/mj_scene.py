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
from robosuite.utils.binding_utils import MjSim

from fignet.scene import SceneInfoDict, SceneInfoKey
from fignet.types import KinematicType
from fignet.utils.mesh import (
    get_vertices_from_history,
    get_vertices_num,
    get_vertices_offsets,
)

# render_context = MjRenderContext(sim)
# viewer = OpenCVRenderer(sim)
# viewer.render()
# sim.add_render_context(render_context)
# im = sim.render(
#     width=640,
#     height=480,
#     # camera_name=viewer.camera_name
# )
# im = np.flip(im, axis=0)


def get_scene_info(
    sim: MjSim,
    body_meshes: Dict,
    properties: Dict[str, Dict[str, Any]],
    obj_positions: np.ndarray,
    obj_quaternions: np.ndarray,
    obj_ids: Dict[str, int],
    collisions: Optional[Dict[str, Dict]] = None,
    contains_targets: bool = False,
) -> SceneInfoDict:
    # ! Adhoc solution:
    name_id_map = {}
    for name, idx in obj_ids.items():
        name_id_map["_".join([name, "main"])] = idx

    # Compute vertices
    vert_seq = get_vertices_from_history(
        obj_positions=obj_positions,
        obj_quaternions=obj_quaternions,
        obj_ids=name_id_map,
        body_meshes=body_meshes,
    )

    vert_offset_dict = get_vertices_offsets(body_meshes=body_meshes)

    vert_num_dict = get_vertices_num(body_meshes=body_meshes)
    vert_ref_pos = []
    for body_info in body_meshes.values():
        for mesh in body_info["meshes"]:
            vert_ref_pos.append(mesh.vertices)
    vert_ref_pos = np.vstack(vert_ref_pos)
    num_verts = vert_ref_pos.shape[0]

    assert vert_seq.shape[1] == num_verts
    # Compute centers of mass (COM)
    seq_len = obj_positions.shape[0]
    num_bodies = len(body_meshes)
    com_ref_pos = np.zeros((num_bodies, 3))
    com_seq = np.zeros((seq_len, num_bodies, 3))
    obj_offsets_dict = {
        body_name: idx for idx, body_name in enumerate(body_meshes.keys())
    }
    for body_name, body_info in body_meshes.items():
        body_index = obj_offsets_dict[body_name]
        body_id = sim.model.body_name2id(body_name)
        # Extract the translation part of the first transform as the reference
        # COM position for all meshes belonging to this body
        com_ref_pos[body_index] = sim.model.body_pos[body_id]

    for t in range(seq_len):
        for body_name, _ in body_meshes.items():
            body_index = obj_offsets_dict[body_name]
            if body_name in name_id_map:
                com_seq[t, body_index] = obj_positions[
                    t, name_id_map[body_name]
                ]
            else:
                com_seq[t, body_index] = com_ref_pos[body_index]

    # Calculate target
    if contains_targets:
        assert seq_len >= 3
        vert_target = (
            vert_seq[-1, ...] - 2 * vert_seq[-2, ...] + vert_seq[-3, ...]
        )
        com_target = com_seq[-1, ...] - 2 * com_seq[-2, ...] + com_seq[-3, ...]
        vert_seq = np.delete(vert_seq, -1, axis=0)
        com_seq = np.delete(com_seq, -1, axis=0)

    # Compute properties
    prop_dim = 0
    # Properties to concat
    prop_list = ["mass", "friction", "restitution"]
    for prop_name, val in next(iter(properties.values())).items():
        if prop_name not in prop_list:
            continue
        if isinstance(val, np.ndarray):
            prop_dim += val.size
        elif isinstance(val, float):
            prop_dim += 1
        else:
            continue
    vert_prop = np.empty((num_verts, prop_dim))
    obj_prop = np.empty((num_bodies, prop_dim))
    vert_kinematic = np.empty((num_verts, 1), dtype=np.int64)
    obj_kinematic = np.empty((num_bodies, 1), dtype=np.int64)
    for body_name, prop in properties.items():
        body_prop = []
        for prop_name, val in prop.items():
            if prop_name in prop_list:
                if isinstance(val, np.ndarray):
                    body_prop.extend(val.flatten())
                else:
                    body_prop.append(float(val))
        body_prop = np.array(body_prop)
        obj_prop[obj_offsets_dict[body_name]] = body_prop
        vert_start_idx = vert_offset_dict[body_name]
        vert_end_idx = vert_start_idx + vert_num_dict[body_name]
        vert_prop[vert_start_idx:vert_end_idx] = body_prop

        body_kinematic = (
            KinematicType.DYNAMIC
            if prop["is_dynamic"]
            else KinematicType.STATIC
        )
        vert_kinematic[vert_start_idx:vert_end_idx] = body_kinematic
        obj_kinematic[obj_offsets_dict[body_name]] = body_kinematic
    # Put everything together
    scn_info = {
        SceneInfoKey.VERT_SEQ: vert_seq,
        SceneInfoKey.COM_SEQ: com_seq,
        SceneInfoKey.VERT_REF_POS: vert_ref_pos,
        SceneInfoKey.COM_REF_POS: com_ref_pos,
        SceneInfoKey.VERT_OFFSETS_DICT: vert_offset_dict,
        SceneInfoKey.NUM_VERTS_DICT: vert_num_dict,
        SceneInfoKey.OBJ_OFFSETS_DICT: obj_offsets_dict,
        SceneInfoKey.VERT_PROP: vert_prop,
        SceneInfoKey.OBJ_PROP: obj_prop,
        SceneInfoKey.VERT_KINEMATIC: vert_kinematic,
        SceneInfoKey.OBJ_KINEMATIC: obj_kinematic,
    }
    if collisions is not None:
        scn_info[SceneInfoKey.CONTACT_PAIRS] = collisions
    if contains_targets:
        scn_info[SceneInfoKey.VERT_TARGET] = vert_target
        scn_info[SceneInfoKey.COM_TARGET] = com_target
    return scn_info
