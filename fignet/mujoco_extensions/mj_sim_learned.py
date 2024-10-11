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

from collections import deque
from typing import Dict

import mujoco
import numpy as np
from robosuite.utils.binding_utils import MjSim
from torch_geometric.transforms import ToDevice

from fignet.data.types import KinematicType
from fignet.modules.simulator import LearnedSimulator
from fignet.mujoco_extensions.mj_utils import set_mjdata
from fignet.mujoco_extensions.physics_state_tracker import PhysicsStateTracker
from fignet.mujoco_extensions.preprocess import get_scene_info
from fignet.utils.conversion import to_numpy
from fignet.utils.geometric import pose_to_transform, transform_to_pose
from fignet.utils.scene import SceneInfoKey, build_graph


class MjSimLearned(MjSim):

    def __init__(self, model) -> None:
        super(MjSimLearned, self).__init__(model)
        self.positions_history = None
        self.quaternions_history = None

    def set_state(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        obj_ids: Dict[str, int],
    ):
        num_obj = len(obj_ids)
        seq_len = 0
        latest_pos = np.empty((num_obj, 3))
        latest_quat = np.empty((num_obj, 4))
        if positions.ndim == 3:
            latest_pos = positions[-1, ...]
            latest_quat = quaternions[-1, ...]
            seq_len = positions.shape[0]
        elif positions.ndim == 2:
            latest_pos = positions
            latest_quat = quaternions

        else:
            raise ValueError(f"Positions has {positions.ndim} dim")

        if seq_len > 0:
            for t in range(seq_len):
                self.positions_history.append(positions[t, ...])
                self.quaternions_history.append(quaternions[t, ...])
        elif self.positions_history is not None:
            self.positions_history.append(positions)
            self.quaternions_history.append(quaternions)

        if getattr(self, "name_id_map", None) is None:
            self.name_id_map = obj_ids
        elif self.name_id_map != obj_ids:
            raise ValueError("The given name to id map is not consistent")

        set_mjdata(
            self,
            positions=latest_pos,
            quaternions=latest_quat,
            obj_ids=obj_ids,
        )
        # self.forward()

    def set_gnn_backend(self, gnn_model: LearnedSimulator):
        self.gnn_model = gnn_model
        self.physical_tracker = PhysicsStateTracker(
            self, gnn_model.cfg.collision_radius
        )
        self.name_id_map = None  # Map body name to idx in positions
        self.positions_history = deque(
            maxlen=gnn_model.cfg.input_sequence_length
        )
        self.quaternions_history = deque(
            maxlen=gnn_model.cfg.input_sequence_length
        )

    def get_body_states(self):
        if self.name_id_map is None:
            raise RuntimeError("Name to state id map is not initialized")
        return (
            self.positions_history[-1].copy(),
            self.quaternions_history[-1].copy(),
        )

    def step(self, backend: str):
        """Step simulation."""

        if backend == "mujoco":
            mujoco.mj_step(self.model._model, self.data._data)
        elif backend == "gnn":
            if (
                len(self.positions_history)
                < self.gnn_model.cfg.input_sequence_length
            ):
                raise RuntimeError("Not enough data in history buffer")

            collisions = self.physical_tracker.detect_collisions(
                bidirectional=True
            )
            # self.physical_tracker.visualize()
            scn_info = get_scene_info(
                model=self.model,
                body_meshes=self.physical_tracker.body_meshes,
                properties=self.physical_tracker.properties,
                obj_positions=np.array(self.positions_history),
                obj_quaternions=np.array(self.quaternions_history),
                pose_addr=self.name_id_map,
                collisions=collisions,
                contains_targets=False,
            )

            graph = build_graph(
                scn_info, self.gnn_model.cfg.build_cfg, self.gnn_model.training
            )
            pred_acc = self.gnn_model(ToDevice(self.gnn_model.device)(graph))
            pred_acc = to_numpy(pred_acc)

            vert_seq = scn_info[SceneInfoKey.VERT_SEQ]
            v_kin = scn_info[SceneInfoKey.VERT_KINEMATIC].squeeze()
            v_mask = v_kin == KinematicType.DYNAMIC

            pred_verts = np.empty_like(vert_seq[-1, ...])
            pred_verts[v_kin == KinematicType.STATIC, :] = vert_seq[
                -1, v_kin == KinematicType.STATIC, :
            ]

            pred_verts[v_mask, :] = (
                pred_acc[v_mask, :]
                + 2 * vert_seq[-1, v_mask, :]
                - vert_seq[-2, v_mask, :]
            )

            rel_transforms = self.physical_tracker.get_transform_updates(
                # source_verts=vert_seq[-1, ...],
                target_verts=pred_verts,
                vert_offsets_dict=scn_info[SceneInfoKey.VERT_OFFSETS_DICT],
                num_verts_dict=scn_info[SceneInfoKey.NUM_VERTS_DICT],
                device=self.gnn_model.device,
            )
            prev_positions = self.positions_history[-1].copy()
            prev_quaternions = self.quaternions_history[-1].copy()
            latest_positions = np.empty_like(prev_positions)
            latest_quaternions = np.empty_like(prev_quaternions)
            for name, rel_transform in rel_transforms.items():
                # name = name.split("_")[0]  # ! Adhoc solution
                pose_id = self.name_id_map[name]
                prev_transform = pose_to_transform(
                    np.concatenate(
                        [prev_positions[pose_id], prev_quaternions[pose_id]],
                        axis=-1,
                    )
                )
                latest_transform = rel_transform @ prev_transform
                latest_pose = transform_to_pose(latest_transform)
                latest_positions[pose_id] = latest_pose[:3]
                latest_quaternions[pose_id] = latest_pose[3:]

            self.set_state(
                positions=latest_positions,
                quaternions=latest_quaternions,
                obj_ids=self.name_id_map,
            )

        else:
            raise TypeError(f"Invalid backend {backend}")
