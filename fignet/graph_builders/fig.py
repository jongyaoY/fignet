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

import numpy as np

from fignet.data.scene_info import SceneInfoDict, SceneInfoKey
from fignet.graph_builders.base import GraphBuilder
from fignet.graph_builders.common import (
    FIGEdgeType,
    FIGNodeType,
    GraphMetaData,
    cal_inner_connectivity,
)

EdgeTypes = [
    FIGEdgeType.OConnectV,
    FIGEdgeType.VConnectO,
    FIGEdgeType.VCollideV,
]

NodeTypes = [FIGNodeType.VERT, FIGNodeType.OBJECT]
meta_data = GraphMetaData(node_dim=8, edge_dim=8)
fedge_dim = 4 + 2 * (3 * 4) + 2 * 3  # TODO


class FIGBuilder(GraphBuilder):

    def cal_edge_features(self, edge_type, index, edge_info):
        features = None
        if index.shape[1] == 0:
            if edge_type == FIGEdgeType.VCollideV:
                return np.empty((0, fedge_dim), dtype=np.float64)  # TODO
            else:
                return np.empty((0, meta_data.edge_dim), dtype=np.float64)
        if (
            edge_type == FIGEdgeType.VConnectO
            or edge_type == FIGEdgeType.OConnectV
        ):
            obj_com = edge_info["com_pos"]
            vert_pos = edge_info["verts_pos"]
            obj_ref_com = edge_info["com_ref_pos"]
            vert_ref_pos = edge_info["verts_ref_pos"]
            s_index = index[0]
            r_index = index[1]
            if edge_type == FIGEdgeType.VConnectO:
                lhs = vert_pos
                rhs = obj_com
                lhs_ref = vert_ref_pos
                rhs_ref = obj_ref_com
            elif edge_type == FIGEdgeType.OConnectV:
                lhs = obj_com
                rhs = vert_pos
                lhs_ref = obj_ref_com
                rhs_ref = vert_ref_pos
            d_rs = lhs[s_index] - rhs[r_index]
            d_rs_ref = lhs_ref[s_index] - rhs_ref[r_index]
            d_rs = np.concatenate(
                [d_rs, np.linalg.norm(d_rs, axis=1, keepdims=True)], axis=-1
            )
            d_rs_ref = np.concatenate(
                [d_rs_ref, np.linalg.norm(d_rs_ref, axis=1, keepdims=True)],
                axis=-1,
            )
            features = np.hstack([d_rs, d_rs_ref])

        elif edge_type == FIGEdgeType.VCollideV:
            assert (
                edge_info["contact_points"].shape[0] == index.shape[1]
            )  # num edges
            s_index = index[0]  # (num_edge, 3)
            r_index = index[1]
            verts1 = edge_info["verts_pos"][s_index]  # (num_edge, 3, mesh_dim)
            verts2 = edge_info["verts_pos"][r_index]
            point1 = edge_info["contact_points"][
                :, 0, :
            ]  # (num_edge, mesh_dim)
            point2 = edge_info["contact_points"][:, 1, :]
            norm_s = edge_info["contact_normals"][:, 0, :]
            norm_r = edge_info["contact_normals"][:, 1, :]
            d_rs = (
                point2 - point1
            )  # p_r - p_s from receiver to sender (num_edge, mesh_dim)
            d_si = verts1 - np.expand_dims(
                point1, axis=1
            )  # sender (num_edge, 3, mesh_dim)
            d_ri = verts2 - np.expand_dims(
                point2, axis=1
            )  # receiver (num_edge, 3, mesh_dim)
            d_si_norms = np.linalg.norm(d_si, axis=-1)  # (num_edge, 3)
            d_ri_norms = np.linalg.norm(d_ri, axis=-1)

            ord_s = np.argsort(d_si_norms, axis=1)
            ord_r = np.argsort(d_ri_norms, axis=1)
            d_si_norms = np.take_along_axis(d_si_norms, ord_s, axis=1)
            d_ri_norms = np.take_along_axis(d_ri_norms, ord_r, axis=1)

            d_si = np.take_along_axis(
                d_si, np.expand_dims(ord_s, axis=2), axis=1
            )
            d_ri = np.take_along_axis(
                d_ri, np.expand_dims(ord_r, axis=2), axis=1
            )

            s_index = np.take_along_axis(s_index, ord_s, axis=1)
            r_index = np.take_along_axis(r_index, ord_r, axis=1)

            d_si_norms_ = np.linalg.norm(d_si, axis=-1)
            d_ri_norms_ = np.linalg.norm(d_ri, axis=-1)
            assert np.allclose(d_si_norms, d_si_norms_)
            assert np.allclose(d_ri_norms, d_ri_norms_)

            index[0] = s_index
            index[1] = r_index

            d_rs = np.concatenate(
                [d_rs, np.linalg.norm(d_rs, axis=1, keepdims=True)], axis=-1
            )
            d_si = np.concatenate(
                [d_si, np.expand_dims(d_si_norms, axis=-1)], axis=-1
            )
            d_ri = np.concatenate(
                [d_ri, np.expand_dims(d_ri_norms, axis=-1)], axis=-1
            )

            features = np.concatenate(
                [
                    d_rs,  # (num_edge, 4)
                    d_si.reshape(d_si.shape[0], -1),  # (num_edge, 3*4)
                    d_ri.reshape(d_ri.shape[0], -1),  # (num_edge, 3*4)
                    norm_s,  # (num_edge, 3)
                    norm_r,  # (num_edge, 3)
                ],
                axis=-1,
            )
        else:
            raise NotImplementedError
        return features

    def cal_connectivity(self, scn_info: SceneInfoDict):
        index_dict = {}
        attr_dict = {}
        for edge_type in EdgeTypes:
            senders = []
            receivers = []
            edge_info = {}
            if edge_type == FIGEdgeType.VCollideV:
                contact_paris = scn_info[SceneInfoKey.CONTACT_PAIRS]
                edge_info["contact_points"] = []
                edge_info["contact_normals"] = []
                for (name1, name2), contact_info in contact_paris.items():
                    num_contacts = len(contact_info["contact_points"])
                    m_offset_s = scn_info[SceneInfoKey.VERT_OFFSETS_DICT].get(
                        name1
                    )
                    m_offset_r = scn_info[SceneInfoKey.VERT_OFFSETS_DICT].get(
                        name2
                    )
                    for contact_i in range(num_contacts):
                        s_idx, r_idx = contact_info["vert_ids_local"][
                            contact_i
                        ]

                        s_idx = m_offset_s + s_idx
                        r_idx = m_offset_r + r_idx
                        # if np.any(s_idx > 1000) or np.any(r_idx > 1000):
                        #     print('wtf')
                        senders.append(s_idx)
                        receivers.append(r_idx)
                        edge_info["contact_points"].append(
                            contact_info["contact_points"][contact_i]
                        )
                        edge_info["contact_normals"].append(
                            contact_info["contact_normals"][contact_i]
                        )
                edge_info["verts_pos"] = scn_info[SceneInfoKey.VERT_SEQ][
                    -1, ...
                ]
                edge_info["contact_points"] = np.asarray(
                    edge_info["contact_points"]
                )
                edge_info["contact_normals"] = np.asarray(
                    edge_info["contact_normals"]
                )
            elif edge_type == FIGEdgeType.OConnectV:
                senders, receivers, edge_info = cal_inner_connectivity(
                    scn_info=scn_info, direction="o-v"
                )
            elif edge_type == FIGEdgeType.VConnectO:
                senders, receivers, edge_info = cal_inner_connectivity(
                    scn_info=scn_info, direction="v-o"
                )

            else:
                raise NotImplementedError

            if len(senders) > 0:
                if edge_type == FIGEdgeType.VCollideV:
                    senders = np.expand_dims(
                        np.vstack(senders), axis=0
                    )  # (1, num_edge, 3)
                    receivers = np.expand_dims(
                        np.vstack(receivers), axis=0
                    )  # (1, num_edge, 3)
                else:
                    senders = np.concatenate(senders, dtype=np.int64, axis=-1)
                    receivers = np.concatenate(
                        receivers, dtype=np.int64, axis=-1
                    )
                index = np.vstack([senders, receivers])
            else:
                if edge_type == FIGEdgeType.VCollideV:
                    index = np.empty((2, 0, 3), dtype=np.int64)
                else:
                    index = np.empty((2, 0), dtype=np.int64)
                # attr = np.asarray([])
            attr = self.cal_edge_features(
                edge_type=edge_type,
                index=index,  # index modified for ff edges
                edge_info=edge_info,
            )
            index_dict.update({edge_type: index})
            attr_dict.update({edge_type: attr})
        return index_dict, attr_dict

    def cal_node_attr(self, scn_info: SceneInfoDict):
        node_attr_dict = {}
        seq_len = scn_info[SceneInfoKey.VERT_SEQ].shape[0]
        vert_seq = scn_info[SceneInfoKey.VERT_SEQ]
        com_seq = scn_info[SceneInfoKey.COM_SEQ]
        m_kin = scn_info[SceneInfoKey.VERT_KINEMATIC]
        o_kin = scn_info[SceneInfoKey.OBJ_KINEMATIC]
        m_prop = scn_info[SceneInfoKey.VERT_PROP]
        o_prop = scn_info[SceneInfoKey.OBJ_PROP]
        m_target = scn_info.get(SceneInfoKey.VERT_TARGET, None)
        o_target = scn_info.get(SceneInfoKey.COM_TARGET, None)
        m_features = []
        o_features = []
        for i in range(seq_len):
            if i + 1 < seq_len:
                m_vel = vert_seq[i + 1, ...] - vert_seq[i, ...]
                o_vel = com_seq[i + 1, ...] - com_seq[i, ...]
                m_features.append(m_vel)
                o_features.append(o_vel)
        m_features.append(m_prop)
        m_features.append(m_kin)
        o_features.append(o_prop)
        o_features.append(o_kin)
        m_features = np.concatenate(m_features, axis=-1)
        o_features = np.concatenate(o_features, axis=-1)
        node_attr_dict = {
            FIGNodeType.VERT: {
                "x": m_features,
                "kinematic": m_kin,
                "y": m_target,
            },
            FIGNodeType.OBJECT: {
                "x": o_features,
                "kinematic": o_kin,
                "y": o_target,
            },
        }
        return node_attr_dict
