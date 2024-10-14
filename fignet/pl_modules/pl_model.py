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

from typing import Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm

import fignet
from fignet.data.hetero_graph import HeteroGraph
from fignet.data.types import KinematicType
from fignet.graph_builders import FIGNodeType
from fignet.modules.simulator import LearnedSimulator, SimCfg
from fignet.utils.geometric import rot_diff


def warm_up(
    model: LearnedSimulator, sim_cfg: SimCfg, dataloader, num_batches: int
):
    from torch_geometric.transforms import ToDevice

    if not model.initialized:
        sample = next(iter(dataloader))
        model.init(sample, sim_cfg)
    device = model.device
    to_device = ToDevice(device)
    for batch_idx, batch in tqdm.tqdm(
        enumerate(dataloader), desc="Warm up step"
    ):
        if batch_idx >= num_batches:
            break
        batch = to_device(batch)
        model(batch)
        model.normalize_accelerations(batch[FIGNodeType.VERT].y)


class LighteningLearnedSimulator(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        num_rollouts: int = 1,
        lr_init: float = 1e-3,
        lr_decay_rate: float = 0.1,
        lr_decay_steps: int = 1000000,
        **kwargs
    ):
        super(LighteningLearnedSimulator, self).__init__()
        self.gnn_model = LearnedSimulator(**kwargs)
        self.save_hyperparameters()

    def forward(self, x):
        return self.gnn_model(x)

    def training_step(self, batch: HeteroGraph):
        loss = self._cal_loss(batch)
        self.log("train_loss", loss)

        return loss

    def validation_step(
        self,
        batch: Union[HeteroGraph, Dict],
        batch_idx: int,
        dataloader_idx: int = 0,
        *args,
        **kwargs
    ):
        if isinstance(batch, HeteroGraph):
            # https://github.com/pyg-team/pytorch_geometric/issues/9315
            loss = self._cal_loss(batch)
            return loss
        elif isinstance(batch, dict):
            if batch_idx >= self.hparams.num_rollouts:
                return
            errors = self._rollout_step(batch)
            # for k, v in errors.items():
            #     self.log(
            #         k,
            #         v,
            #         batch_size=1
            #     )
            return errors

        else:
            raise NotImplementedError

    def validation_epoch_end(self, outputs):
        val_loss = outputs[0]
        errors = outputs[1]
        # for idx, val in enumerate(val_loss):
        avg_val_loss = torch.stack(val_loss).mean()
        self.log("val_loss", avg_val_loss)
        keys = next(iter(errors)).keys()
        # avg_errors = {}
        for k in keys:
            avg_err = torch.stack([torch.tensor(x[k]) for x in errors]).mean()
            self.log(
                "/".join(["errors", k]),
                avg_err,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr_init
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=self.hparams.lr_decay_rate,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def state_dict(self, *args, **kwargs):
        return self.gnn_model.state_dict()

    def load_state_dict(self, *args, **kwargs):
        return self.gnn_model.load_state_dict(*args, **kwargs)

    def _cal_loss(self, batch: HeteroGraph):
        non_kinematic_mask = (
            batch[FIGNodeType.VERT].kinematic == KinematicType.DYNAMIC
        ).to(batch[FIGNodeType.VERT].kinematic.device)

        pred_y = self(batch)
        y = self.gnn_model.normalize_accelerations(batch[FIGNodeType.VERT].y)
        loss = torch.nn.functional.mse_loss(pred_y, y, reduction="none")
        loss = loss.sum(dim=-1)
        num_non_kinematic = non_kinematic_mask.sum()
        loss = loss * non_kinematic_mask.squeeze()
        loss = loss.sum() / num_non_kinematic

        return loss

    def _rollout_step(self, sample: dict):
        gnn_model = self.gnn_model
        input_seq_length = gnn_model.cfg.input_sequence_length
        gt_traj = sample["pose_seq"]
        pose_addr = sample["pose_addr"]
        mujoco_xml = sample["mujoco_xml"]
        ep_len = gt_traj.shape[0]
        pred_traj = fignet.rollout(
            gnn_model=gnn_model,
            init_obj_poses=gt_traj[:input_seq_length, ...],
            obj_ids=pose_addr,
            mujoco_xml=mujoco_xml,
            nsteps=ep_len,
        )
        # Calculate rollout errors

        rollout_errors = self._cal_pose_errors(
            pred_pose=pred_traj[-1, ...], target_pose=gt_traj[-1, ...]
        )
        # Calculate one step errors
        one_step_errors = self._cal_pose_errors(
            pred_pose=pred_traj[1, ...], target_pose=gt_traj[1, ...]
        )

        return {
            "onestep_trans": one_step_errors["translation"],
            "onestep_rot": one_step_errors["rotation"],
            "rollout_trans": rollout_errors["translation"],
            "rollout_rot": rollout_errors["rotation"],
        }

    def _cal_pose_errors(self, pred_pose, target_pose):
        error_t = (target_pose[:, :3] - pred_pose[:, :3]) ** 2
        error_t = np.mean(error_t.sum(axis=-1))
        error_t = np.sqrt(error_t)
        error_r = rot_diff(target_pose[:, 3:], pred_pose[:, 3:])
        error_r = np.mean(error_r)

        return {"translation": error_t, "rotation": error_r}
