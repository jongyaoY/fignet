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

import json
import os

import numpy as np
import torch
import torch.utils
import torch.utils.data
import tqdm
from torchvision import transforms as T

from fignet.data_loader import MujocoDataset, ToTensor
from fignet.logger import Logger
from fignet.scene import Scene
from fignet.simulator import LearnedSimulator
from fignet.utils import (
    KinematicType,
    optimizer_to,
    rollout,
    rot_diff,
    to_numpy,
    visualize_trajectory,
)


class Trainer:
    """Trainer for the GNN-based simulator"""

    def __init__(
        self,
        sim: LearnedSimulator,
        ref_sim,
        config: dict,
        logger: Logger,
        stats: dict = None,
    ):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._sim = sim
        self._ref_sim = ref_sim
        self._input_seq_length = 2  # TODO
        data_path = config["data_path"]
        meta_data_path = os.path.join(data_path, "metadata.json")
        with open(meta_data_path) as f:
            self._metadata = json.load(f)
        self._scene = Scene(self._metadata["scene_config"])
        self._datasets = {}
        self._datasets["train"] = MujocoDataset(
            data_path,
            self._input_seq_length,
            mode="sample",
            transform=T.Compose([ToTensor(self._device)]),
        )
        self._datasets["val"] = MujocoDataset(
            data_path,
            config["rollout_steps"],
            mode="trajectory",
            transform=T.Compose([ToTensor(self._device)]),
        )
        self._dataloaders = {}
        self._dataloaders["train"] = torch.utils.data.DataLoader(
            self._datasets["train"],
            batch_size=1,
            shuffle=True,
        )
        self._dataloaders["val"] = torch.utils.data.DataLoader(
            self._datasets["val"],
            batch_size=1,
            shuffle=True,
        )

        self._logger = logger
        # Optimization params
        self._lr_init = config["lr_init"]
        self._lr_decay_rate = config["lr_decay_rate"]
        self._lr_decay_steps = config["lr_decay_steps"]
        self._loss_report_step = config["loss_report_step"]
        self._save_model_step = config["save_model_step"]
        self._eval_step = config["eval_step"]
        self._optimizer = torch.optim.Adam(
            self._sim.parameters(), lr=self._lr_init
        )
        self._clip_norm = config["clip_norm"]
        self._stop_step = config.get("training_steps")
        optimizer_to(self._optimizer, self._device)
        # Evaluation params
        self._rollout_steps = config["rollout_steps"]
        self._run_validate = config["run_validate"]
        self._num_eval_rollout = config.get("num_eval_rollout")
        self._save_video = config["save_video"]
        if config.get("model_path"):
            self._sim.load(config.get("model_path"))

        def list_to_numpy(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = list_to_numpy(v)
                else:
                    d[k] = np.asarray(v)
            return d

        if stats is not None:
            self._stats = list_to_numpy(stats)

    def train(self):
        """Run the training loop"""
        self._sim.to(self._device)
        step = 0
        for sample in tqdm.tqdm(self._dataloaders["train"]):
            self._sim.train()
            m_mask = (
                sample["kinematic"]["mesh"].squeeze() == KinematicType.DYNAMIC
            )
            o_mask = (
                sample["kinematic"]["object"].squeeze()
                == KinematicType.DYNAMIC
            )

            m_pred_acc, o_pred_acc = self._sim.predict_accelerations(sample)
            m_target_acc = self._sim.normalize_accelerations(
                sample["target_acc"]["mesh"].squeeze()[m_mask],
                # self._stats['acc']['mesh']
            )
            o_target_acc = self._sim.normalize_accelerations(
                sample["target_acc"]["object"].squeeze()[o_mask],
                # self._stats['acc']['object']
            )
            m_loss = torch.nn.functional.mse_loss(
                m_pred_acc[m_mask], m_target_acc
            )
            o_loss = torch.nn.functional.mse_loss(
                o_pred_acc[o_mask], o_target_acc
            )

            loss = o_loss + m_loss
            self._optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad.clip_grad_norm(
            # self._sim.parameters(),
            # self._clip_norm)
            self._optimizer.step()

            lr_new = (
                self._lr_init
                * self._lr_decay_rate ** (step / self._lr_decay_steps)
                + 1e-6
            )
            for param in self._optimizer.param_groups:
                param["lr"] = lr_new
            self._logger.tb.add_scalar("loss/mesh_nodes", m_loss.item(), step)
            self._logger.tb.add_scalar("loss/obj_nodes", o_loss.item(), step)
            self._logger.tb.add_scalar("loss/total_loss", loss.item(), step)

            if step % self._loss_report_step == 0:
                self._logger.print(f"Loss: {loss}.")
            if step % self._save_model_step == 0:
                ckpt_path = os.path.join(
                    self._logger.log_folder,
                    "models",
                    "weights_itr_{}.ckpt".format(step),
                )
                self._sim.save(ckpt_path)
            if step % self._eval_step == 0 and self._run_validate:
                self.validate(step)

            if self._stop_step is not None and step >= self._stop_step:
                self._logger.print(
                    f"Stopping training after reaching {step} step"
                )
                break
            step += 1

    def validate(self, step: int):
        """Run the validation loop. Sample multiple rollout and calculate
        translational and rotational errors on the last step.

        Args:
            step (int): Current training step
        """
        self._logger.print(f"Validate after {step} steps")
        input_seq_length = self._input_seq_length
        self._sim.eval()
        t_errors = []
        r_errors = []
        for i, traj in enumerate(self._dataloaders["val"]):
            # Get initial states
            init_poses = traj["pose_seq"].squeeze()[0:input_seq_length, ...]
            gt_poses = to_numpy(traj["pose_seq"]).squeeze()
            gt_poses = np.delete(gt_poses, 0, axis=0)
            obj_ids = traj["obj_ids"]
            for k, v in obj_ids.items():
                obj_ids[k] = v.squeeze().cpu().item()
            self._logger.print(f"sampling rollout {i}")
            pred_traj = []
            try:
                pred_traj = rollout(
                    sim=self._sim,
                    init_obj_poses=init_poses,
                    obj_ids=obj_ids,
                    scene=self._scene,
                    device=self._device,
                    nsteps=self._rollout_steps,
                )
            except Exception as e:
                self._logger.print(
                    f"Failed to sample rollout with exception: {e}",
                    level="warn",
                )

            # Calculate translational and rotational errors
            if len(pred_traj) > 0:
                pred_pose = pred_traj[-1, ...]
                gt_pose = gt_poses[-1, ...]
                error_t = (gt_pose[:, :3] - pred_pose[:, :3]) ** 2
                error_t = np.mean(error_t.sum(axis=-1))
                error_r = rot_diff(gt_pose[:, 3:], pred_pose[:, 3:])
                error_r = np.mean(error_r)
                t_errors.append(error_t)
                r_errors.append(error_r)
                # Record video
                if i == 0 and self._save_video:
                    try:
                        screens_pred = visualize_trajectory(
                            self._ref_sim, pred_traj, obj_ids, True
                        )
                        screens_gt = visualize_trajectory(
                            self._ref_sim, gt_poses, obj_ids, True
                        )
                        screens_pred = screens_pred.transpose(0, 3, 1, 2)
                        screens_gt = screens_gt.transpose(0, 3, 1, 2)
                        self._logger.tb.add_video(
                            "video/simulated",
                            torch.from_numpy(screens_pred[None, :]),
                            step,
                            fps=60,
                        )
                        self._logger.tb.add_video(
                            "video/ground_truth",
                            torch.from_numpy(screens_gt[None, :]),
                            step,
                            fps=60,
                        )
                        self
                    except Exception as e:
                        self._logger.print(e, level="warn")

            if (
                self._num_eval_rollout is not None
                and i >= self._num_eval_rollout
            ):
                break
        if len(t_errors) > 0:
            t_mean_error = np.mean(t_errors)
            r_mean_error = np.mean(r_errors)
            self._logger.tb.add_scalar("val/t_mean_error", t_mean_error, step)
            self._logger.tb.add_scalar("val/r_mean_error", r_mean_error, step)
