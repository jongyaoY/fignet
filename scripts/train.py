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

import torch

from fignet.logger import Logger
from fignet.simulator import LearnedSimulator
from fignet.trainer import Trainer

if __name__ == "__main__":

    latent_dim = 128

    config = {
        "test_data_path": "datasets/mujoco_moviA_1000.npz",
        "data_path": "datasets/mujoco_moviA_1000000.npz",
        "logging_folder": "log",
        # "log_level": "debug",
        "log_level": "info",
        "lr_init": 1e-3,
        "lr_decay_rate": 0.1,
        "lr_decay_steps": 1e6,
        "loss_report_step": 10,
        "save_model_step": 1000,
        "eval_step": 1000,
        # "clip_norm": 1e-2,
        "rollout_steps": 50,
        "run_validate": True,
        "num_eval_rollout": 10,
        "save_video": True,
        "warmup_steps": 1000,
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda")
    else:
        device = torch.device("cpu")
        print("using cpu")
    sim = LearnedSimulator(
        mesh_dimensions=3,
        latent_dim=latent_dim,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=latent_dim,
        noise_std=1e-4,
        device=device,
    )
    logger = Logger(config)
    trainer = Trainer(sim=sim, logger=logger, config=config)
    trainer.train()
