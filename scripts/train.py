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

import argparse

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import TensorBoardLogger

from fignet.data.datasets import create_dataloaders
from fignet.graph_builders import GraphBuildCfg
from fignet.modules.simulator import SimCfg
from fignet.pl_modules.pl_model import LighteningLearnedSimulator, warm_up

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", required=True)
args = parser.parse_args()


def load_config(args):
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_config(args)
    model = LighteningLearnedSimulator(
        batch_size=config["datasets"]["batch_size"],
        latent_dim=config["gnn"]["latent_dim"],
        message_passing_steps=config["gnn"]["message_passing_steps"],
        mlp_layers=config["gnn"]["mlp_layers"],
        mlp_hidden_dim=config["gnn"]["mlp_hidden_dim"],
    )
    sim_cfg = SimCfg(
        build_cfg=GraphBuildCfg(
            type=config["simulator"]["graph_builder"]["type"],
            noise_std=config["simulator"]["graph_builder"]["noise_std"],
        ),
        input_sequence_length=config["simulator"]["input_seq_length"],
        collision_radius=config["simulator"]["collision_radius"],
    )
    data_loaders = create_dataloaders(
        val_ratio=config["datasets"]["val_ratio"],
        num_workers=config["datasets"]["num_workers"],
        batch_size=config["datasets"]["batch_size"],
        rollout_datapath=config["datasets"]["rollout"],
        rollout_length=config["datasets"]["rollout_length"],
        root=config["datasets"]["train"],
        build_config=sim_cfg.build_cfg,
    )
    warm_up(
        model=model.gnn_model,
        sim_cfg=sim_cfg,
        dataloader=data_loaders["train"],
        num_batches=config["training"]["warmup_steps"],
    )
    logger = TensorBoardLogger(
        save_dir=config["logging"]["folder"],
        name=config["logging"]["experiment"],
    )

    trainer = pl.Trainer(
        logger=logger,
        max_steps=config["training"]["total_steps"],
        log_every_n_steps=10,
        val_check_interval=config["training"]["val_step"],
        gpus=1,
    )
    trainer.fit(
        model,
        train_dataloaders=data_loaders["train"],
        val_dataloaders=[data_loaders["val"], data_loaders["rollout"]],
    )
