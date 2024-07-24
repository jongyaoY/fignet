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
import json
import os

import torch

from fignet.logger import Logger
from fignet.simulator import LearnedSimulator
from fignet.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file", required=False, default="config/train.json"
)
args = parser.parse_args()

if __name__ == "__main__":

    latent_dim = 128

    with open(os.path.join(os.getcwd(), args.config_file)) as f:
        config = json.load(f)
    logger = Logger(config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.print("Using GPU")
    else:
        device = torch.device("cpu")
        logger.print("Using CPU")
    sim = LearnedSimulator(
        mesh_dimensions=3,
        latent_dim=latent_dim,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=latent_dim,
        device=device,
    )
    trainer = Trainer(sim=sim, logger=logger, config=config)
    trainer.train()
