# MIT License
#
# Copyright (c) 2021 Stanford Interactive Perception and Robot Learning Lab
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

import datetime
import logging
import os
import sys
import time

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, config):
        self.config = config
        time_str = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%Y-%m-%d-%H:%M"
        )
        self.tb_prefix = ""
        if config.get("continue_log_from") is None:
            prefix_str = time_str
        else:
            prefix_str = config["continue_log_from"]
            if not os.path.exists(
                os.path.join(self.config["logging_folder"], prefix_str)
            ):
                prefix_str = time_str
        self.log_folder = os.path.join(
            self.config["logging_folder"], prefix_str
        )
        self.create_folder_structure()
        self.setup_loggers()

    def create_folder_structure(self):
        """
        Creates the folder structure for logging. Subfolders can be added here
        """
        base_dir = self.log_folder
        sub_folders = ["runs", "models"]

        if not os.path.exists(self.config["logging_folder"]):
            os.mkdir(self.config["logging_folder"])

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        for sf in sub_folders:
            if not os.path.exists(os.path.join(base_dir, sf)):
                os.mkdir(os.path.join(base_dir, sf))

    def setup_loggers(self):
        """
        Sets up a logger that logs to both file and stdout
        """
        log_path = os.path.join(self.log_folder, "log.log")

        self.print_logger = logging.getLogger()
        self.print_logger.setLevel(
            getattr(logging, self.config["log_level"].upper(), None)
        )
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ]
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        for h in handlers:
            h.setFormatter(formatter)
            self.print_logger.addHandler(h)

        # Setup Tensorboard
        self.tb = SummaryWriter(
            os.path.join(self.log_folder, "runs", self.tb_prefix)
        )

    def print(self, *args, level="info"):
        """
        Wrapper for print statement
        """
        if level == "warn":
            self.print_logger.warning(args)
        else:
            self.print_logger.info(args)
