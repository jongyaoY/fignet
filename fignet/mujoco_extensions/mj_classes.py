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

import mujoco
from robosuite.utils.binding_utils import MjSim

from fignet.mujoco_extensions.mj_utils import set_mjdata


class MjSimLearned(MjSim):

    def __init__(self, model) -> None:
        super(MjSimLearned, self).__init__(model)

    # @classmethod
    # def from_xml_string(cls, xml):
    #     model = mujoco.MjModel.from_xml_string(xml)
    #     return cls(model)

    # @classmethod
    # def from_xml_file(cls, xml_file):
    #     f = open(xml_file, "r")
    #     xml = f.read()
    #     f.close()
    #     return cls.from_xml_string(xml)

    def set_state(self, positions, quaternions, obj_ids):
        set_mjdata(
            self, positions=positions, quaternions=quaternions, obj_ids=obj_ids
        )
        self.forward()

    # def forward(self):
    #     """Forward call to synchronize derived quantities."""
    #     mujoco.mj_forward(self.model._model, self.data._data)

    def step(self, backend: str):
        """Step simulation."""

        if backend == "mujoco":
            mujoco.mj_step(self.model._model, self.data._data)
        elif backend == "gnn":
            raise NotImplementedError
        else:
            raise TypeError(f"Invalid backend {backend}")
