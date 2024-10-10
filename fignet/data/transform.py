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

from fignet.data.hetero_graph import HeteroGraph
from fignet.data.scene_info import SceneInfoDict
from fignet.graph_builders import GraphBuildCfg
from fignet.utils.conversion import dict_to_tensor
from fignet.utils.scene import build_graph


class ToHeteroGraph(object):

    def __init__(self, config: GraphBuildCfg):
        self.config = config

    def __call__(self, scn_info: SceneInfoDict) -> HeteroGraph:

        return build_graph(scn_info, self.config, True)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        # convert numpy arrays to pytorch tensors
        if isinstance(sample, dict):
            return dict_to_tensor(sample, self.device)
        else:
            raise TypeError(f"Cannot convert {type(sample)} to tensor.")
