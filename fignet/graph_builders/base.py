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

from fignet.graph_builders.common import GraphBuildResult
from fignet.scene import SceneInfoDict

REGISTERED_BUILDERS = {}


def register_builder(builder):
    if builder.__name__ not in REGISTERED_BUILDERS:
        REGISTERED_BUILDERS[
            builder.__name__.replace("Builder", "").lower()
        ] = builder


def make_builder(build_type, *args, **kwargs):
    if build_type not in REGISTERED_BUILDERS and build_type:
        raise Exception(
            f"Unknown build type '{build_type}', \
            available types are {list(REGISTERED_BUILDERS.keys())}"
        )
    return REGISTERED_BUILDERS[build_type](*args, **kwargs)


class BuilderMeta(type):

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        _exclude_classes = ["GraphBuilder"]
        if cls.__name__ not in _exclude_classes:
            register_builder(cls)
        return cls


class GraphBuilder(metaclass=BuilderMeta):
    def __init__(self, config=None, *args, **kwargs) -> None:
        self.config = config
        self.name = "base_builder"
        self.meta_data = None  # TODO: info about node, edge dim, num etc.

    def build(self, scn_info: SceneInfoDict) -> GraphBuildResult:
        edge_index_dict, edge_attr_dict = self.cal_connectivity(scn_info)
        node_attr_dict = self.cal_node_attr(scn_info)
        return GraphBuildResult(
            node_attr_dict=node_attr_dict,
            edge_attr_dict=edge_attr_dict,
            edge_index_dict=edge_index_dict,
        )

    def cal_connectivity(self, scn_info: SceneInfoDict):
        raise NotImplementedError

    def cal_node_attr(self, scn_info: SceneInfoDict):
        raise NotImplementedError
