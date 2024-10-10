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

import enum
from typing import Any, Dict

from fignet.data.types import MetaEnum


class SceneInfoKey(enum.Enum, metaclass=MetaEnum):
    VERT_SEQ = "vert_seq"
    COM_SEQ = "com_seq"
    VERT_PROP = "vert_prop"
    OBJ_PROP = "obj_prop"
    VERT_KINEMATIC = "vert_kinematic"
    OBJ_KINEMATIC = "obj_kinematic"
    VERT_REF_POS = "vert_ref_pos"
    COM_REF_POS = "com_ref_pos"
    VERT_OFFSETS_DICT = "vert_offsets_dict"
    NUM_VERTS_DICT = "num_verts_dict"
    OBJ_OFFSETS_DICT = "obj_offsets_dict"
    VERT_TARGET = "vert_target"
    COM_TARGET = "com_target"

    CONTACT_PAIRS = "contact_pairs"


SceneInfoDict = Dict[SceneInfoKey, Any]
