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
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

from numpy.typing import NDArray

from fignet.mujoco_extensions.constants import JointType


class MetaEnum(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class StrEnum(enum.Enum, metaclass=MetaEnum):
    def __str__(self) -> str:
        return self.value


K = TypeVar("K", bound=enum.Enum)  # Keys are Enums
V = TypeVar("V")  # Values can be any type


class EnumKeyDict(Dict[str, V], Generic[K, V]):

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, enum.Enum):
            key = str(key)

        super().__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, enum.Enum):
            key = str(key)
        return super().__getitem__(key)


class KinematicType(enum.IntEnum):
    STATIC = 0
    DYNAMIC = 1
    SIZE = 1


class StaticInfo(TypedDict):
    kinematic: KinematicType
    prperties: NDArray


class DynamicInfo(TypedDict):
    pos_seq: NDArray
    ref_pos: NDArray
    quat_seq: Optional[NDArray]
    target: Optional[NDArray]


class StaticDynamicInfo(TypedDict):
    static_info: StaticInfo
    dynamic_info: DynamicInfo


class ContactSpecs(TypedDict):
    points: List[Tuple[NDArray, NDArray]]
    normals: List[Tuple[NDArray, NDArray]]
    vert_idx: List[Tuple[NDArray, NDArray]]
    face_idx: List[Tuple[NDArray, NDArray]]
    mesh_idx: List[Tuple[NDArray, NDArray]]


class JointSpecs(TypedDict):
    name: str
    axis: NDArray
    type: JointType


ContactInfo = Dict[Tuple[str, str], ContactSpecs]
JointInfo = Dict[Tuple[str, str], JointSpecs]


class RelationType(enum.Enum):
    CONTACT = "contact"
    JOINT = "joint"


class MeshInteractionInfo(TypedDict):
    vert_addr_dict: Dict[str, int]
    com_addr_dict: Dict[str, int]
    vert_info: StaticDynamicInfo
    com_info: StaticDynamicInfo
    relations: Dict[RelationType, Union[ContactInfo, JointInfo]]
