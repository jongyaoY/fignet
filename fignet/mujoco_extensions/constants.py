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

from enum import Enum

import mujoco
import numpy as np


# Define a custom enum for joint types
class JointType(Enum):
    FREE = mujoco.mjtJoint.mjJNT_FREE
    BALL = mujoco.mjtJoint.mjJNT_BALL
    SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
    HINGE = mujoco.mjtJoint.mjJNT_HINGE


def create_onehot_table(types):
    lookup_table = {}

    # Create one-hot vectors and populate the lookup table
    for index, joint in enumerate(types):
        one_hot_vector = np.zeros(len(types), dtype=int)
        one_hot_vector[index] = 1
        lookup_table[joint] = one_hot_vector

    return lookup_table


JOINT_TYPE_VECTOR = create_onehot_table(list(JointType))
