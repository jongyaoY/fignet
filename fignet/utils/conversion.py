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

import dataclasses
from typing import Union

import numpy as np
import torch


def to_numpy(tensor: torch.Tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
        else:
            return tensor.detach().numpy()
    else:
        return tensor


def to_tensor(
    array: Union[np.ndarray, torch.Tensor, int, list], device: str = None
):
    if isinstance(array, torch.Tensor):
        if array.dtype == torch.float64:
            tensor = array.float()
        else:
            tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.int64:
            tensor = torch.from_numpy(array).long()
        else:
            tensor = torch.from_numpy(array).float()
    elif isinstance(array, int):
        tensor = torch.LongTensor([array])
    elif isinstance(array, list):
        for i in len(array):
            array[i] = to_tensor(array[i])
    else:
        raise TypeError(f"Cannot conver {type(array)} to tensor")

    if device:
        return tensor.to(device)
    else:
        return tensor


def dict_to_tensor(d: dict):
    new_dict = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = dict_to_tensor(v)
        else:
            new_dict[k] = to_tensor(v)
    return new_dict


def dataclass_to_tensor(d, device=None):

    if isinstance(d, np.ndarray) or isinstance(d, torch.Tensor):
        return to_tensor(d, device=device)
    elif dataclasses.is_dataclass(d):
        for f in dataclasses.fields(d):
            setattr(
                d,
                f.name,
                dataclass_to_tensor(getattr(d, f.name), device=device),
            )
        return d
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = dataclass_to_tensor(v, device=device)
        return d
