# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import torch
from itertools import zip_longest
import numpy as np
from ...utils import assert_tensor_equal, logger


def assert_shape(params, settings):
    shape_0 = list(params[0].shape)
    shape_1 = list(params[1].shape)
    if settings["transpose"]:
        shape_1.reverse()
    assert shape_0 == shape_1, f"Shape not same. {shape_0} vs {shape_1}\n"


def assert_weight(params, settings):
    assert_shape(params, settings)
    if settings["transpose"]:
        params[1] = np.transpose(params[1])

    assert_tensor_equal(params[0], params[1], settings)


def assert_grad(params, settings):
    if params[0] is None and params[1] is None:
        return
    elif params[0] is None:
        raise RuntimeError(f"Found grad in base_model is `None`, when another is not!")
    elif params[1] is None:
        raise RuntimeError(f"Found grad in raw_model is `None`, when another is not!")
    assert_shape(params, settings)

    if settings["transpose"]:
        params[1] = np.transpose(params[1])

    assert_tensor_equal(params[0], params[1], settings)


def check_dataloader(first_loader, second_loader, **kwargs):
    def get_numpy(data):
        if isinstance(data, (paddle.Tensor, torch.Tensor)):
            return data.detach().cpu().numpy()
        return data

    options = {
        "atol": 1e-6,
        "rtol": 1e-6,
        "compare_mode": "mean",
    }
    options.update(kwargs)

    for data_0, data_1 in zip_longest(first_loader, second_loader, fillvalue=None):
        if data_0 is None or data_1 is None:
            raise RuntimeError("Given dataloader return difference number of datas.")
        try:
            assert_tensor_equal(get_numpy(data_0), get_numpy(data_1), options)
        except Exception as e:
            logger.error(f"check dataloader failed!!! {type(e).__name__}: {str(e)}")
            return False
    return True
