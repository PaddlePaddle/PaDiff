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
from ...configs import global_yaml_loader
from ...utils import load_numpy, struct_info_log, assert_tensor_equal, log


def process_each_param(process, node_lists, reports, compare_target, cfg):
    for node_0, node_1 in zip_longest(node_lists[0], node_lists[1], fillvalue=None):
        if node_0 is None or node_1 is None:
            raise RuntimeError("Found model with difference number of sublayers. Check your model.")
        for (param_name_0, param_path_0), (param_name_1, param_path_1) in zip(
            node_0[compare_target].items(),
            node_1[compare_target].items(),
        ):
            try:
                settings = global_yaml_loader.get_weight_settings(
                    (node_0["name"], node_1["name"]),
                    (reports[0]["framework"], reports[1]["framework"]),
                    (param_name_0, param_name_1),
                )
                settings.update(cfg)
                param_0 = load_numpy(param_path_0)
                param_1 = load_numpy(param_path_1)
                process([node_0, node_1], [param_name_0, param_name_1], [param_0, param_1], settings)
            except Exception as e:
                err_str = f"{type(e).__name__ + ':  ' + str(e)}\n"
                err_str += f"Error occured between:\n"
                err_str += f"    (base_model):  {node_0['route'] + '.' + param_name_0}\n"
                err_str += f"    (raw_model):   {node_1['route'] + '.' + param_name_1}\n\n"

                err_str += struct_info_log(reports, (compare_target, compare_target), compare_target)

                raise RuntimeError(err_str)


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
        "atol": 0,
        "rtol": 1e-7,
        "compare_mode": "mean",
    }
    options.update(kwargs)

    for data_0, data_1 in zip_longest(first_loader, second_loader, fillvalue=None):
        if data_0 is None or data_1 is None:
            raise RuntimeError("Given dataloader return difference number of datas.")
        try:
            assert_tensor_equal(get_numpy(data_0), get_numpy(data_1), options)
        except Exception as e:
            log("check dataloader failed!!!")
            print(f"{type(e).__name__ + ':  ' + str(e)}")
            return False
    return True
