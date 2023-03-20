# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from itertools import zip_longest

import numpy
import paddle
import torch

from .utils import (
    log,
    map_for_each_sublayer,
    LayerMap,
    weight_struct_info,
    model_repr_info,
)
from .file_loader import global_yaml_loader as yamls

from .special_init import global_special_init_pool as init_pool
from .special_init import build_name


def process_each_weight(process, layer, module, layer_map=LayerMap()):
    """
    Apply process for each pair of parameters in layer(paddle) and module(torch)

    Args:
        process (function): process applied to parameters
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_map (dict, optional): manually map paddle layer and torch module.
    """

    def _process_runner(
        process,
        paddle_sublayer,
        torch_submodule,
        paddle_pname,
        torch_pname,
        paddle_param,
        torch_param,
    ):

        settings = yamls.get_weight_settings(paddle_sublayer, torch_submodule, paddle_pname)

        process(
            paddle_sublayer,
            torch_submodule,
            paddle_pname,
            torch_pname,
            paddle_param,
            torch_param,
            settings,
        )

    layers = layer_map.weight_init_layers(layer)
    modules = layer_map.weight_init_layers(module)

    for paddle_sublayer, torch_submodule in zip_longest(layers, modules, fillvalue=None):
        if paddle_sublayer is None or torch_submodule is None:
            raise RuntimeError("Torch and Paddle return difference number of sublayers. Check your model.")
        for (paddle_pname, paddle_param), (torch_pname, torch_param) in zip(
            paddle_sublayer.named_parameters(prefix="", include_sublayers=False),
            torch_submodule.named_parameters(prefix="", recurse=False),
        ):
            try:
                _process_runner(
                    process,
                    paddle_sublayer,
                    torch_submodule,
                    paddle_pname,
                    torch_pname,
                    paddle_param,
                    torch_param,
                )
            except Exception as e:
                err_str = f"Error occured between:\n"
                err_str += f"    paddle: `{model_repr_info(paddle_sublayer)}`\n"
                err_str += f"            `{paddle_sublayer.padiff_path + '.' + paddle_pname}`\n"
                err_str += f"    torch: `{model_repr_info(torch_submodule)}`\n"
                err_str += f"           `{torch_submodule.padiff_path + '.' + torch_pname}`\n\n"
                err_str += f"{str(e)}\n"
                err_str += weight_struct_info(layer, module, paddle_sublayer, torch_submodule)
                raise RuntimeError(err_str)


def shape_check(
    paddle_sublayer,
    torch_submodule,
    paddle_pname,
    torch_pname,
    paddle_param,
    torch_param,
    settings,
):
    p_shape = list(paddle_param.shape)
    t_shape = list(torch_param.shape)
    if settings["transpose"]:
        t_shape.reverse()
    assert p_shape == t_shape, ("Shape of paddle param `{}` and torch param `{}` is not the same. {} vs {}\n").format(
        paddle_pname, torch_pname, p_shape, t_shape
    )


def _assign_weight(
    paddle_sublayer,
    torch_submodule,
    paddle_pname,
    torch_pname,
    paddle_param,
    torch_param,
    settings,
):
    shape_check(
        paddle_sublayer,
        torch_submodule,
        paddle_pname,
        torch_pname,
        paddle_param,
        torch_param,
        settings,
    )
    np_value = torch_param.data.detach().cpu().numpy()
    if settings["transpose"]:
        np_value = numpy.transpose(np_value)

    paddle.assign(paddle.to_tensor(np_value), paddle_param)


def assign_weight(layer, module, layer_map=LayerMap()):
    """
    Init weights of layer(paddle) and module(torch) with same value

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
    """

    assert isinstance(layer, paddle.nn.Layer), "The first param of assign_weight should be a paddle.nn.Layer"
    assert isinstance(module, torch.nn.Module), "The second param of assign_weight should be a torch.nn.Module"

    for torch_submodule, paddle_sublayer in layer_map.special_init_layers():
        paddle_layer_name = paddle_sublayer.__class__.__name__
        torch_module_name = torch_submodule.__class__.__name__
        key_name = build_name(paddle_layer_name, torch_module_name)
        if key_name not in init_pool.funcs.keys():
            log(
                "*** Special init paddle layer `{}` and torch module `{}` is not supported ***".format(
                    paddle_layer_name, torch_module_name
                )
            )
            log("    Checkout the parameters are inited by yourself")
            log("    ,or you can register your init method!")
        else:
            try:
                init_pool.funcs[key_name](paddle_sublayer, torch_submodule)
            except Exception as e:
                print(
                    f"Special init paddle layer `{paddle_layer_name}` and torch module `{torch_module_name}` failed."
                )
                print(str(e))
                log("Assign weight Failed !!!")
                return False

    try:
        process_each_weight(_assign_weight, layer, module, layer_map)
        log("Assign weight success !!!")
        return True
    except Exception as e:
        log("Assign weight Failed !!!\n")
        print(str(e))
        return False


def remove_inplace(layer, module):
    """
    Set `inplace` tag to `False` for torch module

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
    """

    def _remove_inplace(layer, module):
        if hasattr(module, "inplace"):
            module.inplace = False

    map_for_each_sublayer(_remove_inplace, layer, module)
