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
    assert_tensor_equal,
    LayerMap,
    weight_struct_info,
    log_file,
    diff_log_path,
)
from .file_loader import global_yaml_loader as yamls

from .special_init import global_special_init_pool as init_pool


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
                log(
                    f"Error occurred while process paddle parameter `{paddle_pname}` and torch parameter `{torch_pname}`!"
                )
                log("Error Msg:\n")
                print(f"{str(e)}")
                weight_struct_info(layer, module, paddle_sublayer, torch_submodule)
                raise RuntimeError("Error occured while `process_each_weight`")


def _shape_check(
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
    assert p_shape == t_shape, (
        "Shape of paddle param `{}` and torch param `{}` is not the same. {} vs {}\n"
        "Hint: \n"
        "      1. check whether your paddle model definition and torch model definition are corresponding.\n"
        "      2. check the weight shape of paddle:`{}` and torch:`{}` is the same.\n"
    ).format(paddle_pname, torch_pname, p_shape, t_shape, paddle_sublayer, torch_submodule)


def _assign_weight(
    paddle_sublayer,
    torch_submodule,
    paddle_pname,
    torch_pname,
    paddle_param,
    torch_param,
    settings,
):
    _shape_check(
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
        layer_name = paddle_sublayer.__class__.__name__
        if layer_name not in init_pool.funcs.keys():
            log(
                "*** Special init paddle layer `{}` and torch module `{}` is not supported ***".format(
                    paddle_sublayer.__class__.__name__, torch_submodule.__class__.__name__
                )
            )
            log("    Checkout the parameters are inited by yourself")
            log("    ,or you can register your init method!")
        else:
            try:
                init_pool.funcs[layer_name](paddle_sublayer, torch_submodule)
            except Exception as e:
                print(f"Special init Layer`{layer_name}` failed.")
                print(str(e))
                log("Assign weight Failed !!!")
                return False

    try:
        process_each_weight(_assign_weight, layer, module, layer_map)
        log("Assign weight success !!!")
        return True
    except Exception as e:
        log("Assign weight Failed !!!")
        print(str(e))
        return False


def check_weight_grad(layer, module, options, layer_map=LayerMap()):
    """
    Compare weights and grads between layer(paddle) and module(torch)

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_map (dict, optional): manually map paddle layer and torch module.
    """
    if options["diff_phase"] == "forward":
        log("Diff_phase is `forward`. Weight and grad check skipped.")
        return True, True

    _weight_check = True
    _grad_check = True

    def _check_weight_grad(
        paddle_sublayer,
        torch_submodule,
        paddle_pname,
        torch_pname,
        paddle_param,
        torch_param,
        settings,
    ):
        nonlocal _weight_check, _grad_check
        _shape_check(
            paddle_sublayer,
            torch_submodule,
            paddle_pname,
            torch_pname,
            paddle_param,
            torch_param,
            settings,
        )
        p_param = paddle_param.numpy()
        t_param = torch_param.detach().cpu().numpy()
        p_grad = paddle_param.grad.numpy() if paddle_param.grad is not None else None
        t_grad = torch_param.grad.detach().cpu().numpy() if torch_param.grad is not None else None
        if settings["transpose"]:
            t_param = numpy.transpose(t_param)
            if t_grad is not None:
                t_grad = numpy.transpose(t_grad)

        # check weight
        try:
            assert_tensor_equal(p_param, t_param, settings)
        except Exception as e:
            _weight_check = False
            info = (
                "=" * 25 + "\n" + "After training, weight value is different.\n"
                "between paddle: `{}`, torch: `{}` \n"
                "paddle path:\n    {}\n"
                "torch path:\n    {}\n"
                "{}\n\n".format(
                    paddle_sublayer,
                    torch_submodule,
                    paddle_sublayer.padiff_path + "." + paddle_pname,
                    torch_submodule.padiff_path + "." + torch_pname,
                    str(e),
                )
            )
            log_file("weight_diff.log", "a", info)

        # check grad
        try:
            if (p_grad is not None or t_grad is not None) and settings["diff_phase"] == "both":
                assert_tensor_equal(p_grad, t_grad, settings)
        except Exception as e:
            _grad_check = False
            info = (
                "=" * 25 + "\n" + "After training, grad value is different.\n"
                "between paddle: `{}`, torch: `{}` \n"
                "paddle path:\n    {}\n"
                "torch path:\n    {}\n"
                "{}\n\n".format(
                    paddle_sublayer,
                    torch_submodule,
                    paddle_sublayer.padiff_path + "." + paddle_pname,
                    torch_submodule.padiff_path + "." + torch_pname,
                    str(e),
                )
            )
            log_file("grad_diff.log", "a", info)

    process_each_weight(_check_weight_grad, layer, module, layer_map)

    if _weight_check == False:
        log(f"Diff found in model weights, check report `{diff_log_path + '/weight_diff.log'}`.")
    if _grad_check == False:
        log(f"Diff found in model grad, check report `{diff_log_path + '/grad_diff.log'}`.")

    if _weight_check and _grad_check:
        log("weight and grad compared.")

    return _weight_check, _grad_check


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
