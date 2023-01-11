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

import os
import sys
import os.path as osp

import numpy
import paddle
import torch
import yaml


from .utils import log, map_for_each_sublayer, compare_tensor, traversal_layers


def process_each_weight(process, layer, module, layer_map, options={}):
    """
    Apply process for each pair of parameters in layer(paddle) and module(torch)

    Args:
        process (function): process applied to parameters
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_map (dict, optional): manually map paddle layer and torch module.
        options (dict, optional):
            atol, compare_mode
    """
    yaml_path = osp.join(osp.dirname(__file__), "configs", "assign_weight.yaml")
    with open(yaml_path, "r") as yaml_file:
        assign_yaml = yaml.safe_load(yaml_file)

    yamls = {
        "assign_yaml": assign_yaml,
    }

    def _process_runner(
        process,
        paddle_sublayer,
        torch_submodule,
        param_name,
        paddle_param,
        torch_param,
    ):
        assign_config = yamls["assign_yaml"].get(paddle_sublayer.__class__.__name__, None)
        settings = {
            "atol": options.get("atol", 1e-7),
            "transpose": False,
            "compare_mode": options.get("compare_mode", "mean"),
        }

        if assign_config is not None:
            try:
                assert (
                    torch_submodule.__class__.__name__ in assign_config["torch"]
                ), "Not correspond, paddle layer {}  vs torch module {}. check your __init__ to make sure every sublayer is corresponded, or view the model struct reports in diff_log.".format(
                    paddle_sublayer.__class__.__name__, torch_submodule.__class__.__name__
                )
            except Exception as e:
                p_model_log = os.path.join(sys.path[0], "diff_log", "paddle_model_struct.log")
                t_model_log = os.path.join(sys.path[0], "diff_log", "torch_model_struct.log")
                with open(p_model_log, "w") as log:
                    log.write(str(layer))
                with open(t_model_log, "w") as log:
                    log.write(str(module))
                raise e
        if assign_config is None or param_name not in assign_config["param"]:
            pass
        else:
            if assign_config["param"][param_name] == "transpose":
                settings["transpose"] = True

        process(
            paddle_sublayer,
            torch_submodule,
            param_name,
            paddle_param,
            torch_param,
            settings,
        )

    paddle_layers = [layer]
    torch_modules = [module]

    traversal_layers(paddle_layers, layer, layer_map)
    traversal_layers(torch_modules, module, layer_map)

    assert len(paddle_layers) == len(
        torch_modules
    ), "Torch and Paddle return difference number of sublayers. Check your model."

    for paddle_sublayer, torch_submodule in zip(paddle_layers, torch_modules):

        for (name, paddle_param), torch_param in zip(
            paddle_sublayer.named_parameters("", False),
            torch_submodule.parameters(False),
        ):
            _process_runner(
                process,
                paddle_sublayer,
                torch_submodule,
                name,
                paddle_param,
                torch_param,
            )


def _shape_check(
    paddle_sublayer,
    torch_submodule,
    param_name,
    paddle_param,
    torch_param,
    settings,
):
    p_shape = list(paddle_param.shape)
    t_shape = list(torch_param.shape)
    if settings["transpose"]:
        t_shape.reverse()
    assert p_shape == t_shape, (
        "Shape of param `{}` is not the same. {} vs {}\n"
        "Hint: \n"
        "      1. check whether your paddle model definition and torch model definition are corresponding.\n"
        "      2. check the weight shape of paddle:`{}` and torch:`{}` is the same.\n"
    ).format(param_name, p_shape, t_shape, paddle_sublayer, torch_submodule)


def assign_weight(layer, module, layer_map=None):
    """
    Init weights of layer(paddle) and module(torch) with same value

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_map (dict, optional): manually map paddle layer and torch module.
    """

    def _assign_weight(
        paddle_sublayer,
        torch_submodule,
        param_name,
        paddle_param,
        torch_param,
        settings,
    ):
        _shape_check(
            paddle_sublayer,
            torch_submodule,
            param_name,
            paddle_param,
            torch_param,
            settings,
        )
        np_value = paddle.randn(paddle_param.shape).numpy()
        paddle.assign(paddle.to_tensor(np_value), paddle_param)
        if settings["transpose"]:
            torch_param.data = torch.as_tensor(numpy.transpose(np_value)).type(torch_param.dtype)
        else:
            torch_param.data = torch.as_tensor(np_value).type(torch_param.dtype)

    process_each_weight(_assign_weight, layer, module, layer_map)


def check_weight_grad(layer, module, layer_map=None, options={}):
    """
    Compare weights and grads between layer(paddle) and module(torch)

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_map (dict, optional): manually map paddle layer and torch module.
        options (dict, optional):
            atol, compare_mode
    """
    _weight_check = True
    _grad_check = True

    def _check_weight_grad(
        paddle_sublayer,
        torch_submodule,
        param_name,
        paddle_param,
        torch_param,
        settings,
    ):
        nonlocal _weight_check, _grad_check
        _shape_check(
            paddle_sublayer,
            torch_submodule,
            param_name,
            paddle_param,
            torch_param,
            settings,
        )
        p_param = paddle_param.numpy()
        t_param = torch_param.detach().numpy()
        p_grad = paddle_param.grad.numpy()
        t_grad = torch_param.grad.detach().numpy()
        if settings["transpose"]:
            t_param = numpy.transpose(t_param)
            t_grad = numpy.transpose(t_grad)

        weight_log_path = os.path.join(sys.path[0], "diff_log", "weight_diff.log")
        grad_log_path = os.path.join(sys.path[0], "diff_log", "grad_diff.log")

        _weight_check = compare_tensor(
            p_param,
            t_param,
            atol=settings["atol"],
            compare_mode=settings["compare_mode"],
        )
        _grad_check = compare_tensor(p_grad, t_grad, atol=settings["atol"], compare_mode=settings["compare_mode"])

        if _weight_check is False:
            with open(weight_log_path, "a") as f:
                f.write(
                    "After training, weight value is different for param `{}`.\n"
                    "paddle: `{}` with value:\n{}\n"
                    "torch: `{}` with value:\n{}\n\n".format(
                        param_name, paddle_sublayer, p_param, torch_submodule, t_param
                    )
                )

        if _grad_check is False:
            with open(grad_log_path, "a") as f:
                f.write(
                    "After training, grad value is different for param `{}`.\n"
                    "paddle: `{}` with value\n{}\n"
                    "torch: `{}` with value\n{}\n\n".format(
                        param_name, paddle_sublayer, p_grad, torch_submodule, t_grad
                    )
                )

    process_each_weight(_check_weight_grad, layer, module, layer_map, options)

    if _weight_check and _grad_check:
        log("weight and weight.grad is compared.")
    else:
        diff_log_path = os.path.join(sys.path[0], "diff_log")
        log("Differences in weight or grad !!!")
        log("Check reports at `{}`\n".format(diff_log_path))

    return _weight_check, _grad_check


def remove_inplace(layer, module):
    """
    Set `inplace` tag to `False` for torch module

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        options (dict, optional):
            atol, compare_mode
    """

    def _remove_inplace(layer, module):
        if hasattr(module, "inplace"):
            module.inplace = False

    map_for_each_sublayer(_remove_inplace, layer, module)
