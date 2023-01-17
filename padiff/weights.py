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
from itertools import zip_longest

import numpy
import paddle
import torch


from .utils import log, map_for_each_sublayer, compare_tensor, traversal_layers
from .yaml_loader import global_yaml_loader as yamls


def process_each_weight(process, layer, module, options, layer_mapping={}):
    """
    Apply process for each pair of parameters in layer(paddle) and module(torch)

    Args:
        process (function): process applied to parameters
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_mapping (dict, optional): manually map paddle layer and torch module.
        options (dict, optional):
            atol, rtol, compare_mode, single_step
    """

    def _process_runner(
        process,
        paddle_sublayer,
        torch_submodule,
        param_name,
        paddle_param,
        torch_param,
    ):
        try:
            settings = yamls.get_weight_settings(paddle_sublayer, torch_submodule, param_name)
        except Exception as e:
            p_model_log = os.path.join(sys.path[0], "diff_log", "paddle_model_struct.log")
            t_model_log = os.path.join(sys.path[0], "diff_log", "torch_model_struct.log")
            with open(p_model_log, "w") as log:
                log.write(str(layer))
            with open(t_model_log, "w") as log:
                log.write(str(module))
            raise e

        process(
            paddle_sublayer,
            torch_submodule,
            param_name,
            paddle_param,
            torch_param,
            settings,
        )

    layers = [layer]
    modules = [module]
    layers.extend(filter(lambda x: x not in layer_mapping.keys(), traversal_layers(layer, layer_mapping)))
    modules.extend(filter(lambda x: x not in layer_mapping.values(), traversal_layers(module, layer_mapping)))

    for paddle_sublayer, torch_submodule in zip_longest(layers, modules, fillvalue=None):
        if paddle_sublayer is None or torch_submodule is None:
            raise RuntimeError("Torch and Paddle return difference number of sublayers. Check your model.")
        for (name, paddle_param), torch_param in zip(
            paddle_sublayer.named_parameters(prefix="", include_sublayers=False),
            torch_submodule.parameters(recurse=False),
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


def assign_weight(layer, module, options, layer_mapping={}):
    """
    Init weights of layer(paddle) and module(torch) with same value

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_mapping (dict, optional): manually map paddle layer and torch module.
    """

    for paddle_sublayer, torch_submodule in layer_mapping.items():
        assign_config = yamls.assign_yaml.get(paddle_sublayer.__class__.__name__, None)
        if assign_config is None or assign_config.get("init", False) == False:
            log(
                "*** Auto weight paddle layer `{}` and torch module `{}` is not supported ***".format(
                    paddle_sublayer.__class__.__name__, torch_submodule.__class__.__name__
                )
            )
            log("*** Checkout the parameters are inited by yourself!!! ***")
        else:
            special_init(paddle_sublayer, torch_submodule)

    process_each_weight(_assign_weight, layer, module, options, layer_mapping)


def check_weight_grad(layer, module, options, layer_mapping={}):
    """
    Compare weights and grads between layer(paddle) and module(torch)

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_mapping (dict, optional): manually map paddle layer and torch module.
        options (dict, optional):
            atol, compare_mode
    """
    if options["diff_phase"] == "forward":
        log("Diff_phase is `forward`. Weight and grad check skipped.")
        return True, True

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
        p_grad = paddle_param.grad.numpy() if paddle_param.grad is not None else None
        t_grad = torch_param.grad.detach().numpy() if torch_param.grad is not None else None
        if settings["transpose"]:
            t_param = numpy.transpose(t_param)
            if t_grad is not None:
                t_grad = numpy.transpose(t_grad)

        weight_log_path = os.path.join(sys.path[0], "diff_log", "weight_diff.log")
        grad_log_path = os.path.join(sys.path[0], "diff_log", "grad_diff.log")

        _weight_check = compare_tensor(
            p_param,
            t_param,
            atol=settings["atol"],
            rtol=settings["rtol"],
            compare_mode=settings["compare_mode"],
        )
        _grad_check = compare_tensor(
            p_grad, t_grad, atol=settings["atol"], rtol=settings["rtol"], compare_mode=settings["compare_mode"]
        )

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

    process_each_weight(_check_weight_grad, layer, module, options, layer_mapping)

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
    """

    def _remove_inplace(layer, module):
        if hasattr(module, "inplace"):
            module.inplace = False

    map_for_each_sublayer(_remove_inplace, layer, module)


def special_init(paddle_layer, torch_module):
    def init_LSTM(layer, module):
        for (name, paddle_param), torch_param in zip(
            layer.named_parameters(prefix="", include_sublayers=False),
            module.parameters(recurse=False),
        ):
            settings = yamls.get_weight_settings(layer, module, name)
            _assign_weight(layer, module, name, paddle_param, torch_param, settings)

    def init_MultiHeadAttention(layer, module):
        name_param_dict = {}
        for i, param in enumerate(layer.named_parameters()):
            pname = param[0]
            if "cross_attn" in pname:
                pname = pname.replace("cross_attn", "multihead_attn")
            elif "q" not in pname and "k" not in pname and "v" not in pname:
                continue
            param_np = param[1].numpy()
            pname = pname.replace("q_proj.", "in_proj_")
            pname = pname.replace("k_proj.", "in_proj_")
            pname = pname.replace("v_proj.", "in_proj_")
            if pname not in name_param_dict:
                name_param_dict[pname] = param_np
            elif "_weight" in pname:
                name_param_dict[pname] = numpy.concatenate((name_param_dict[pname], param_np), axis=1)
            else:
                name_param_dict[pname] = numpy.concatenate((name_param_dict[pname], param_np), axis=0)

        device = torch.device("cuda:0")
        for i, param in enumerate(module.named_parameters()):
            pname, pa = param[0], param[1]
            if "in_proj" in pname or "multihead_attn" in pname:
                param_np = name_param_dict[pname]
            else:
                param_np = layer.state_dict()[pname].numpy()
            if pname.endswith("weight"):
                param_np = numpy.transpose(param_np)

            param[1].data = torch.from_numpy(param_np)

    special_init_tools = {
        "LSTM": init_LSTM,
        "MultiHeadAttention": init_MultiHeadAttention,
    }
    name = paddle_layer.__class__.__name__
    special_init_tools[name](paddle_layer, torch_module)
