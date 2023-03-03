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


from .utils import log, map_for_each_sublayer, assert_tensor_equal, LayerMap, log_file, diff_log_path
from .file_loader import global_yaml_loader as yamls


def weight_struct_info(layer, module, paddle_sublayer, torch_submodule):
    t_title = "Torch Model\n" + "=" * 25 + "\n"
    t_retval = print_weight_struct(module, mark=torch_submodule, prefix=[" " * 4])
    t_info = t_title + "\n".join(t_retval)

    p_title = "Paddle Model\n" + "=" * 25 + "\n"
    p_retval = print_weight_struct(layer, mark=paddle_sublayer, prefix=[" " * 4])
    p_info = p_title + "\n".join(p_retval)

    if len(p_retval) + len(t_retval) > 100:
        log_file("paddle_weight_check.log", "w", p_info)
        log_file("torch_weight_check.log", "w", t_info)
        log(
            f"Model Struct saved to `{diff_log_path + '/torch_weight_check.log'}` and `{diff_log_path + '/paddle_weight_check.log'}`."
        )
        log("Please view the reports and checkout the layers which is marked with `<---  *** HERE ***` !")
    else:
        log("Print model Struct while checking model weights:")
        print(t_info)
        print(p_info)

    print("\nHint:")
    print("      1. check the init order of param or layer in definition is the same.")
    print(
        "      2. try to use `LayerMap` to skip the diff in models, you can find the instructions at `https://github.com/PaddlePaddle/PaDiff`."
    )


def print_weight_struct(net, mark=None, prefix=[]):
    cur_str = ""
    for i, s in enumerate(prefix):
        if i == len(prefix) - 1:
            cur_str += s
        else:
            if s == " |--- ":
                cur_str += " |    "
            elif s == " +--- ":
                cur_str += "      "
            else:
                cur_str += s

    cur_str += str(net.__class__.__name__)
    if mark is net:
        cur_str += "    <---  *** HERE ***"

    ret_strs = [cur_str]

    children = list(net.children())
    for i, child in enumerate(children):
        pre = " |--- "
        if i == len(children) - 1:
            pre = " +--- "
        prefix.append(pre)
        retval = print_weight_struct(child, mark, prefix)
        ret_strs.extend(retval)
        prefix.pop()

    return ret_strs


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
        param_name,
        paddle_param,
        torch_param,
    ):

        settings = yamls.get_weight_settings(paddle_sublayer, torch_submodule, param_name)

        process(
            paddle_sublayer,
            torch_submodule,
            param_name,
            paddle_param,
            torch_param,
            settings,
        )

    layers = layer_map.weight_init_layers(layer)
    modules = layer_map.weight_init_layers(module)

    for paddle_sublayer, torch_submodule in zip_longest(layers, modules, fillvalue=None):
        if paddle_sublayer is None or torch_submodule is None:
            raise RuntimeError("Torch and Paddle return difference number of sublayers. Check your model.")
        for (name, paddle_param), torch_param in zip(
            paddle_sublayer.named_parameters(prefix="", include_sublayers=False),
            torch_submodule.parameters(recurse=False),
        ):
            try:
                _process_runner(
                    process,
                    paddle_sublayer,
                    torch_submodule,
                    name,
                    paddle_param,
                    torch_param,
                )
            except Exception as e:
                log(f"Error occurred while process parameter `{name}`!")
                log("Error Msg:\n")
                print(f"{str(e)}")
                weight_struct_info(layer, module, paddle_sublayer, torch_submodule)
                raise e


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

    for torch_submodule, paddle_sublayer in layer_map.special_init_layers():
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
    try:
        process_each_weight(_assign_weight, layer, module, layer_map)
        log("Assign weight success !!!")
        return True
    except:
        log("Assign weight Failed !!!")
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
                "=" * 25 + "\n" + "After training, weight value is different for param `{}`.\n"
                "between paddle: `{}`, torch: `{}` \n"
                "{}\n\n".format(
                    param_name, paddle_sublayer.__class__.__name__, torch_submodule.__class__.__name__, str(e)
                )
            )
            log_file("weight_diff.log", "a", info)

        # check grad
        try:
            assert_tensor_equal(p_grad, t_grad, settings)
        except Exception as e:
            _grad_check = False
            info = (
                "=" * 25 + "\n" + "After training, grad value is different for param `{}`.\n"
                "between paddle: `{}`, torch: `{}` \n"
                "{}\n\n".format(
                    param_name, paddle_sublayer.__class__.__name__, torch_submodule.__class__.__name__, str(e)
                )
            )
            log_file("grad_diff.log", "a", info)

    process_each_weight(_check_weight_grad, layer, module, layer_map)

    print("")
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


def special_init(paddle_layer, torch_module):
    # NOTICE: make sure torch params is in the same device after init

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

        for i, param in enumerate(module.named_parameters()):
            pname, pa = param[0], param[1]
            if "in_proj" in pname or "multihead_attn" in pname:
                param_np = name_param_dict[pname]
            else:
                param_np = layer.state_dict()[pname].numpy()
            if pname.endswith("weight"):
                param_np = numpy.transpose(param_np)
            device = param[1].device
            param[1].data = torch.from_numpy(param_np).to(device)

    special_init_tools = {
        "LSTM": init_LSTM,
        "MultiHeadAttention": init_MultiHeadAttention,
    }
    name = paddle_layer.__class__.__name__
    special_init_tools[name](paddle_layer, torch_module)
