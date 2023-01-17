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

import contextlib
from functools import partial

import paddle
import torch

from .report import Report, check_forward_and_backward, current_torch_report, current_paddle_report, report_guard
from .stack_info import *
from .utils import (
    for_each_grad_tensor,
    log,
    max_diff,
    reset_log_dir,
    tensors_mean,
    traversal_layers,
    map_structure_and_replace_key,
    init_options,
    modify_layer_mapping,
)
from .weights import assign_weight, check_weight_grad, remove_inplace
from .yaml_loader import global_yaml_loader as yamls
from .cmd import PaDiff_Cmd

paddle.set_printoptions(precision=10)
torch.set_printoptions(precision=10)


def auto_diff(layer, module, example_inp, auto_weights=True, options={}, layer_mapping={}):
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        layer (paddle.nn.Layer): paddle layer that needs compare
        module (torch.nn.Module): torch module that needs compare
        example_inp (paddle_input, torch_input): input data for paddle layer and torch module.
            paddle_input and torch_input should be dict and send into net like `module(**input)`.
        auto_weights (boolean, optional): uniformly init the parameters of models
        options (dict, optional):
            atol, compare_mode
        layer_mapping (dict, optional): manually map paddle layer and torch module.
    Returns:
        True for success, False for failed.
    """
    assert isinstance(layer, paddle.nn.Layer), "Invalid Argument."
    assert isinstance(module, torch.nn.Module), "Invalid Argument."
    assert isinstance(example_inp, (tuple, list)), "Invalid Argument."
    log("Start auto_diff, may need a while to generate reports...")

    paddle_input, torch_input = example_inp
    assert isinstance(paddle_input, dict), "Invalid Argument."
    assert isinstance(torch_input, dict), "Invalid Argument."
    paddle.set_device("cpu")
    module = module.to("cpu")

    reset_log_dir()
    _preprocess(layer, module, auto_weights, options, layer_mapping)

    torch_report = Report("torch")
    paddle_report = Report("paddle")
    with report_guard(torch_report, paddle_report):
        with _register_torch_hooker(module, options, layer_mapping):
            try:
                torch_output = module(**torch_input)
                loss = tensors_mean(torch_output, "torch")
                if options["diff_phase"] == "both":
                    loss.backward()
            except Exception as e:
                raise RuntimeError(
                    "Exception is thrown while running forward of torch_module, please check the legality of module.\n{}".format(
                        str(e)
                    )
                )

        with _register_paddle_hooker(layer, options, layer_mapping):
            try:
                paddle_output = layer(**paddle_input)
                loss = tensors_mean(paddle_output, "paddle")
                if options["diff_phase"] == "both":
                    loss.backward()
            except Exception as e:
                raise RuntimeError(
                    "Exception is thrown while running forward of paddle_layer, please check the legality of layer.\n{}".format(
                        str(e)
                    )
                )

    log("Max elementwise output diff is {}\n".format(max_diff(paddle_output, torch_output)))

    weight_check, grad_check = check_weight_grad(layer, module, layer_mapping=layer_mapping, options=options)
    ret = check_forward_and_backward(torch_report, paddle_report, options)
    ret = ret and weight_check and grad_check

    if options["cmd"]:
        PaDiff_Cmd(paddle_report, torch_report, options).cmdloop()

    # TODO(linjieccc): pytest failed if log clean is enabled
    # clean_log_dir()
    return ret


def tensor_hook(x_grad, bwd_item, nth_tensor):
    # print (nth_tensor, bwd_item.input_grads, bwd_item.input)
    bwd_item.set_input_grads(nth_tensor, x_grad)
    return x_grad


def torch_layer_hook(module, input, output, idx, options):
    rep = current_torch_report()
    frame_info, frames = extract_frame_summary()
    fwd_item = rep.put_item("forward", input, output, module, idx, frame_info, frames)
    bwd_item = rep.put_item("backward", input, output, module, idx, frame_info, frames)
    bwd_item.set_forward(fwd_item)
    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i))
    return None


def paddle_layer_hook(module, input, output, idx, options):
    p_rep = current_paddle_report()
    frame_info, frames = extract_frame_summary()
    fwd_item = p_rep.put_item("forward", input, output, module, idx, frame_info, frames)
    bwd_item = p_rep.put_item("backward", input, output, module, idx, frame_info, frames)
    bwd_item.set_forward(fwd_item)
    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i))

    if options["single_step"]:
        t_rep = current_torch_report()
        t_fwd_item = t_rep.find_item(p_rep, idx)

        def tt2pt(tt):
            if isinstance(tt, torch.Tensor):
                return paddle.to_tensor(tt.detach().numpy())
            else:
                return tt

        return map_structure_and_replace_key(tt2pt, [t_fwd_item.output], output)
    else:
        return None


@contextlib.contextmanager
def _register_paddle_hooker(layer, options, layer_mapping={}):
    remove_handles = []
    # TODO(xiongkun): duplicate layer is not support, implement custom generator to support (different net_id is ok).
    idx = 0
    layers = [layer]
    layers.extend(traversal_layers(layer, layer_mapping))
    for mod in layers:
        handle = mod.register_forward_post_hook(partial(paddle_layer_hook, idx=idx, options=options))
        remove_handles.append(handle)
        idx += 1
    yield
    for h in remove_handles:
        h.remove()


@contextlib.contextmanager
def _register_torch_hooker(module, options, layer_mapping={}):
    remove_handles = []
    idx = 0
    modules = [module]
    modules.extend(traversal_layers(module, layer_mapping))
    for mod in modules:
        handle = mod.register_forward_hook(partial(torch_layer_hook, idx=idx, options=options))
        remove_handles.append(handle)
        idx += 1
    yield
    for h in remove_handles:
        h.remove()


def _preprocess(layer, module, auto_weights, options, layer_mapping):
    init_options(options)
    modify_layer_mapping(layer_mapping)
    remove_inplace(layer, module)
    yamls.options = options
    if auto_weights:
        assign_weight(layer, module, options=options, layer_mapping=layer_mapping)
