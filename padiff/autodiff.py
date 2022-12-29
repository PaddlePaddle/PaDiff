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

import numpy
import paddle
import torch

from .report import (Report, check_forward_and_backward, current_report,
                     report_guard)
from .stack_info import *
from .utils import (build_log_dir, clean_log_dir, for_each_grad_tensor,
                    torch_mean)
from .weights import assign_weight, check_weight_grad, remove_inplace


def autodiff(layer, module, example_inp, auto_weights=True, options={}):
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        layer (paddle.nn.Layer):
        module (torch.nn.Module):
        example_inp (numpy.array):
        auto_weights (boolean, optional):
        options (dict, optional):
            atol
    Returns:
        True for success, False for failed.
    """
    assert isinstance(layer, paddle.nn.Layer), "Invalid Argument."
    assert isinstance(module, torch.nn.Module), "Invalid Argument."
    assert isinstance(example_inp, numpy.ndarray), "Invalid Argument."

    paddle.set_device("cpu")
    module = module.to("cpu")

    build_log_dir()

    _preprocess(layer, module, example_inp, auto_weights, options)

    torch_report = Report("torch")
    paddle_report = Report("paddle")
    with report_guard(torch_report):
        with _register_torch_hooker(module):
            try:
                torch_input = torch.as_tensor(example_inp)
                torch_input.requires_grad = True
                torch_output = module(torch_input)
                loss = torch_mean(torch_output)
                loss.backward()
            except Exception as e:
                raise RuntimeError(
                    "Exception is thrown while running forward of torch_module, please check the legality of module.\n{}".format(
                        str(e)
                    )
                )

    with report_guard(paddle_report):
        with _register_paddle_hooker(layer):
            try:
                paddle_input = paddle.to_tensor(example_inp)
                paddle_input.stop_gradient = False
                paddle_output = layer(paddle_input)
                loss = paddle.mean(paddle_output)
                loss.backward()
            except Exception as e:
                raise RuntimeError(
                    "Exception is thrown while running forward of paddle_layer, please check the legality of layer.\n{}".format(
                        str(e)
                    )
                )

    print(
        "Max output diff is {}\n".format(
            numpy.abs(paddle_output.numpy() - torch_output.detach().numpy()).max()
        )
    )

    weight_check, grad_check = check_weight_grad(layer, module, options)
    ret = check_forward_and_backward(torch_report, paddle_report, options)
    ret = ret and weight_check and grad_check

    clean_log_dir()
    return ret


def tensor_hook(x_grad, bwd_item, nth_tensor):
    # print (nth_tensor, bwd_item.input_grads, bwd_item.input)
    bwd_item.set_input_grads(nth_tensor, x_grad)
    return x_grad


def layer_hook(module, input, output, idx):
    rep = current_report()
    frame_info, frames = extract_frame_summary()
    fwd_item = rep.put_item("forward", input, output, module, idx, frame_info, frames)
    bwd_item = rep.put_item("backward", input, output, module, idx, frame_info, frames)
    bwd_item.set_forward(fwd_item)
    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i))
    return None


@contextlib.contextmanager
def _register_paddle_hooker(layer):
    remove_handles = []
    # TODO(xiongkun): duplicate layer is not support, implement custom generator to support (different net_id is ok).
    for idx, mod in enumerate(layer.sublayers(True)):
        handle = mod.register_forward_post_hook(partial(layer_hook, idx=idx))
        remove_handles.append(handle)
    yield
    for h in remove_handles:
        h.remove()


@contextlib.contextmanager
def _register_torch_hooker(module):
    remove_handles = []
    for idx, mod in enumerate(module.modules()):
        handle = mod.register_forward_hook(partial(layer_hook, idx=idx))
        remove_handles.append(handle)
    yield
    for h in remove_handles:
        h.remove()


def _preprocess(layer, module, example_inp, auto_weights, options):
    remove_inplace(layer, module)
    if auto_weights:
        assign_weight(layer, module)
