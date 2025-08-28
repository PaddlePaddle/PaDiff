# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import contextlib
from functools import partial

import numpy as np
import paddle
import torch

from ...utils import (
    clone_tensors,
    extract_frame_summary,
    flatten,
    for_each_grad_tensor,
    logger,
    to_sequence,
    map_structure,
)
from .base import current_report, find_base_report_node, single_step_state


@contextlib.contextmanager
def register_hooker(model):
    marker = model.marker

    remove_handles = []
    idx = 0
    # traversal_for_hook includes layers which we need add pre and post hook
    # for model structure, but not info hook (they are in black list)
    models = list(marker.traversal_for_hook())

    # register model-level hooks
    handle_init = model.register_forward_pre_hook(partial(init_weights_hook))
    handle_input = model.register_forward_pre_hook(partial(first_input_hook))
    remove_handles.extend([handle_init, handle_input])

    # register layer-level hooks
    for mod in models:
        pre_handle = mod.register_forward_pre_hook(partial(pre_structure_hook))
        if mod.model not in marker.black_list:
            logger.debug(f"Register(info_hook): {mod.model.__class__.__name__}(net_id={idx})")
            handle = mod.register_forward_post_hook(partial(info_hook, net_id=idx))
            remove_handles.append(handle)
            idx += 1
        else:
            logger.debug(f"Skip(info_hook): {mod.model.__class__.__name__}(blacklisted)")
        post_handle = mod.register_forward_post_hook(partial(post_structure_hook))
        remove_handles.extend([pre_handle, post_handle])
    yield
    for h in remove_handles:
        h.remove()


"""
    hooks for catching data
"""


def init_weights_hook(model, input):
    report = current_report()
    if report is not None and not hasattr(report, "init_weights_saved"):
        init_weights = {}
        for name, param in model.named_parameters():
            if isinstance(param, (paddle.Tensor, torch.Tensor)):
                if param.dtype == torch.bfloat16:
                    np_array = param.detach().cpu().float().numpy()
                elif param.dtype == paddle.bfloat16:
                    np_array = param.detach().cpu().astype("float32").numpy()
                else:
                    np_array = param.detach().cpu().numpy()
                init_weights[name] = np_array
        report.init_weights = init_weights
        report.init_weights_saved = True
    return None


def first_input_hook(model, input):
    report = current_report()
    if report is None:
        logger.debug("first_input_hook: current_report is None")
        return None

    if hasattr(report, "_loaded_inputs") and report._loaded_inputs is not None:
        logger.debug("first_input_hook: loading first input")
        loaded_inputs = report._loaded_inputs
        if isinstance(loaded_inputs, list):
            return tuple(loaded_inputs)
        return loaded_inputs

    if not hasattr(report, "first_input_captured"):
        logger.debug("first_input_hook: capturing first input")

        def serialize(x):
            if isinstance(x, (paddle.Tensor, torch.Tensor)):
                return ("Tensor", x.detach().cpu().numpy())
            elif isinstance(x, dict):
                return ("dict", {k: serialize(v) for k, v in x.items()})
            elif isinstance(x, (list, tuple)):
                return (type(x).__name__, [serialize(item) for item in x])
            else:
                return (type(x).__name__, str(x))

        try:
            serialized = [serialize(x) for x in to_sequence(input)]
            if len(serialized) == 0:
                logger.warning(f"No first input captured")
            report.first_input = serialized
            report.first_input_captured = True
        except Exception as e:
            logger.warning(f"Failed to capture first input: {e}")
    return None


"""
    hooks used to build module structure
"""


def pre_structure_hook(layer, input):
    report = current_report()
    report.stack.push_layer(layer)
    if layer in report.marker.layer_map:
        report.stack._top().layer_type = "in map"
    return None


def post_structure_hook(layer, input, output):
    report = current_report()
    retval = report.stack.pop_layer(layer)
    if retval.net in report.marker.black_list:
        report.stack._top().children.pop()
    return None


"""
    hook for record forward infos
"""

# do not enter api layer which is triggered under info_hook
__in_info_hook__ = False


def info_hook(model, input, output, net_id):
    """
    Notice: the model is a origin layer/module, not ProxyModel
    """
    global __in_info_hook__
    if __in_info_hook__:
        return None

    report = current_report()

    if report is None or report.stack._top() is None:
        return None

    # if this api is not processing tensors, do not create report
    if output is None or all([not isinstance(x, (paddle.Tensor, torch.Tensor)) for x in flatten(output)]):
        return None

    # if an api under black_list_recursively, do not create report
    # a layer under black_list_recursively will not register this hook, except it is a mapped layer
    # report.stack._top().net can not be an api layer !!!
    if report.stack._top().net in report.marker.black_list_recursively and hasattr(model, "__api__"):
        return None

    # if this api is called under layer/module provided by framework, skip it
    python_module = report.stack._top().net.__module__
    if hasattr(model, "__api__") and (python_module.startswith("paddle.") or python_module.startswith("torch.")):
        return None

    __in_info_hook__ = True

    # if current model is an api layer, we do not want to hold it
    if hasattr(model, "__api__"):
        _model = padiff_layer_str(model)
    else:
        _model = model

    _, frames = extract_frame_summary()

    new_in = clone_tensors(input)
    new_out = clone_tensors(output)
    fwd_item = report.put_item("forward", new_in, new_out, _model, net_id, frames)
    bwd_item = report.put_item("backward", new_in, new_out, _model, net_id, frames)
    bwd_item.set_forward(fwd_item)

    report.stack.push_api(_model, fwd_item, bwd_item)

    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i, net_id=net_id))

    # if under single step forward guard
    if single_step_state() == "forward" and net_id != -1:
        # two report_item with same id, the step_idx should be corresponded
        step_idx = len(list(filter(lambda x: x.type == "forward" and x.net_id == net_id, report.items))) - 1

        try:
            base_report_node = find_base_report_node(net_id, step_idx)
        except (IndexError, RuntimeError) as e:
            error_msg = str(e)
            base_max_calls = "unknown"
            if "list length=" in error_msg:
                try:
                    base_max_calls = int(error_msg.split("list length=")[1].split()[0])
                except:
                    pass
            current_calls = step_idx + 1
            route = getattr(model, "route", "unknown")
            logger.error(
                f"\n   ❌ Single-step alignment FAILED: Execution path mismatch!"
                f"\n   📌 Layer '{route}' called {current_calls} times (current) vs {base_max_calls} times (base)."
                f"\n   📌 Check the forward logic in both models around this layer."
            )
            sys.exit(1)

        if base_report_node["name"] != _model.__class__.__name__:
            warning_msg = (
                f"\n   ⚠️ Single-step alignment FAILED: Layer with net_id={net_id} mismatch!"
                f"\n   📌 Mismatch Layer: {base_report_node['name']}(base) vs {_model.__class__.__name__}(raw)"
                f"\n   💡 Suggestion: Models have different architectures or initialization order. "
                "Please check the model implementation or decrease 'align_depth' to reduce the alignment "
                "granularity, or add layers that do not require alignment to the blacklist."
            )
            logger.warning(warning_msg)
        else:
            logger.debug(f"Single Step: {_model.__class__.__name__}(net_id={net_id})")

        retval = map_structure(replace_forward_output(base_report_node), output)
        __in_info_hook__ = False
        return retval
    else:
        __in_info_hook__ = False
        return None


"""
    hook for record backward infos
"""


def tensor_hook(x_grad, bwd_item, nth_tensor, net_id):
    new_grad = clone_tensors(x_grad)
    bwd_item.set_input_grads(nth_tensor, new_grad[0])

    if single_step_state() == "backward" and net_id != -1:
        report = current_report()
        step_idx = (
            list(filter(lambda x: x.type == "backward" and x.net_id == net_id, report.items)).index(bwd_item) - 1
        )
        base_report_node = find_base_report_node(net_id, step_idx)

        value = np.load(base_report_node["bwd_grads"][nth_tensor]["path"])
        if isinstance(x_grad, paddle.Tensor):
            return paddle.to_tensor(value)
        else:
            return torch.as_tensor(value, device=x_grad.device)

    return x_grad


"""
    utils
"""


def padiff_layer_str(model):
    if isinstance(model, paddle.nn.Layer):
        return PaddleLayerStr(model)
    else:
        return TorchModuleStr(model)


class PaddleLayerStr(paddle.nn.Layer):
    def __init__(self, net):
        super().__init__()
        self.__name__ = net.__name__
        self.__api__ = net.__api__


class TorchModuleStr(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.__name__ = net.__name__
        self.__api__ = net.__api__


def replace_forward_output(node):
    numpy_file_list = node["fwd_outputs"]
    cur_idx = 0

    def inner(input_):
        if isinstance(input_, (paddle.Tensor, torch.Tensor)):
            if cur_idx >= len(numpy_file_list):
                raise RuntimeError(
                    "In single step mode, try to replace tensor by dumpped numpy value, but the number of tensors and numpy is not equal. Maybe the models are not corresponded."
                )
            value = np.load(numpy_file_list[cur_idx]["path"])
            if isinstance(input_, paddle.Tensor):
                return paddle.to_tensor(value, dtype=input_.dtype)
            else:
                return torch.as_tensor(value, dtype=input_.dtype, device=input_.device)
        else:
            return input_

    return inner
