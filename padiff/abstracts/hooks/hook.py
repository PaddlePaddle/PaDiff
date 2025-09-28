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
    map_structure,
    get_numpy_from_tensor,
    is_require_grad,
)
from .base import current_report, find_base_report_node, single_step_state


_seen_warnings = set()


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
    remove_handles.append(handle_init)

    param_handles = creat_param_handles(model)
    remove_handles.extend(param_handles)

    # register layer-level hooks
    for mod in models:
        pre_handle = mod.register_forward_pre_hook(partial(pre_structure_hook))
        if mod.model not in marker.black_list:
            logger.debug(f"Register(info_hook): {mod.class_name}(net_id={idx})")
            handle = mod.register_forward_post_hook(partial(info_hook, net_id=idx))
            remove_handles.append(handle)
            idx += 1
        else:
            logger.debug(f"Skip(info_hook): {mod.class_name}(blacklisted)")
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
                np_array = get_numpy_from_tensor(param)
                init_weights[name] = np_array
                logger.debug(f"Register(init_weights_hook): '{name}'(param)")

        for name, buffer in model.named_buffers():
            if isinstance(buffer, (paddle.Tensor, torch.Tensor)) and name not in init_weights:
                try:
                    np_array = get_numpy_from_tensor(buffer)
                    init_weights[name] = np_array
                    logger.debug(f"Register(init_weights_hook): '{name}'(buffer)")
                except Exception as e:
                    logger.warning(f"Skip(init_weights_hook): '{name}'(unitialized buffer): {e}")

        report.init_weights = init_weights
        report.init_weights_saved = True
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
    if single_step_state() in ("forward", "both") and net_id != -1:
        # two report_item with same id, the step_idx should be corresponded
        model_name = _model.__class__.__name__
        step_idx = len(list(filter(lambda x: x.type == "forward" and x.net_id == net_id, report.items))) - 1
        base_report_node = single_step_check(report, net_id, step_idx, model_name, "forward")
        retval = map_structure(replace_forward_output(base_report_node, model_name), output)
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

    if single_step_state() in ("backward", "both") and net_id != -1:
        report = current_report()
        step_idx = (
            list(filter(lambda x: x.type == "backward" and x.net_id == net_id, report.items)).index(bwd_item) - 1
        )
        base_report_node = single_step_check(
            report, net_id, step_idx, bwd_item.net.__class__.__name__, "backward", bwd_item=bwd_item
        )

        value = np.load(base_report_node["bwd_grads"][nth_tensor]["path"])
        if isinstance(x_grad, paddle.Tensor):
            return paddle.to_tensor(value)
        else:
            return torch.as_tensor(value, device=x_grad.device)

    return x_grad


def creat_param_handles(model):
    handles = []

    for name, proxy_param in model.named_parameters(recursively=True):
        if is_require_grad(proxy_param.param):

            def make_hook(param, param_name):
                def hook(grad):
                    logger.debug(f"Grad hook triggered for {param_name}")
                    param._collected_grad = grad
                    return grad

                return hook

            handle = proxy_param.param.register_hook(make_hook(proxy_param.param, name))
            handles.append(handle)

    return handles


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


def replace_forward_output(node, current_name=None):
    numpy_file_list = node["fwd_outputs"]
    cur_idx = 0

    def inner(input_):
        nonlocal cur_idx
        if not isinstance(input_, (paddle.Tensor, torch.Tensor)):
            return input_

        if cur_idx >= len(numpy_file_list):
            warning_key = ("single-step: output_count_mismatch", current_name)
            if warning_key not in _seen_warnings:
                logger.warning(
                    f"\n   ⚠️ Single-step alignment SKIPPED: the {cur_idx + 1}st output is requested, "
                    f"but only {len(numpy_file_list)} pre-saved from base model, skip the current output."
                    "\n   ⚠️ This warning will not repeat for this layer."
                    f"\n   📌 Layer Name: {current_name}(raw)"
                    "\n   💡 Possible Causes and Solutions:"
                    "\n     - The number of outputs from the current layer in the raw model is bigger than"
                    "that of its corresponding layer in the base model."
                    "\n     - Verify that both models have identical architectures for this layer."
                    "\n     - If the corresponding relationship of the current layer is correct, "
                    "please disable single step mode, or add the layer to blacklist to skip the check of this layer."
                    "\n     - Or when you are sure that the extra output does not need to be compared, "
                    "you can swap the execution order of the base model and the raw model."
                )
                _seen_warnings.add(warning_key)
            return input_

        value = np.load(numpy_file_list[cur_idx]["path"])
        cur_idx += 1
        base_shape = tuple(value.shape)
        raw_shape = tuple(input_.shape)

        if base_shape == raw_shape:
            pass

        elif np.prod(base_shape) != np.prod(raw_shape):
            warning_key = ("single-step: shape_mismatch", current_name)
            if warning_key not in _seen_warnings:
                logger.warning(
                    f"\n   ⚠️ Single-step alignment SKIPPED: shape mismatch."
                    "\n   ⚠️ This warning will not repeat for this layer."
                    f"\n   📌 Layer Name: {current_name}(raw)"
                    f"\n   📌 Shape: {base_shape}(base) vs {raw_shape}(raw)"
                )
                _seen_warnings.add(warning_key)
            return input_
        else:
            value = value.reshape(input_.shape)
            debug_key = ("single-step: reshape_used", current_name)
            if debug_key not in _seen_warnings:
                logger.debug(
                    f"\n   ⚠️ Try to reshape loaded value to input's shape of layer {current_name}(raw). "
                    "This may lead to numerical errors even if reshape succeeds."
                    "\n   ⚠️ This warning will not repeat for this layer."
                )
                _seen_warnings.add(debug_key)

        if isinstance(input_, paddle.Tensor):
            return paddle.to_tensor(value, dtype=input_.dtype)
        else:
            return torch.as_tensor(value, dtype=input_.dtype, device=input_.device)

    return inner


def single_step_check(report, net_id, step_idx, current_name, node_type, bwd_item=None):

    try:
        base_report_node = find_base_report_node(net_id, step_idx)
        if base_report_node["name"] != current_name:
            warning_key = ("single-step: name_mismatch", current_name)
            if warning_key not in _seen_warnings:
                logger.warning(
                    f"\n   ⚠️ Single-step alignment WARNING: {node_type} with net_id={net_id} mismatch!"
                    "\n   ⚠️ This warning will not repeat for this layer."
                    f"\n   📌 Mismatch {node_type.capitalize()}: {base_report_node['name']}(base) vs {current_name}(raw)"
                    "\n   💡 Suggestion: Models have different architectures or class name or initialization order. "
                    "Please check the model implementation or decrease 'align_depth' to reduce the alignment "
                    "granularity, or add layers that do not require alignment to the blacklist."
                )
                _seen_warnings.add(warning_key)
        else:
            logger.debug(f"Single Step: {current_name}(net_id={net_id})")

        return base_report_node

    except (IndexError, RuntimeError) as e:
        error_msg = str(e)
        base_max_calls = "unknown"
        if "list length=" in error_msg:
            try:
                base_max_calls = int(error_msg.split("list length=")[1].split()[0])
            except:
                pass
        current_calls = step_idx + 1
        route = "unknown"
        if bwd_item and hasattr(bwd_item.net, "route"):
            route = bwd_item.net.route
        elif hasattr(report.stack._top().net, "route"):
            route = report.stack._top().net.route

        logger.error(
            f"\n   ❌ Single-step alignment FAILED: {node_type} with net_id={net_id} mismatch!"
            f"\n   📌 Layer '{route}' called {current_calls} times (current) vs {base_max_calls} times (base)."
            f"\n   📌 This indicates that the layer might be called but the call is not needed."
            f"\n   📌 Please check the {node_type} logic in both models around this layer."
        )

    return None
