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

import contextlib
from functools import partial
from .report import current_torch_report, current_paddle_report
from .. import utils
import os
import paddle
import torch


"""
    torch_tensor_hook, paddle_tensor_hook are used to create backward report items
"""


def torch_tensor_hook(x_grad, bwd_item, nth_tensor):
    # print (nth_tensor, bwd_item.input_grads, bwd_item.input)
    bwd_item.set_input_grads(nth_tensor, x_grad)
    return x_grad


def paddle_tensor_hook(x_grad, bwd_item, nth_tensor, net_id):
    # print (nth_tensor, bwd_item.input_grads, bwd_item.input)
    # record paddle grad
    bwd_item.set_input_grads(nth_tensor, x_grad)

    options = utils.yamls.options
    # single_step and not an API
    if net_id != -1 and options["single_step"] and options["diff_phase"] == "backward":
        paddle_report = current_paddle_report()
        torch_report = current_torch_report()
        t_fwd_item = torch_report.find_item(paddle_report, net_id, "backward")

        return utils.map_structure_and_replace_key(_torch_tensor_to_paddle_tensor, [t_fwd_item.output], x_grad)
    return x_grad


"""
    torch_api_hook,paddle_api_hook are used to record info to reports
"""


class PaddleLayerStr(paddle.nn.Layer):
    def __init__(self, net):
        super(PaddleLayerStr, self).__init__()
        self.__name__ = net.__name__
        self.__api__ = net.__api__


class TorchModuleStr(torch.nn.Module):
    def __init__(self, net):
        super(TorchModuleStr, self).__init__()
        self.__name__ = net.__name__
        self.__api__ = net.__api__


__in_torch_api_hook__ = False
__in_paddle_api_hook__ = False


def torch_api_hook(module, input, output, net_id):
    global __in_torch_api_hook__
    if __in_torch_api_hook__:
        return None

    torch_report = current_torch_report()

    # not in report_guard
    # if stack is emtpy, this api might be used in loss function or optimizer, skip
    if torch_report is None or torch_report.stack._top() is None:
        return None

    # if this api is not processing tensors, do not create report
    if output is None or all([not isinstance(x, torch.Tensor) for x in utils.flatten(output)]):
        return None

    # if an api under _layer_ignore_sublayer, do not create report
    # a layer under _layer_ignore_sublayer will not register this hook except it is a mapped one2one layer
    # torch_report.stack._top().net can not be an api layer !!!
    if torch_report.stack._top().net in torch_report.layer_map._layer_ignore_sublayer and hasattr(module, "__api__"):
        return None

    __in_torch_api_hook__ = True

    # if current module is an api layer, we do not want to hold it
    if hasattr(module, "__api__"):
        _module = TorchModuleStr(module)
    else:
        _module = module

    frame_info, frames = extract_frame_summary()
    new_in = utils.clone_structure(input)
    new_out = utils.clone_structure(output)
    fwd_item = torch_report.put_item("forward", new_in, new_out, _module, net_id, frame_info, frames)
    bwd_item = torch_report.put_item("backward", new_in, new_out, _module, net_id, frame_info, frames)
    bwd_item.set_forward(fwd_item)

    torch_report.stack.push_api(_module, fwd_item, bwd_item)

    for i, (t,) in enumerate(utils.for_each_grad_tensor(input)):
        t.register_hook(partial(torch_tensor_hook, bwd_item=bwd_item, nth_tensor=i))

    __in_torch_api_hook__ = False

    return None


def paddle_api_hook(module, input, output, net_id):
    """
    Notice: only wrapped api and layer in one2one will trigger this hook. They are leaves.
    """
    global __in_paddle_api_hook__
    if __in_paddle_api_hook__:
        return None

    paddle_report = current_paddle_report()

    if paddle_report is None or paddle_report.stack._top() is None:
        return None

    if output is None or all([not isinstance(x, paddle.Tensor) for x in utils.flatten(output)]):
        return None

    if paddle_report.stack._top().net in paddle_report.layer_map._layer_ignore_sublayer and hasattr(module, "__api__"):
        return None

    __in_paddle_api_hook__ = True

    if hasattr(module, "__api__"):
        _module = PaddleLayerStr(module)
    else:
        _module = module

    options = utils.yamls.options
    frame_info, frames = extract_frame_summary()
    new_in = utils.clone_structure(input)
    new_out = utils.clone_structure(output)
    fwd_item = paddle_report.put_item("forward", new_in, new_out, _module, net_id, frame_info, frames)
    bwd_item = paddle_report.put_item("backward", new_in, new_out, _module, net_id, frame_info, frames)
    bwd_item.set_forward(fwd_item)

    paddle_report.stack.push_api(_module, fwd_item, bwd_item)

    for i, (t,) in enumerate(utils.for_each_grad_tensor(input)):
        t.register_hook(partial(paddle_tensor_hook, bwd_item=bwd_item, nth_tensor=i, net_id=net_id))

    # if single_step, need return torch output
    if net_id != -1 and options["single_step"] and options["diff_phase"] == "forward":
        torch_report = current_torch_report()
        t_fwd_item = torch_report.find_item(paddle_report, net_id, "forward")

        retval = utils.map_structure_and_replace_key(_torch_tensor_to_paddle_tensor, [t_fwd_item.output], output)
        __in_paddle_api_hook__ = False
        return retval
    else:
        __in_paddle_api_hook__ = False
        return None


"""
    codes below are used to build module structure
"""


def paddle_pre_layer_hook(layer, input):
    rep = current_paddle_report()
    rep.stack.push_layer(layer)
    if layer in rep.layer_map._layer_one2one.values():
        rep.stack._top().is_one2one_layer = True
        rep.stack._top().is_leaf = True
    return None


def paddle_post_layer_hook(layer, input, output):
    rep = current_paddle_report()
    rep.stack.pop_layer(layer)
    return None


def torch_pre_layer_hook(module, input):
    rep = current_torch_report()
    rep.stack.push_layer(module)
    if module in rep.layer_map._layer_one2one.keys():
        rep.stack._top().is_one2one_layer = True
        rep.stack._top().is_leaf = True
    return None


def torch_post_layer_hook(module, input, output):
    rep = current_torch_report()
    rep.stack.pop_layer(module)
    return None


@contextlib.contextmanager
def register_paddle_hooker(runner):
    layer = runner.layer
    layer_map = runner.layer_map

    if os.getenv("PADIFF_CUDA_MEMORY") != "OFF":
        device = runner.paddle_device
        layer.to(device)

    remove_handles = []
    # TODO(xiongkun): duplicate layer is not support, implement custom generator to support (different net_id is ok).
    idx = 0
    layers = layer_map.struct_hook_layers(layer)
    for mod in layers:
        pre_handle = mod.register_forward_pre_hook(paddle_pre_layer_hook)
        # layers includes layer marked by ignore_recursively
        # if ignore_recursively, skip add report hook. if one2one, add report hook
        if mod not in layer_map._layer_ignore:
            handle = mod.register_forward_post_hook(partial(paddle_api_hook, net_id=idx))
            remove_handles.append(handle)
        post_handle = mod.register_forward_post_hook(paddle_post_layer_hook)
        remove_handles.extend([pre_handle, post_handle])
        idx += 1
    yield
    for h in remove_handles:
        h.remove()

    if os.getenv("PADIFF_CUDA_MEMORY") != "OFF":
        layer.to("cpu")
        paddle.device.cuda.empty_cache()


@contextlib.contextmanager
def register_torch_hooker(runner):
    module = runner.module
    layer_map = runner.layer_map

    if os.getenv("PADIFF_CUDA_MEMORY") != "OFF":
        device = runner.torch_device
        module.to(device)

    remove_handles = []
    idx = 0
    modules = layer_map.struct_hook_layers(module)
    for mod in modules:
        pre_handle = mod.register_forward_pre_hook(torch_pre_layer_hook)
        if mod not in layer_map._layer_ignore:
            handle = mod.register_forward_hook(partial(torch_api_hook, net_id=idx))
            remove_handles.append(handle)
        post_handle = mod.register_forward_hook(torch_post_layer_hook)
        remove_handles.extend([pre_handle, post_handle])
        idx += 1
    yield
    for h in remove_handles:
        h.remove()

    if os.getenv("PADIFF_CUDA_MEMORY") != "OFF":
        module.to("cpu")
        torch.cuda.empty_cache()


def _torch_tensor_to_paddle_tensor(tt):
    if isinstance(tt, torch.Tensor):
        if tt.numel() == 0:
            if tt.dtype == torch.float32 or tt.dtype == torch.float:
                return paddle.to_tensor([], dtype="float32")
            elif tt.dtype == torch.float64:
                return paddle.to_tensor([], dtype="float64")
            elif tt.dtype == torch.float16:
                return paddle.to_tensor([], dtype="float16")
            elif tt.dtype == torch.int32 or tt.dtype == torch.int:
                return paddle.to_tensor([], dtype="int32")
            elif tt.dtype == torch.int16:
                return paddle.to_tensor([], dtype="int16")
            elif tt.dtype == torch.int64:
                return paddle.to_tensor([], dtype="int64")
            else:
                raise RuntimeError(f"In single step mode, copy torch tensor {tt} with dtype {tt.dtype} Failed")
        return paddle.to_tensor(tt.detach().cpu().numpy())
    else:
        return tt


"""
    stack tools
"""


import os.path as osp
import traceback


def _is_system_package(filename):
    exclude = [
        "lib/python",
        "/usr/local",
        osp.dirname(paddle.__file__),
        osp.dirname(torch.__file__),
        osp.dirname(__file__),  # exclude padiff
    ]
    for pattern in exclude:
        if pattern in filename:
            return True
    return False


def extract_frame_summary():
    """
    extract the current call stack by traceback module.
    gather the call information and put them into ReportItem to helper locate the error.

    frame_summary:
        line: line of the code
        lineno: line number of the file
        filename: file name of the stack
        name: the function name.
    """
    frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
    last_user_fs = None
    for fs in frame_summarys:
        if not _is_system_package(fs.filename):
            last_user_fs = fs
            break
    assert last_user_fs is not None, "Error happend, can't return None."
    return last_user_fs, frame_summarys
