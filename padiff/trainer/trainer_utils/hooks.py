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
from .report import current_reports
from .. import utils
import os
import paddle
import torch


"""
    torch_tensor_hook, paddle_tensor_hook are used to create backward report items
"""


def paddle_tensor_hook(x_grad, bwd_item, nth_tensor, net_id, model_idx):
    # print (nth_tensor, bwd_item.input_grads, bwd_item.input)
    # record paddle grad
    bwd_item.set_input_grads(nth_tensor, x_grad)

    options = utils.yamls.options
    # single_step and not an API
    if model_idx == 0 and net_id != -1 and options["single_step"] and options["diff_phase"] == "backward":
        paddle_report = current_reports()[0]
        torch_report = current_reports()[1]
        t_fwd_item = torch_report.find_item(paddle_report, net_id, "backward")

        return utils.map_structure_and_replace_key(
            partial(_transform_tensor, type="paddle" if isinstance(x_grad, paddle.Tensor) else "torch"),
            [t_fwd_item.output],
            x_grad,
        )
    return x_grad


"""
    torch_api_hook,paddle_api_hook are used to record info to reports
"""


def padiff_layer_str(model):
    if isinstance(model, paddle.nn.Layer):
        return PaddleLayerStr(model)
    else:
        return TorchModuleStr(model)


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


__in_api_hook__ = False


# this model is not PadiffModel
def api_hook(model, input, output, net_id):
    """
    Notice: only wrapped api and layer in one2one will trigger this hook. They are leaves.
    """
    options = utils.yamls.options
    if not options or options["curent_model_idx"] is None:
        return None
    model_idx = options["curent_model_idx"]

    global __in_api_hook__
    if __in_api_hook__:
        return None

    report = current_reports()[model_idx]

    # not in report_guard
    # if stack is emtpy, this api might be used in loss function or optimizer, skip
    if report is None or report.stack._top() is None:
        return None

    # if this api is not processing tensors, do not create report
    if output is None or all([not isinstance(x, (paddle.Tensor, torch.Tensor)) for x in utils.flatten(output)]):
        return None

    # if an api under _layer_ignore_sublayer, do not create report
    # a layer under _layer_ignore_sublayer will not register this hook except it is a mapped one2one layer
    # torch_report.stack._top().net can not be an api layer !!!
    if report.stack._top().net in report.layer_map._layer_ignore_sublayer and hasattr(model, "__api__"):
        return None

    __in_api_hook__ = True

    # if current model is an api layer, we do not want to hold it
    if hasattr(model, "__api__"):
        _model = padiff_layer_str(model)
    else:
        _model = model

    frame_info, frames = extract_frame_summary()
    new_in = utils.clone_structure(input)
    new_out = utils.clone_structure(output)
    fwd_item = report.put_item("forward", new_in, new_out, _model, net_id, frame_info, frames)
    bwd_item = report.put_item("backward", new_in, new_out, _model, net_id, frame_info, frames)
    bwd_item.set_forward(fwd_item)

    report.stack.push_api(_model, fwd_item, bwd_item)

    for i, (t,) in enumerate(utils.for_each_grad_tensor(input)):
        t.register_hook(
            partial(paddle_tensor_hook, bwd_item=bwd_item, nth_tensor=i, net_id=net_id, model_idx=model_idx)
        )

    # if single_step, need return torch output
    if model_idx == 0 and net_id != -1 and options["single_step"] and options["diff_phase"] == "forward":
        torch_report = current_reports()[1]
        t_fwd_item = torch_report.find_item(report, net_id, "forward")

        retval = utils.map_structure_and_replace_key(
            partial(_transform_tensor, type="paddle" if isinstance(model, paddle.nn.Layer) else "torch"),
            [t_fwd_item.output],
            output,
        )
        __in_api_hook__ = False
        return retval
    else:
        __in_api_hook__ = False
        return None


def _transform_tensor(tt, type):
    if isinstance(tt, (torch.Tensor, paddle.Tensor)):
        if tt.numel() == 0:
            if tt.dtype == torch.float32 or tt.dtype == torch.float:
                retval = paddle.to_tensor([], dtype="float32")
            elif tt.dtype == torch.float64:
                retval = paddle.to_tensor([], dtype="float64")
            elif tt.dtype == torch.float16:
                retval = paddle.to_tensor([], dtype="float16")
            elif tt.dtype == torch.int32 or tt.dtype == torch.int:
                retval = paddle.to_tensor([], dtype="int32")
            elif tt.dtype == torch.int16:
                retval = paddle.to_tensor([], dtype="int16")
            elif tt.dtype == torch.int64:
                retval = paddle.to_tensor([], dtype="int64")
            else:
                raise RuntimeError(f"In single step mode, copy torch tensor {tt} with dtype {tt.dtype} Failed")
        else:
            retval = paddle.to_tensor(tt.detach().cpu().numpy())

        return retval if type == "paddle" else torch.Tensor(retval.numpy())
    else:
        return tt


"""
    codes below are used to build module structure
"""


def pre_layer_hook(layer, input, model_idx):
    rep = current_reports()[model_idx]
    rep.stack.push_layer(layer)
    if layer in rep.layer_map.layers_in_map():
        rep.stack._top().is_one2one_layer = True
        rep.stack._top().is_leaf = True
    return None


def post_layer_hook(layer, input, output, model_idx):
    rep = current_reports()[model_idx]
    rep.stack.pop_layer(layer)
    return None


@contextlib.contextmanager
def register_hooker(runner, model_idx):
    options = utils.yamls.options
    options["curent_model_idx"] = model_idx

    model = runner.models[model_idx]
    device = runner.devices[model_idx]
    layer_map = runner.layer_map

    if os.getenv("PADIFF_CUDA_MEMORY") != "OFF":
        model.to(device)

    remove_handles = []
    idx = 0
    models = layer_map.struct_hook_layers(model)
    for mod in models:
        pre_handle = mod.register_forward_pre_hook(partial(pre_layer_hook, model_idx=model_idx))
        # layer includes layer marked by ignore_recursively
        # if ignore_recursively, skip add report hook. if one2one, add report hook
        if mod not in layer_map._layer_ignore:
            handle = mod.register_forward_post_hook(partial(api_hook, net_id=idx))
            remove_handles.append(handle)
        post_handle = mod.register_forward_post_hook(partial(post_layer_hook, model_idx=model_idx))
        remove_handles.extend([pre_handle, post_handle])
        idx += 1
    yield
    for h in remove_handles:
        h.remove()

    if os.getenv("PADIFF_CUDA_MEMORY") != "OFF":
        model.to_cpu()
    options["curent_model_idx"] = None


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
