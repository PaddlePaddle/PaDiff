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
from .report import current_report
from ..utils import (
    clone_structure,
    clone_tensors,
    map_structure_and_replace_key,
    flatten,
    for_each_grad_tensor,
)

import paddle
import torch


@contextlib.contextmanager
def register_hooker(model):
    marker = model.marker

    remove_handles = []
    idx = 0
    # traversal_for_hook includes layers which we need add pre and post hook
    # for model structure, but not info hook (they are in black list)
    models = list(marker.traversal_for_hook(model))
    for mod in models:
        pre_handle = mod.register_forward_pre_hook(partial(pre_structure_hook))
        if mod not in marker.black_list:
            handle = mod.register_forward_post_hook(partial(info_hook, net_id=idx))
            remove_handles.append(handle)
        post_handle = mod.register_forward_post_hook(partial(post_structure_hook))
        remove_handles.extend([pre_handle, post_handle])
        idx += 1
    yield
    for h in remove_handles:
        h.remove()

"""
    hooks used to build module structure
"""

def pre_structure_hook(layer, input):
    report = current_report()
    report.stack.push_layer(layer)
    if layer in report.marker.layer_map:
        report.stack._top().layer_type = "in map"
        report.stack._top().layer_map_idx = report.marker.layer_map.index(layer)
        report.stack._top().is_leaf = True
    return None


def post_structure_hook(layer, input, output):
    report = current_report()
    retval = report.stack.pop_layer(layer)
    if retval in report.marker.black_list:
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

    __in_info_hook__ = True

    # if current model is an api layer, we do not want to hold it
    if hasattr(model, "__api__"):
        _model = padiff_layer_str(model)
    else:
        _model = model

    new_in = clone_tensors(input)
    new_out = clone_tensors(output)
    fwd_item = report.put_item("forward", new_in, new_out, _model, net_id)
    bwd_item = report.put_item("backward", new_in, new_out, _model, net_id)
    bwd_item.set_forward(fwd_item)

    report.stack.push_api(_model, fwd_item, bwd_item)

    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i, net_id=net_id))

    # if under single step forward guard
    if net_id != -1 and False:
        base_report = None # get_base_report()
        base_fwd_item = base_report.find_item(report, net_id, "forward")

        retval = map_structure_and_replace_key(
            partial(_transform_tensor, type_="paddle" if isinstance(model, paddle.nn.Layer) else "torch"),
            [base_fwd_item.output],
            output,
        )
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

    # if under single step backward guard
    if net_id != -1 and False:
        base_report = None # TODO
        raw_report = current_report()
        base_bwd_item = raw_report.find_item(base_report, net_id, "backward")

        return map_structure_and_replace_key(
            partial(_transform_tensor, type="paddle" if isinstance(x_grad, paddle.Tensor) else "torch"),
            [base_bwd_item.input_grads],
            x_grad,
        )
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
        super(PaddleLayerStr, self).__init__()
        self.__name__ = net.__name__
        self.__api__ = net.__api__


class TorchModuleStr(torch.nn.Module):
    def __init__(self, net):
        super(TorchModuleStr, self).__init__()
        self.__name__ = net.__name__
        self.__api__ = net.__api__


def _transform_tensor(tensor, type_):
    if isinstance(tensor, (torch.Tensor, paddle.Tensor)):
        if tensor.numel() == 0:
            if tensor.dtype == torch.float32 or tensor.dtype == torch.float:
                retval = paddle.to_tensor([], dtype="float32")
            elif tensor.dtype == torch.float64:
                retval = paddle.to_tensor([], dtype="float64")
            elif tensor.dtype == torch.float16:
                retval = paddle.to_tensor([], dtype="float16")
            elif tensor.dtype == torch.int32 or tensor.dtype == torch.int:
                retval = paddle.to_tensor([], dtype="int32")
            elif tensor.dtype == torch.int16:
                retval = paddle.to_tensor([], dtype="int16")
            elif tensor.dtype == torch.int64:
                retval = paddle.to_tensor([], dtype="int64")
            else:
                raise RuntimeError(f"In single step mode, copy torch tensor {tensor} with dtype {tensor.dtype} Failed")
        else:
            retval = paddle.to_tensor(tensor.detach().cpu().numpy())

        return retval if type_ == "paddle" else torch.Tensor(retval.numpy())
    else:
        return tensor