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
from .stack_info import *
from .utils import (
    for_each_grad_tensor,
    map_structure_and_replace_key,
)
from .yaml_loader import global_yaml_loader as yamls

"""
    torch_api_hook, paddle_api_hook are used to create report items
"""


def tensor_hook(x_grad, bwd_item, nth_tensor):
    # print (nth_tensor, bwd_item.input_grads, bwd_item.input)
    bwd_item.set_input_grads(nth_tensor, x_grad)
    return x_grad


# torch_api_hook,paddle_api_hook are used to record info to reports
def torch_api_hook(module, input, output, idx):
    """
    Notice: only wrapped api and mapped one2one layer will trigger this hook. They are leaves.
    """
    t_rep = current_torch_report()

    # not in report_guard
    if t_rep is None:
        return None

    # if this api is not processing tensors, do not create report
    if output is None or all([not isinstance(x, torch.Tensor) for x in paddle.fluid.layers.utils.flatten(output)]):
        return None

    # if an api under _layer_ignore_sublayer, do not create report
    # a layer under _layer_ignore_sublayer will not register this hook
    # except a mapped one2one layer
    if t_rep.stack._top().net in t_rep.layer_map._layer_ignore_sublayer and hasattr(module, "__api__"):
        return None

    frame_info, frames = extract_frame_summary()
    fwd_item = t_rep.put_item("forward", input, output, module, idx, frame_info, frames)
    bwd_item = t_rep.put_item("backward", input, output, module, idx, frame_info, frames)
    bwd_item.set_forward(fwd_item)

    t_rep.stack.push_api(module, fwd_item, bwd_item)

    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i))
    return None


def paddle_api_hook(module, input, output, idx):
    """
    Notice: only wrapped api and layer in one2one will trigger this hook. They are leaves.
    """

    p_rep = current_paddle_report()

    if p_rep is None:
        return None

    if output is None or all([not isinstance(x, paddle.Tensor) for x in paddle.fluid.layers.utils.flatten(output)]):
        return None

    if p_rep.stack._top().net in p_rep.layer_map._layer_ignore_sublayer and hasattr(module, "__api__"):
        return None

    options = yamls.options
    frame_info, frames = extract_frame_summary()
    fwd_item = p_rep.put_item("forward", input, output, module, idx, frame_info, frames)
    bwd_item = p_rep.put_item("backward", input, output, module, idx, frame_info, frames)
    bwd_item.set_forward(fwd_item)

    p_rep.stack.push_api(module, fwd_item, bwd_item)

    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i))

    # if single_step, need return torch output
    if options["single_step"] and idx != -1:
        t_rep = current_torch_report()
        t_fwd_item = t_rep.find_item(p_rep, idx)

        def tt2pt(tt):
            if isinstance(tt, torch.Tensor):
                return paddle.to_tensor(tt.detach().cpu().numpy())
            else:
                return tt

        return map_structure_and_replace_key(tt2pt, [t_fwd_item.output], output)
    else:
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
def _register_paddle_hooker(layer, layer_map):
    remove_handles = []
    # TODO(xiongkun): duplicate layer is not support, implement custom generator to support (different net_id is ok).
    idx = 0
    layers = layer_map.layers_skip_ignore(layer)
    for mod in layers:
        pre_handle = mod.register_forward_pre_hook(paddle_pre_layer_hook)
        # call api_hook before post_layer_hook => current will be module itself
        # if mod in layer_map._layer_one2one.keys():
        if True:
            handle = mod.register_forward_post_hook(partial(paddle_api_hook, idx=idx))
            remove_handles.append(handle)
        post_handle = mod.register_forward_post_hook(paddle_post_layer_hook)
        remove_handles.extend([pre_handle, post_handle])
        idx += 1
    yield
    for h in remove_handles:
        h.remove()


@contextlib.contextmanager
def _register_torch_hooker(module, layer_map):
    remove_handles = []
    idx = 0
    modules = layer_map.layers_skip_ignore(module)
    for mod in modules:
        pre_handle = mod.register_forward_pre_hook(torch_pre_layer_hook)
        # if mod in layer_map._layer_one2one.values():
        if True:
            handle = mod.register_forward_hook(partial(torch_api_hook, idx=idx))
            remove_handles.append(handle)
        post_handle = mod.register_forward_hook(torch_post_layer_hook)
        remove_handles.extend([pre_handle, post_handle])
        idx += 1
    yield
    for h in remove_handles:
        h.remove()
