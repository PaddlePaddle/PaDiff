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
    rep = current_torch_report()

    if rep is None:
        return None
    if output is None or all([not isinstance(x, torch.Tensor) for x in paddle.fluid.layers.utils.flatten(output)]):
        return None

    frame_info, frames = extract_frame_summary()
    fwd_item = rep.put_item("forward", input, output, module, idx, frame_info, frames)
    bwd_item = rep.put_item("backward", input, output, module, idx, frame_info, frames)
    bwd_item.set_forward(fwd_item)
    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i))
    return None


def paddle_api_hook(module, input, output, idx):
    p_rep = current_paddle_report()

    if p_rep is None:
        return None

    if output is None or all([not isinstance(x, paddle.Tensor) for x in paddle.fluid.layers.utils.flatten(output)]):
        return None

    options = yamls.options
    frame_info, frames = extract_frame_summary()
    fwd_item = p_rep.put_item("forward", input, output, module, idx, frame_info, frames)
    bwd_item = p_rep.put_item("backward", input, output, module, idx, frame_info, frames)
    bwd_item.set_forward(fwd_item)
    for i, (t,) in enumerate(for_each_grad_tensor(input)):
        t.register_hook(partial(tensor_hook, bwd_item=bwd_item, nth_tensor=i))

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


class LayerStack(object):
    def __init__(self, type_):
        super(LayerStack, self).__init__()
        self.type = type_
        self.stack = []
        self.sons = {}

    def _push(self, value):
        self.stack.append(value)

    def _pop(self):
        return self.stack.pop()

    def _top(self):
        return self.stack[-1]

    def _empty(self):
        return len(self.stack) == 0

    @property
    def current(self):
        return self._top()

    def in_layer(self, module):
        if not self._empty():
            self.sons[self.current].append(module)
        self._push(module)
        self.sons[module] = []

    def out_layer(self, module):
        assert id(self.current) == id(module)
        self._pop()

    def in_api(self, api):
        if not self._empty():
            if self.current not in self.sons.keys():
                self.sons[self.current] = []
            self.sons[self.current].append(api)


def torch_pre_layer_hook(module, input, output, idx):
    pass


def paddle_pre_layer_hook(module, input, output, idx):
    pass


def torch_post_layer_hook(module, input, output, idx):
    pass


def paddle_post_layer_hook(module, input, output, idx):
    pass


@contextlib.contextmanager
def _register_paddle_hooker(layer, layer_map):
    remove_handles = []
    # TODO(xiongkun): duplicate layer is not support, implement custom generator to support (different net_id is ok).
    idx = 0
    layers = layer_map.layers(layer)
    for mod in layers:
        handle = mod.register_forward_post_hook(partial(paddle_api_hook, idx=idx))
        remove_handles.append(handle)
        idx += 1
    yield
    for h in remove_handles:
        h.remove()


@contextlib.contextmanager
def _register_torch_hooker(module, layer_map):
    remove_handles = []
    idx = 0
    modules = layer_map.layers(module)
    for mod in modules:
        handle = mod.register_forward_hook(partial(torch_api_hook, idx=idx))
        remove_handles.append(handle)
        idx += 1
    yield
    for h in remove_handles:
        h.remove()
