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
from .report import current_torch_report, current_paddle_report, report_guard
from .stack_info import *
from .utils import (
    for_each_grad_tensor,
    log,
    max_diff,
    tensors_mean,
    map_structure_and_replace_key,
)
from .weights import remove_inplace, assign_weight


class Trainer(object):
    def __init__(self, layer, module, loss_fn, opt):
        self.layer = layer  # paddle layer
        self.module = module  # torch module
        if loss_fn is not None:
            self.paddle_loss = loss_fn[0]
            self.torch_loss = loss_fn[1]
        if opt is not None:
            self.has_opt = True
            self.paddle_opt = opt[0]
            self.torch_opt = opt[1]
        else:
            self.has_opt = False

        remove_inplace(self.layer, self.module)

        self.paddle_rep = None
        self.torch_rep = None

    def assign_weight_(self, layer_map):
        assign_weight(self.layer, self.module, layer_map=layer_map)

    def set_report(self, paddle_rep, torch_rep):
        self.paddle_rep = paddle_rep
        self.torch_rep = torch_rep

    def train_step(self, example_inp, options, layer_map):
        paddle_input, torch_input = example_inp
        with report_guard(self.torch_rep, self.paddle_rep):
            with _register_torch_hooker(self.module, options, layer_map):
                try:
                    torch_output = self.module(**torch_input)
                    if options["loss_fn"]:
                        loss = self.torch_loss(torch_output)
                        self.torch_rep.set_loss(loss)
                    else:
                        loss = tensors_mean(torch_output, "torch")
                    if options["diff_phase"] == "both":
                        loss.backward()
                        if options["opt"]:
                            self.torch_opt.step()
                except Exception as e:
                    raise RuntimeError(
                        "Exception is thrown while running forward of torch_module, please check the legality of module.\n{}".format(
                            str(e)
                        )
                    )

            with _register_paddle_hooker(self.layer, options, layer_map):
                try:
                    paddle_output = self.layer(**paddle_input)
                    if options["loss_fn"]:
                        loss = self.paddle_loss(paddle_output)
                        self.paddle_rep.set_loss(loss)
                    else:
                        loss = tensors_mean(paddle_output, "paddle")
                    if options["diff_phase"] == "both":
                        loss.backward()
                        if options["opt"]:
                            self.paddle_opt.step()
                except Exception as e:
                    raise RuntimeError(
                        "Exception is thrown while running forward of paddle_layer, please check the legality of layer.\n{}".format(
                            str(e)
                        )
                    )

        log("Max elementwise output diff is {}\n".format(max_diff(paddle_output, torch_output)))

    def clear_grad(self):
        if self.has_opt:
            self.paddle_opt.clear_grad()
            self.torch_opt.zero_grad()


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
                return paddle.to_tensor(tt.detach().cpu().numpy())
            else:
                return tt

        return map_structure_and_replace_key(tt2pt, [t_fwd_item.output], output)
    else:
        return None


@contextlib.contextmanager
def _register_paddle_hooker(layer, options, layer_map):
    remove_handles = []
    # TODO(xiongkun): duplicate layer is not support, implement custom generator to support (different net_id is ok).
    idx = 0
    layers = layer_map.layers(layer)
    for mod in layers:
        handle = mod.register_forward_post_hook(partial(paddle_layer_hook, idx=idx, options=options))
        remove_handles.append(handle)
        idx += 1
    yield
    for h in remove_handles:
        h.remove()


@contextlib.contextmanager
def _register_torch_hooker(module, options, layer_map):
    remove_handles = []
    idx = 0
    modules = layer_map.layers(module)
    for mod in modules:
        handle = mod.register_forward_hook(partial(torch_layer_hook, idx=idx, options=options))
        remove_handles.append(handle)
        idx += 1
    yield
    for h in remove_handles:
        h.remove()
