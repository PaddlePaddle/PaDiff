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

from .report import report_guard
from .utils import (
    log,
    max_diff,
    tensors_mean,
)
from .weights import remove_inplace, assign_weight
from .hooks import _register_paddle_hooker, _register_torch_hooker


class Trainer(object):
    def __init__(self, layer, module, loss_fn, opt, layer_map):
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

        # layer_map should be part of the module
        self.layer_map = layer_map

        remove_inplace(self.layer, self.module)

        self.paddle_rep = None
        self.torch_rep = None

    def assign_weight_(self):
        return assign_weight(self.layer, self.module, self.layer_map)

    def set_report(self, paddle_rep, torch_rep):
        paddle_rep.layer_map = self.layer_map
        torch_rep.layer_map = self.layer_map
        self.paddle_rep = paddle_rep
        self.torch_rep = torch_rep

    def train_step(self, example_inp, options):
        paddle_input, torch_input = example_inp
        with report_guard(self.torch_rep, self.paddle_rep):
            with _register_torch_hooker(self.module, self.layer_map):
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
                            self.torch_opt()
                except Exception as e:
                    raise RuntimeError(
                        "Exception is thrown while running forward of torch_module, please check the legality of module.\n{}".format(
                            str(e)
                        )
                    )

            with _register_paddle_hooker(self.layer, self.layer_map):
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
                            self.paddle_opt()
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
