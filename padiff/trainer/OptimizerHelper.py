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


import paddle
import torch

__all__ = [
    "OptimizerHelper",
]


class OptimizerHelper:
    def __init__(self, optimizers, options):
        self.use_opt = options["use_opt"]
        self.optimizers = optimizers

    def step(self):
        if self.use_opt:
            for opt in self.optimizers:
                if isinstance(
                    opt,
                    paddle.optimizer.Optimizer,
                ):
                    opt.step()
                    opt.clear_grad()
                elif isinstance(opt, torch.optim.Optimizer):
                    opt.step()
                    opt.zero_grad()
                else:
                    opt()
