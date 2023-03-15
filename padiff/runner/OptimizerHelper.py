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


__all__ = [
    "OptimizerHelper",
]


class OptimizerHelper:
    def __init__(self, opt, options):
        self.use_opt = options["use_opt"]
        if options["use_opt"]:
            self.paddle_opt = opt[0]
            self.torch_opt = opt[1]
            self.opt_type = options["opt_type"]

    def step(self):
        if self.use_opt:
            if self.opt_type == "Lambda":
                self.paddle_opt()
                self.torch_opt()
            elif self.opt_type == "Opt":
                self.paddle_opt.step()
                self.paddle_opt.clear_grad()
                self.torch_opt.step()
                self.torch_opt.zero_grad()
