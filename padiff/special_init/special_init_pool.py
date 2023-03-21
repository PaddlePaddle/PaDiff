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


# NOTICE: make sure torch params is in the same device after init


def build_name(paddle_name, torch_name):
    name = paddle_name + "###" + torch_name
    return name


class SpecialInitPool(object):
    def __init__(self):
        self.funcs = {}
        self.registered_paddle_layers = set()
        self.registered_torch_modules = set()

    def register(self, paddle_name, torch_name):
        name = build_name(paddle_name, torch_name)
        self.registered_paddle_layers.add(paddle_name)
        self.registered_torch_modules.add(torch_name)

        def do_reg(func):
            self.funcs[name] = func
            return func

        return do_reg


global_special_init_pool = SpecialInitPool()


def add_special_init(paddle_name, torch_name, func):
    name = build_name(paddle_name, torch_name)
    global_special_init_pool.registered_paddle_layers.add(paddle_name)
    global_special_init_pool.registered_torch_modules.add(torch_name)
    global_special_init_pool.funcs[name] = func
