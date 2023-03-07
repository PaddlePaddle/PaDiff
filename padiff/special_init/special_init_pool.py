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


class SpecialInitPool(object):
    def __init__(self):
        self.funcs = {}

    def register(self, name):
        def do_reg(func):
            self.funcs[name] = func
            return func

        return do_reg


global_special_init_pool = SpecialInitPool()


def add_special_init(inp_dict):
    global_special_init_pool.funcs.update(inp_dict)
