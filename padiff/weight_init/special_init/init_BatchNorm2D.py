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

from .special_init_pool import global_special_init_pool as init_pool


@init_pool.register("torch", "BatchNorm2d", "paddle", "BatchNorm2D")
def init_BatchNorm2D(module, layer):
    param_dict = {}
    for name, param in module.state_dict().items():
        name = name.replace("running_var", "_variance").replace("running_mean", "_mean")
        param_dict[name] = paddle.to_tensor(param.cpu().detach().numpy())
    layer.set_state_dict(param_dict)
