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


@init_pool.register(paddle_name="Conv1D", torch_name="Conv1d")  # 此处填写模型的类名
@init_pool.register(paddle_name="Conv2D", torch_name="Conv2d")  # 此处填写模型的类名
@init_pool.register(paddle_name="Conv3D", torch_name="Conv3d")  # 此处填写模型的类名
def init_weight_norm_conv(layer, module):
    """
    经过weight_norm封装后存在torch和paddle中 weight_g和weight_v的顺序不一致导致权重转换失败
    """
    state_dict = layer.state_dict()
    if isinstance(layer, (paddle.nn.Conv1D, paddle.nn.Conv2D, paddle.nn.Conv3D)):
        if 'weight_g' in state_dict.keys():
            layer.weight_g.set_value(module.weight_g.squeeze().cpu().detach().numpy())
            layer.weight_v.set_value(module.weight_v.cpu().detach().numpy())
        else:
            layer.weight.set_value(module.weight.squeeze().cpu().detach().numpy())

    if 'bias' in state_dict.keys():
        layer.bias.set_value(module.bias.cpu().detach().numpy())