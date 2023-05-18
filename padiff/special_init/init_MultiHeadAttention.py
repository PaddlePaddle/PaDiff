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


from .special_init_pool import global_special_init_pool as init_pool
import numpy
import torch


@init_pool.register("paddle", "MultiHeadAttention", "torch", "MultiheadAttention")
def init_MultiHeadAttention(layer, module):
    name_param_dict = {}
    for i, param in enumerate(layer.named_parameters()):
        pname = param[0]
        if "cross_attn" in pname:
            pname = pname.replace("cross_attn", "multihead_attn")
        elif "q" not in pname and "k" not in pname and "v" not in pname:
            continue
        param_np = param[1].numpy()
        pname = pname.replace("q_proj.", "in_proj_")
        pname = pname.replace("k_proj.", "in_proj_")
        pname = pname.replace("v_proj.", "in_proj_")
        if pname not in name_param_dict:
            name_param_dict[pname] = param_np
        elif "_weight" in pname:
            name_param_dict[pname] = numpy.concatenate((name_param_dict[pname], param_np), axis=1)
        else:
            name_param_dict[pname] = numpy.concatenate((name_param_dict[pname], param_np), axis=0)

    for i, param in enumerate(module.named_parameters()):
        pname, pa = param[0], param[1]
        if "in_proj" in pname or "multihead_attn" in pname:
            param_np = name_param_dict[pname]
        else:
            param_np = layer.state_dict()[pname].numpy()
        if pname.endswith("weight"):
            param_np = numpy.transpose(param_np)
        device = param[1].device
        param[1].data = torch.from_numpy(param_np).to(device)
