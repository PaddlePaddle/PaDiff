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

from .init_LSTM import init_LSTM
from .init_MultiHeadAttention import init_MultiHeadAttention


# NOTICE: make sure torch params is in the same device after init
special_init_tools = {
    "LSTM": init_LSTM,
    "MultiHeadAttention": init_MultiHeadAttention,
}


def add_special_init(inp_dict):
    special_init_tools.update(inp_dict)
