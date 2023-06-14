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

import unittest

import paddle
import torch

from padiff import *


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 10)
        self.linear2 = paddle.nn.Linear(100, 20)
        self.linear3 = paddle.nn.Linear(100, 30)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x3 = self.linear3(x)
        return [x1, x2, x3]


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear(100, 10)
        self.linear2 = torch.nn.Linear(100, 20)
        self.linear3 = torch.nn.Linear(100, 30)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x3 = self.linear3(x)
        return [x1, x2, x3]


class TestCaseName(unittest.TestCase):
    def test_success(self):
        layer = create_model(SimpleLayer())
        module = create_model(SimpleModule())
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        assert auto_diff(layer, module, inp, atol=1e-4) is True, "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
