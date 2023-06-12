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

import os

os.environ["PADIFF_API_CHECK"] = "ON"

import unittest

from padiff import *
import paddle
import torch


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.relu = paddle.nn.functional.relu

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = x * 2
        x = x + 1
        return x


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = x * 2
        x = x + 1
        return x


class TestCaseName(unittest.TestCase):
    def test_api_to_Layer(self):
        layer = SimpleLayer()
        layer = create_model(layer)

        # module = SimpleModule()
        # module = create_model(module)
        inp = paddle.rand((100, 100), dtype="float32")

        layer(inp)
        layer.report

        assert len(layer.report.items) == 12


if __name__ == "__main__":
    unittest.main()
    os.environ["PADIFF_API_CHECK"] = "OFF"
