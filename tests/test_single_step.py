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
        self.layers = paddle.nn.Sequential(*[paddle.nn.Linear(100, 100) for i in range(10)])

    def forward(self, x):
        for net in self.layers:
            x = net(x)
        return x


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.layers = torch.nn.Sequential(*[torch.nn.Linear(100, 100) for i in range(10)])

    def forward(self, x):
        for net in self.layers:
            x = net(x)
        return x


class SimpleLayerDiff(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayerDiff, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.linear2 = paddle.nn.Linear(100, 100)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear1(x)
        x3 = self.linear2(x)
        x3.register_hook(lambda g: g + 0.01)
        return x1 + x2 + x3


class SimpleModuleDiff(torch.nn.Module):
    def __init__(self):
        super(SimpleModuleDiff, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)
        self.linear2 = torch.nn.Linear(100, 100)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear1(x)
        x3 = self.linear2(x)
        return x2 + x1 + x3


class TestSingleStep(unittest.TestCase):
    def test_success(self):
        layer = create_model(SimpleLayer())
        module = create_model(SimpleModule())
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        atol = 1e-5
        want_True = auto_diff(layer, module, inp, atol=atol, single_step=True)
        if want_True is not True:
            print("err atol too small")

    def test_failed(self):
        layer = create_model(SimpleLayerDiff())
        module = create_model(SimpleModuleDiff())
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        assert auto_diff(layer, module, inp, atol=1e-4, single_step=True) is False, "Success. expected failed."


if __name__ == "__main__":
    unittest.main()
