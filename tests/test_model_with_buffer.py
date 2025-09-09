# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from padiff import *
import paddle
import torch


class LayerWithinitializedBuffer(paddle.nn.Layer):
    initialized_buffer: paddle.Tensor

    def __init__(self):
        super(LayerWithinitializedBuffer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.linear2 = paddle.nn.Linear(100, 10)
        self.act = paddle.nn.ReLU()
        self.register_buffer("initialized_buffer", paddle.zeros([1]))

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual + self.initialized_buffer
        x = self.linear2(x)
        return x


class ModuleWithinitializedBuffer(torch.nn.Module):
    initialized_buffer: torch.Tensor

    def __init__(self):
        super(ModuleWithinitializedBuffer, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)
        self.linear2 = torch.nn.Linear(100, 10)
        self.act = torch.nn.ReLU()
        self.register_buffer("initialized_buffer", torch.zeros([1]))

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual + self.initialized_buffer
        x = self.linear2(x)
        return x


class LayerWithUninitializedBuffer(paddle.nn.Layer):
    uninitialized_buffer: paddle.Tensor

    def __init__(self):
        super(LayerWithUninitializedBuffer, self).__init__()
        self.linear = paddle.nn.Linear(10, 10)
        self.register_buffer("uninitialized_buffer", None)
        self._first_forward = True

    def forward(self, x):
        if self._first_forward:
            self.uninitialized_buffer = paddle.zeros_like(x)
            self._first_forward = False
        return self.linear(x) + self.uninitialized_buffer


class ModuleWithUninitializedBuffer(torch.nn.Module):
    uninitialized_buffer: torch.Tensor

    def __init__(self):
        super(ModuleWithUninitializedBuffer, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.register_buffer("uninitialized_buffer", torch.empty(0))
        self._first_forward = True

    def forward(self, x):
        if self._first_forward:
            self.uninitialized_buffer = torch.zeros_like(x)
            self._first_forward = False
        return self.linear(x) + self.uninitialized_buffer


class TestModelWithBuffer(unittest.TestCase):
    def test_initialized_buffer(self):
        layer = create_model(LayerWithinitializedBuffer())
        module = create_model(ModuleWithinitializedBuffer())

        inp = paddle.rand((1, 100)).numpy().astype("float32")
        inp = ({"x": torch.as_tensor(inp)}, {"x": paddle.to_tensor(inp)})
        assert auto_diff(module, layer, inp, atol=1e-4) is True, "Failed. expected success."

    def test_uninitialized_buffer(self):
        layer = create_model(LayerWithUninitializedBuffer())
        module = create_model(ModuleWithUninitializedBuffer())

        inp = paddle.rand((1, 10)).numpy().astype("float32")
        inp = ({"x": torch.as_tensor(inp)}, {"x": paddle.to_tensor(inp)})

        assert auto_diff(module, layer, inp, atol=1e-4) is True, "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
