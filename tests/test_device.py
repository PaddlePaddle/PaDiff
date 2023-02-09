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

from padiff import auto_diff


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.linear2 = paddle.nn.Linear(100, 10)
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)
        self.linear2 = torch.nn.Linear(100, 10)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x


class TestCaseName(unittest.TestCase):
    def test_cpu_cpu(self):
        paddle.set_device("cpu")
        layer = SimpleLayer()
        module = SimpleModule().to("cpu")
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp).to("cpu")})
        assert (
            auto_diff(layer, module, inp, auto_weights=True, options={"atol": 1e-4}) is True
        ), "Failed. expected success."

    def test_cpu_gpu(self):
        if not paddle.device.is_compiled_with_cuda():
            return
        paddle.set_device("cpu")
        layer = SimpleLayer()
        module = SimpleModule()
        module = module.to("cuda")
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp).to("cuda")})
        assert (
            auto_diff(layer, module, inp, auto_weights=True, options={"atol": 1e-4}) is True
        ), "Failed. expected success."

    def test_gpu_cpu(self):
        if not paddle.device.is_compiled_with_cuda():
            return
        paddle.set_device("gpu")
        layer = SimpleLayer()
        module = SimpleModule().to("cpu")
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp).to("cpu")})
        assert (
            auto_diff(layer, module, inp, auto_weights=True, options={"atol": 1e-4}) is True
        ), "Failed. expected success."

    def test_gpu_gpu(self):
        if not paddle.device.is_compiled_with_cuda():
            return
        paddle.set_device("gpu")
        layer = SimpleLayer()
        module = SimpleModule().to("cuda")
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp).to("cuda")})
        assert (
            auto_diff(layer, module, inp, auto_weights=True, options={"atol": 1e-4}) is True
        ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
