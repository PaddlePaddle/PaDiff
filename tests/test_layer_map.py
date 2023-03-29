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

import os

os.environ["PADIFF_CUDA_MEMO"] = "OFF"

import unittest
from padiff import auto_diff
import paddle
import torch


class SimpleLayer1(paddle.nn.Layer):
    def __init__(self, in_size, hidden_size, out_size):
        super(SimpleLayer1, self).__init__()

        layers = []
        layers.append(paddle.nn.Linear(in_size, hidden_size))
        layers.append(paddle.nn.LayerNorm(hidden_size))
        layers.append(paddle.nn.Linear(hidden_size, out_size))
        self.layers = paddle.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SimpleModule1(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(SimpleModule1, self).__init__()

        self.linear1 = torch.nn.Linear(in_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.layer_norm(x)
        x = self.linear2(x)
        return x


class SimpleLayer2(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer2, self).__init__()
        self.embedder = paddle.nn.Embedding(3, 16)
        self.lstm = paddle.nn.LSTM(16, 8, 2, time_major=True)

    def forward(self, x):
        x = self.embedder(x)
        x, _ = self.lstm(x)
        return x


class SimpleModule2(torch.nn.Module):
    def __init__(self):
        super(SimpleModule2, self).__init__()
        self.embedder = torch.nn.Embedding(3, 16)
        self.lstm = torch.nn.LSTM(
            input_size=16,
            hidden_size=8,
            num_layers=2,
        )

    def forward(self, x):
        x = self.embedder(x)
        x, _ = self.lstm(x)
        return x


class SimpleLayer3(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer3, self).__init__()
        self.attn = paddle.nn.MultiHeadAttention(16, 1)

    def forward(self, q, k, v):
        x = self.attn(q, k, v)
        return x


class SimpleModule3(torch.nn.Module):
    def __init__(self):
        super(SimpleModule3, self).__init__()
        self.attn = torch.nn.MultiheadAttention(16, 1, batch_first=True)

    def forward(self, q, k, v):
        x, _ = self.attn(q, k, v)
        return x


class NOPLayer(paddle.nn.Layer):
    def __init__(self):
        super(NOPLayer, self).__init__()

    def forward(self, x):
        return x


class SimpleLayer4(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer4, self).__init__()
        self.nop = NOPLayer()
        self.linear = paddle.nn.Linear(100, 10)

    def forward(self, x):
        x = self.nop(x)
        x = self.linear(x)
        return x


class SimpleModule4(torch.nn.Module):
    def __init__(self):
        super(SimpleModule4, self).__init__()
        self.linear = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.linear(x)
        return x


class SimpleLayer5(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer5, self).__init__()
        self.conv = paddle.nn.Conv2D(3, 32, 3, padding=1)
        self.bn = paddle.nn.BatchNorm2D(32)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SimpleModule5(torch.nn.Module):
    def __init__(self):
        super(SimpleModule5, self).__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class TestCaseName(unittest.TestCase):
    def test_layer_map_1(self):
        layer = SimpleLayer1(4, 8, 4)
        module = SimpleModule1(4, 8, 4)

        inp = paddle.to_tensor([[1, 2, 0, 1]]).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        assert (
            auto_diff(layer, module, inp, auto_weights=True, layer_map={}, options={"atol": 1e-4}) is True
        ), "Failed. expected success."

    def test_layer_map_2(self):
        layer = SimpleLayer2()
        module = SimpleModule2()

        layer_map = {layer.lstm: module.lstm}

        inp = paddle.to_tensor([[1] * 9]).numpy().astype("int64")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        assert (
            auto_diff(layer, module, inp, auto_weights=True, layer_map=layer_map, options={"atol": 1e-4}) is True
        ), "Failed. expected success."

    def test_layer_map_3(self):
        layer = SimpleLayer3()
        module = SimpleModule3()

        layer_map = {layer.attn: module.attn}

        inp = paddle.rand((2, 4, 16)).numpy()
        inp = (
            {"q": paddle.to_tensor(inp), "k": paddle.to_tensor(inp), "v": paddle.to_tensor(inp)},
            {"q": torch.as_tensor(inp), "k": torch.as_tensor(inp), "v": torch.as_tensor(inp)},
        )

        assert (
            auto_diff(layer, module, inp, auto_weights=True, layer_map=layer_map, options={"atol": 1e-4}) is True
        ), "Failed. expected success."

    def test_layer_map_4(self):
        layer = SimpleLayer4()
        module = SimpleModule4()

        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})

        # ignore wrap layers automatically now
        assert auto_diff(layer, module, inp, auto_weights=True, options={"atol": 1e-4})

        # layer_map = LayerMap()
        # layer_map.ignore(layer.nop)

        # assert auto_diff(
        #     layer, module, inp, auto_weights=True, layer_map=layer_map, options={"atol": 1e-4}
        # ), "Failed. expected success."

        # with self.assertRaises(Exception, msg="Sucess, expected exception."):
        # auto_diff(layer, module, inp, auto_weights=True, options={"atol": 1e-4})

    def test_layer_map_5(self):
        layer = SimpleLayer5()
        module = SimpleModule5()
        layer.eval()
        module.eval()
        layer_map = {layer.bn: module.bn}

        inp = paddle.rand((1, 3, 32, 32)).numpy()
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})

        assert (
            auto_diff(layer, module, inp, auto_weights=True, layer_map=layer_map, options={"atol": 1e-4}) is True
        ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
