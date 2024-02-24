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


class SimpleLayer1(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer1, self).__init__()
        self.embedder = paddle.nn.Embedding(3, 16)
        self.lstm = paddle.nn.LSTM(16, 8, 2, time_major=True)

    def forward(self, x):
        x = self.embedder(x)
        x, _ = self.lstm(x)
        return x


class SimpleModule1(torch.nn.Module):
    def __init__(self):
        super(SimpleModule1, self).__init__()
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


class SimpleLayer2(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer2, self).__init__()
        self.attn = paddle.nn.MultiHeadAttention(16, 1)

    def forward(self, q, k, v):
        x = self.attn(q, k, v)
        return x


class SimpleModule2(torch.nn.Module):
    def __init__(self):
        super(SimpleModule2, self).__init__()
        self.attn = torch.nn.MultiheadAttention(16, 1, batch_first=True)

    def forward(self, q, k, v):
        x, _ = self.attn(q, k, v)
        return x


class NOPLayer(paddle.nn.Layer):
    def __init__(self):
        super(NOPLayer, self).__init__()

    def forward(self, x):
        return x


class SimpleLayer3(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer3, self).__init__()
        self.conv = paddle.nn.Conv2D(3, 32, 3, padding=1)
        self.bn = paddle.nn.BatchNorm2D(32)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SimpleModule3(torch.nn.Module):
    def __init__(self):
        super(SimpleModule3, self).__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class TestCaseName(unittest.TestCase):
    # def test_layer_map_1(self):
    #     layer = create_model(SimpleLayer1())
    #     module = create_model(SimpleModule1())

    #     module.set_layer_map([module.model.lstm])
    #     layer.set_layer_map([layer.model.lstm])

    #     inp = paddle.to_tensor([[1] * 9]).numpy().astype("int64")
    #     inp = ({"x": torch.as_tensor(inp)}, {"x": paddle.to_tensor(inp)})
    #     assert auto_diff(module, layer, inp, atol=1e-4) is True, "Failed. expected success."

    def test_layer_map_2(self):
        layer = create_model(SimpleLayer2())
        module = create_model(SimpleModule2())

        layer.set_layer_map([layer.model.attn])
        module.set_layer_map([module.model.attn])

        inp = paddle.rand((2, 4, 16)).numpy()
        inp = (
            {"q": torch.as_tensor(inp), "k": torch.as_tensor(inp), "v": torch.as_tensor(inp)},
            {"q": paddle.to_tensor(inp), "k": paddle.to_tensor(inp), "v": paddle.to_tensor(inp)},
        )

        assert auto_diff(module, layer, inp, atol=1e-4) is True, "Failed. expected success."

    # def test_layer_map_3(self):
    #     layer = SimpleLayer3()
    #     module = SimpleModule3()

    #     layer.eval()
    #     module.eval()

    #     layer = create_model(layer)
    #     module = create_model(module)

    #     module.set_layer_map([module.model.bn])
    #     layer.set_layer_map([layer.model.bn])

    #     inp = paddle.rand((1, 3, 32, 32)).numpy()
    #     inp = ({"x": torch.as_tensor(inp)}, {"x": paddle.to_tensor(inp)})

    #     assert auto_diff(module, layer, inp, atol=1e-4) is True, "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
