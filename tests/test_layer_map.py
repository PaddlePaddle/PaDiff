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

import paddle
import torch

from padiff import auto_diff


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


class TestCaseName(unittest.TestCase):
    def test_layer_map_1(self):
        layer = SimpleLayer1(4, 8, 4)
        module = SimpleModule1(4, 8, 4)

        layer_map = {
            layer.layers: [module.linear1, module.layer_norm, module.linear2],
        }

        inp = paddle.to_tensor([[1, 2, 0, 1]]).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        assert (
            auto_diff(layer, module, inp, auto_weights=True, layer_map=layer_map, options={"atol": 1e-4}) is True
        ), "Failed. expected success."

    def test_layer_map_2(self):
        layer = SimpleLayer2()
        module = SimpleModule2()

        layer_map = {
            module.embedder: layer.embedder,
            module.lstm: layer.lstm,
        }

        inp = paddle.to_tensor([[1] * 9]).numpy().astype("int64")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        assert (
            auto_diff(layer, module, inp, auto_weights=True, layer_map=layer_map, options={"atol": 1e-4}) is True
        ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
