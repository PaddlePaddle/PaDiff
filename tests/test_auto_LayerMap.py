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
from padiff import auto_diff, LayerMap
import paddle
import torch


class SimpleLayer2(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer2, self).__init__()
        self.embedder = paddle.nn.Embedding(3, 16)
        self.lstm1 = paddle.nn.LSTM(16, 8, 2, time_major=True)
        self.lstm2 = paddle.nn.LSTM(8, 4, 2, time_major=True)

    def forward(self, x):
        x = self.embedder(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x


class SimpleModule2(torch.nn.Module):
    def __init__(self):
        super(SimpleModule2, self).__init__()
        self.embedder = torch.nn.Embedding(3, 16)
        self.lstm1 = torch.nn.LSTM(
            input_size=16,
            hidden_size=8,
            num_layers=2,
        )
        self.lstm2 = torch.nn.LSTM(
            input_size=8,
            hidden_size=4,
            num_layers=2,
        )

    def forward(self, x):
        x = self.embedder(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x


class TestCaseName(unittest.TestCase):
    def test_auto_layer_map(self):
        layer = SimpleLayer2()
        module = SimpleModule2()

        # layer_map = {layer.lstm1: module.lstm1, layer.lstm2: module.lstm2}
        layer_map = LayerMap.auto(layer, module)

        inp = paddle.to_tensor([[1] * 9]).numpy().astype("int64")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        assert (
            auto_diff(layer, module, inp, auto_weights=True, layer_map=layer_map, options={"atol": 1e-4}) is True
        ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
