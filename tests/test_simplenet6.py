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

from padiff import autodiff


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.embedder = paddle.nn.Embedding(3, 16)
        self.lstm1 = paddle.nn.LSTMCell(16, 8)
        self.lstm2 = paddle.nn.LSTMCell(8, 8)

    def forward(self, sequence):
        inputs = self.embedder(sequence)
        encoder_output, _ = self.lstm1(inputs)
        encoder_output, _ = self.lstm2(encoder_output)
        return encoder_output


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.embedder = torch.nn.Embedding(3, 16)
        self.lstm = torch.nn.LSTM(
            input_size=16,
            hidden_size=8,
            num_layers=2,
        )

    def forward(self, sequence):
        inputs = self.embedder(sequence)
        encoder_output, _ = self.lstm(inputs)
        return encoder_output


class TestCaseName(unittest.TestCase):
    def test_success(self):
        layer = SimpleLayer()
        module = SimpleModule()

        layer_module_map = {
            module.embedder: layer.embedder,
            module.lstm: [layer.lstm1, layer.lstm2],
        }

        inp = paddle.to_tensor([[1] * 9]).numpy().astype("int64")
        assert (
            autodiff(
                layer,
                module,
                inp,
                auto_weights=True,
                layer_module_map=layer_module_map,
                options={"atol": 1e-5},
            )
            is True
        ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
