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
from functools import partial


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.linear2 = paddle.nn.Linear(100, 10)
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        """
        x -> linear1 -> x -> relu -> x -> add -> linear2 -> output
        |                                  |
        |----------------------------------|
        """
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
        """
        x -> linear1 -> x -> relu -> x -> add -> linear2 -> output
        |                                  |
        |----------------------------------|
        """
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x


class TestCaseName(unittest.TestCase):
    def test_loss_fn(self):
        layer = create_model(SimpleLayer())
        module = create_model(SimpleModule())
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        label = paddle.rand([10]).numpy().astype("float32")

        def paddle_loss(inp, label):
            label = paddle.to_tensor(label)
            return inp.mean() - label.mean()

        def torch_loss(inp, label):
            label = torch.tensor(label)
            return inp.mean() - label.mean()

        assert (
            auto_diff(
                layer,
                module,
                inp,
                loss_fn=[partial(paddle_loss, label=label), partial(torch_loss, label=label)],
                atol=1e-4,
            )
            is True
        ), "Failed. expected success."

        paddle_mse = paddle.nn.MSELoss()
        torch_mse = torch.nn.MSELoss()

        assert (
            auto_diff(
                layer,
                module,
                inp,
                loss_fn=[
                    partial(paddle_mse, label=paddle.to_tensor(label)),
                    partial(torch_mse, target=torch.tensor(label)),
                ],
                atol=1e-4,
            )
            is True
        ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
