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

from padiff import auto_diff, LayerMap
from padiff.padiff_abstracts import padiff_model
from padiff.utils import reset_log_dir, init_options
from padiff.trainer.Checker import check_grad, check_weight


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
    def test_weight_grad_check(self):
        layer = SimpleLayer()
        module = SimpleModule()
        options = {"atol": 1e-4}
        init_options(options)

        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        assert auto_diff(layer, module, inp, auto_weights=True, options=options) is True, "Failed. expected success."

        module.zero_grad()
        reset_log_dir()

        weight_check = check_weight((padiff_model(layer), padiff_model(module)), options, LayerMap())
        grad_check = check_grad((padiff_model(layer), padiff_model(module)), options, LayerMap())
        assert weight_check is True, "Weight params should be same"
        assert grad_check is False, "Grad should be different"

        layer = SimpleLayer()
        module = SimpleModule()
        assert auto_diff(layer, module, inp, auto_weights=True, options=options) is True, "Failed. expected success."

        for param in module.parameters():
            param.data = param * 2
        reset_log_dir()
        weight_check = check_weight((padiff_model(layer), padiff_model(module)), options, LayerMap())
        grad_check = check_grad((padiff_model(layer), padiff_model(module)), options, LayerMap())
        assert weight_check is False, "Weight params should be different"
        assert grad_check is True, "Grad should be same"


if __name__ == "__main__":
    unittest.main()
