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
from padiff.checker import check_grads, check_weights
from padiff.dump_tools import dump_grads, dump_weights
from padiff.interfaces.diff_utils import default_loss


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
    def test_weight_grad_check_0(self):
        layer = create_model(SimpleLayer())
        module = create_model(SimpleModule())

        inp = paddle.rand((100, 100)).numpy().astype("float32")

        assign_weight(layer, module)
        out = layer(paddle.to_tensor(inp))
        loss = default_loss(out, "paddle")
        layer.backward(loss)

        out = module(torch.as_tensor(inp))
        loss = default_loss(out, "torch")
        module.backward(loss)

        module.model.zero_grad()

        dump_weights(layer, layer.dump_path)
        dump_weights(module, module.dump_path)

        dump_grads(layer, layer.dump_path)
        dump_grads(module, module.dump_path)

        weight_check = check_weights(layer.dump_path, module.dump_path)
        grad_check = check_grads(layer.dump_path, module.dump_path)

        assert weight_check is True, "Weight params should be same"
        assert grad_check is False, "Grad should be different"

    def test_weight_grad_check_1(self):
        layer = create_model(SimpleLayer())
        module = create_model(SimpleModule())

        inp = paddle.rand((100, 100)).numpy().astype("float32")

        assign_weight(layer, module)
        out = layer(paddle.to_tensor(inp))
        loss = default_loss(out, "paddle")
        layer.backward(loss)

        out = module(torch.as_tensor(inp))
        loss = default_loss(out, "torch")
        module.backward(loss)

        for param in module.model.parameters():
            param.data = param * 2

        dump_weights(layer, layer.dump_path)
        dump_weights(module, module.dump_path)

        dump_grads(layer, layer.dump_path)
        dump_grads(module, module.dump_path)

        grad_check = check_grads(layer.dump_path, module.dump_path)
        weight_check = check_weights(layer.dump_path, module.dump_path)

        assert weight_check is False, "Weight params should be different"
        assert grad_check is True, "Grad should be same"


if __name__ == "__main__":
    unittest.main()
