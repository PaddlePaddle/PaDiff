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

from padiff import *
import unittest
import paddle
import torch


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


class TestOfflineCompare(unittest.TestCase):
    def test_check_success(self):
        layer = SimpleLayer()
        layer.eval()
        layer = create_model(layer, "layer")
        model = SimpleModule()
        model.eval()
        model = create_model(model, "model")
        assign_weight(model, layer)
        inp = paddle.rand((100, 100)).numpy()
        inp = ({"x": torch.as_tensor(inp)}, {"x": paddle.to_tensor(inp)})
        assert auto_diff(model, layer, inp, atol=1e-4) is True, "Failed. expected success."

    def test_check_fail(self):
        layer = SimpleLayer()
        layer.eval()
        layer = create_model(layer, "layer")
        model = SimpleModule()
        model.eval()
        model = create_model(model, "model")
        assign_weight(model, layer)
        inp = paddle.rand((100, 100)).numpy()
        inp_err = paddle.rand((100, 100)).numpy()
        inp = ({"x": torch.as_tensor(inp)}, {"x": paddle.to_tensor(inp_err)})
        assert auto_diff(model, layer, inp, atol=1e-4) is False, "Success. expected failed."


if __name__ == "__main__":
    unittest.main()
