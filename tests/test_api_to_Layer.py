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

from padiff.trainer.trainer_utils import Report
from padiff.trainer import Trainer
from padiff.utils import init_options, LayerMap
import paddle
import torch


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.relu = paddle.nn.functional.relu

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = x * 2
        x = x + 1
        return x


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = x * 2
        x = x + 1
        return x


class TestCaseName(unittest.TestCase):
    def test_api_to_Layer(self):
        layer = SimpleLayer()
        module = SimpleModule()
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
        options = {}
        init_options(options)

        paddle_report = Report("paddle")
        torch_report = Report("torch")
        trainer = Trainer(layer, module, None, None, LayerMap(), options)

        trainer.do_run(paddle_report, torch_report, inp)

        # [layer(SimpleLayer, Linear) + api(linear, relu) + method(mul, add)] * (fwd, bwd) = 12
        assert len(paddle_report.items) == 12
        assert len(torch_report.items) == 12


if __name__ == "__main__":
    unittest.main()
