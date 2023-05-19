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

from padiff import auto_diff

"""
测试 不同的 `forward顺序`，但是具有同样的 `定义顺序`

期待结果：
Success
"""


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.linear2 = paddle.nn.Linear(100, 100)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        return x1 + x2


class SimplleDIffLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimplleDIffLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.linear2 = paddle.nn.Linear(100, 100)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        x2 = self.linear2(x)
        x2 = self.relu(x2)
        x1 = self.linear1(x)
        return x2 + x1


class TestCaseName(unittest.TestCase):
    def test_check_weight_grad(self):
        model_0 = SimpleLayer()
        model_1 = SimpleLayer()
        model_2 = SimplleDIffLayer()

        inp = paddle.rand((100, 100)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": paddle.to_tensor(inp)})
        options = {"atol": 1e-4, "single_step": True, "model_names": ["paddle1", "paddle2"], "auto_init": True}
        assert auto_diff(model_0, model_1, inp, **options) is True
        assert auto_diff(model_0, model_2, inp, **options) is False


if __name__ == "__main__":
    unittest.main()
