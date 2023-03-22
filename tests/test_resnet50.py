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

import os

os.environ["PADIFF_API_CHECK"] = "OFF"

from padiff import auto_diff
import unittest

import paddle
import torch
import torchvision


class TestCaseName(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_success(self):
        paddle.set_device("cpu")
        layer = paddle.vision.resnet50()
        module = torchvision.models.resnet50().to("cpu")
        inp = paddle.rand((10, 3, 224, 224)).numpy().astype("float32")
        inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp).to("cpu")})
        assert (
            auto_diff(layer, module, inp, auto_weights=True, options={"atol": 5e-5}) is True
        ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
