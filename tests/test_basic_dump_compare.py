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


class TestCaseName(unittest.TestCase):
    def test_check_success(self):
        layer = SimpleLayer()
        layer = create_model(layer, dump_freq=2)
        module = SimpleModule()
        module = create_model(module, dump_freq=2)

        assign_weight(layer, module)

        inp = paddle.rand((100, 100)).numpy().astype("float32")

        for i in range(6):
            out = layer(paddle.to_tensor(inp))
            loss = out.mean()
            layer.backward(loss)
            layer.try_dump()

            out = module(torch.as_tensor(inp))
            loss = out.mean()
            module.backward(loss)
            module.try_dump()

            if i % 2 == 0:
                assert (
                    check_report(layer.dump_path + f"/step_{i}", module.dump_path + f"/step_{i}", cfg={"atol": 1e-4})
                    == True
                )
                assert (
                    check_params(layer.dump_path + f"/step_{i}", module.dump_path + f"/step_{i}", cfg={"atol": 1e-4})
                    == True
                )

    def test_check_fail(self):
        layer = SimpleLayer()
        layer = create_model(layer, dump_freq=2)
        module = SimpleModule()
        module = create_model(module, dump_freq=2)

        inp = paddle.rand((100, 100)).numpy().astype("float32")

        for i in range(6):
            out = layer(paddle.to_tensor(inp))
            loss = out.mean()
            layer.backward(loss)
            layer.try_dump()

            out = module(torch.as_tensor(inp))
            loss = out.mean()
            module.backward(loss)
            module.try_dump()

            try:
                if i % 2 == 0:
                    assert (
                        check_report(
                            layer.dump_path + f"/step_{i}", module.dump_path + f"/step_{i}", cfg={"atol": 1e-4}
                        )
                        == False
                    )
                    assert (
                        check_params(
                            layer.dump_path + f"/step_{i}", module.dump_path + f"/step_{i}", cfg={"atol": 1e-4}
                        )
                        == False
                    )
            except Exception as e:
                print(e)


if __name__ == "__main__":
    unittest.main()
