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

from padiff.Trainer import _register_torch_hooker
from padiff.report import Report, report_guard
from padiff.utils import TreeView, init_options, LayerMap


class SimpleSubModule(torch.nn.Module):
    def __init__(self):
        super(SimpleSubModule, self).__init__()
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


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)
        self.linear2 = torch.nn.Linear(100, 10)
        self.simple = SimpleSubModule()
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
        x = x + self.simple(resdual)
        return x


class TestCaseName(unittest.TestCase):
    def test_add(self):
        option = {}
        init_options(option)
        torch_report = Report("torch")
        paddle_report = Report("Paddle")
        module = SimpleModule()
        example_inp = paddle.rand((100, 100)).numpy().astype("float32")
        with report_guard(torch_report=torch_report, paddle_report=paddle_report):
            with _register_torch_hooker(module, LayerMap()):
                try:
                    torch_input = torch.as_tensor(example_inp)
                    torch_input.requires_grad = True
                    torch_output = module(torch_input)
                    loss = torch_output.mean()
                    loss.backward()
                except Exception as e:
                    raise RuntimeError(
                        "Exception is thrown while running forward of torch_module, please check the legality of module.\n{}".format(
                            str(e)
                        )
                    )
        treeview = TreeView(torch_report.get_fwd_items())
        ans1 = []
        ans2 = []
        for item in treeview.traversal_forward():
            print("===========")
            print(item.net)
            print(item.net_id)
            ans1.append(item.net_id)
        for item in treeview.traversal_backward():
            print("===========")
            print(item.net)
            print(item.net_id)
            ans2.append(item.net_id)
        """
        tree is :
        0 - 1
            7
            2
            3 - 4
                6
                5
        """
        assert ans1 == [1, 7, 2, 4, 6, 5, 3, 0], "Wrong order {}.".format(ans1)
        assert ans2 == [5, 6, 4, 3, 2, 7, 1, 0], "Wrong order {}.".format(ans2)


if __name__ == "__main__":
    unittest.main()
