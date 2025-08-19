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


class SimplePaddle(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.dropout = paddle.nn.Dropout(p=0.1)
        self.linear2 = paddle.nn.Linear(100, 10)
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x


class SimpleTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(100, 100)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.linear2 = torch.nn.Linear(100, 10)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x


def warpped_fn(model, x):
    return model(x)


def run_paddle(warpped=False):
    """ """
    model = SimplePaddle()
    pd_model = create_model(model, name=f"model_PD")
    inp = paddle.rand((100, 100))
    with PaDiffGuard(pd_model):
        if warpped:
            out = warpped_fn(model, inp)
        else:
            out = model(inp)
    dump_report(pd_model, pd_model.dump_path)
    return pd_model.dump_path


def run_torch(weight_path, warpped=False):
    """
    model = SimpleTorch()
    inp = torch.rand((100, 100))
    out = model(inp)
    """
    model = SimpleTorch()
    pt_model = create_model(model, name=f"model_PT")
    load_init_weights_from_dump(weight_path, pt_model)
    loaded_inputs = load_first_input_from_dump(weight_path, tar_framework="torch")
    with PaDiffGuard(pt_model):
        if warpped:
            out = warpped_fn(model, *loaded_inputs)
        else:
            out = model(*loaded_inputs)
    dump_report(pt_model, pt_model.dump_path)
    return pt_model.dump_path


class TestOfflineCompare(unittest.TestCase):
    def test_offline(self):
        pd_path = run_paddle()
        pt_path = run_torch(pd_path)
        report_success = check_report(pd_path, pt_path, cfg={"atol": 1e-4}, diff_phase="forward")
        assert report_success is True, "Failed. expected success."

    def test_offline_wrapped(self):
        pd_path = run_paddle(warpped=True)
        pt_path = run_torch(pd_path, warpped=True)
        report_success = check_report(pd_path, pt_path, cfg={"atol": 1e-4}, diff_phase="forward")
        assert report_success is True, "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
