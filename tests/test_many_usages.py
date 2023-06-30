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
import os
from paddle.distributed.fleet.utils import recompute

default_path = get_dump_root_path()

train_step = 10


class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.dropout1 = paddle.nn.Dropout()
        self.linear2 = paddle.nn.Linear(100, 10)
        self.dropout2 = paddle.nn.Dropout()
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return paddle.rand((100, 100)).numpy().astype("float32")

    def __len__(self):
        return self.num_samples


class Test0SingleModelRun(unittest.TestCase):
    # single model run
    def test_single_model_run(self):
        print("Test for single model run.")
        layer = SimpleLayer()
        set_dump_root_path(os.path.join(default_path, "single_model_run"))
        layer = create_model(layer, dump_freq=2)
        inp = paddle.rand((100, 100)).numpy().astype("float32")
        opt = paddle.optimizer.SGD(learning_rate=1e-3, parameters=layer.model.parameters())

        for i in range(train_step):
            out = layer(paddle.to_tensor(inp))
            loss = out.mean()
            layer.backward(loss)
            opt.step()
            opt.clear_grad()
            layer.try_dump()

        assert check_report(layer.dump_path, layer.dump_path)
        assert check_params(layer.dump_path, layer.dump_path)


class Test1DataloaderRun(unittest.TestCase):
    # single model run
    # real dataloader
    def test_dataloader_run(self):
        print("Test for real dataloader.")
        layer = SimpleLayer()
        set_dump_root_path(os.path.join(default_path, "real_dataLoader"))
        layer = create_model(layer, dump_freq=2)
        opt = paddle.optimizer.SGD(learning_rate=1e-3, parameters=layer.model.parameters())

        dataset = RandomDataset(train_step)
        loader = paddle.io.DataLoader(dataset)
        for inp in loader():
            out = layer(paddle.to_tensor(inp))
            loss = out.mean()
            layer.backward(loss)
            opt.step()
            opt.clear_grad()
            layer.try_dump()

        assert check_report(layer.dump_path, layer.dump_path)
        assert check_params(layer.dump_path, layer.dump_path)


class Test2WhiteLayerRun(unittest.TestCase):
    # single model run
    # real dataloader
    # use class to update white layer
    def test_white_layer_class_run(self):
        print("Test for single model run.")
        layer = SimpleLayer()
        set_dump_root_path(os.path.join(default_path, "white_layer_class"))
        layer = create_model(layer, dump_freq=2)
        layer.update_white_list_with_class(paddle.nn.Linear, mode="all")
        opt = paddle.optimizer.SGD(learning_rate=1e-3, parameters=layer.model.parameters())

        dataset = RandomDataset(train_step)
        loader = paddle.io.DataLoader(dataset)
        for inp in loader():
            out = layer(paddle.to_tensor(inp))
            loss = out.mean()
            layer.backward(loss)
            opt.step()
            opt.clear_grad()
            layer.try_dump()

        assert check_report(layer.dump_path, layer.dump_path)
        assert check_params(layer.dump_path, layer.dump_path)


class Test3GradAccumulationRun(unittest.TestCase):
    # single model run
    # real dataloader
    # use class to update white layer
    # grad accumulation
    def test_grad_accumulation_run(self):
        print("Test for gradient accumulation.")
        layer = SimpleLayer()
        set_dump_root_path(os.path.join(default_path, "grad_accumulation"))
        layer = create_model(layer, dump_freq=2)
        layer.update_white_list_with_class(paddle.nn.Linear, mode="all")
        opt = paddle.optimizer.SGD(learning_rate=1e-3, parameters=layer.model.parameters())

        dataset = RandomDataset(train_step)
        loader = paddle.io.DataLoader(dataset)
        for step, inp in enumerate(loader()):
            out = layer(paddle.to_tensor(inp))
            loss = out.mean()
            layer.backward(loss)
            if (step + 1) % 2 == 0:
                opt.step()
                opt.clear_grad()
            layer.try_dump()

        assert check_report(layer.dump_path, layer.dump_path)
        assert check_params(layer.dump_path, layer.dump_path)


class Test4RecomputeRun(unittest.TestCase):
    # single model run
    # real dataloader
    # use class to update white layer
    # grad accumulation
    # recompute
    def test_recompute_run(self):
        print("Test for recompute.")
        layer = SimpleLayer()
        set_dump_root_path(os.path.join(default_path, "recompute"))
        layer = create_model(layer, dump_freq=2)
        layer.update_white_list_with_class(paddle.nn.Linear, mode="all")
        opt = paddle.optimizer.SGD(learning_rate=1e-3, parameters=layer.model.parameters())

        dataset = RandomDataset(train_step)
        loader = paddle.io.DataLoader(dataset)
        for step, inp in enumerate(loader()):
            inp = paddle.to_tensor(inp)
            inp.stop_gradient = False
            out = recompute(layer, inp)
            loss = out.mean()
            layer.backward(loss)
            if (step + 1) % 2 == 0:
                opt.step()
                opt.clear_grad()
            layer.try_dump()

        assert check_report(layer.dump_path, layer.dump_path)
        assert check_params(layer.dump_path, layer.dump_path)


class Test5AMPRun(unittest.TestCase):
    # single model run
    # real dataloader
    # use class to update white layer
    # grad accumulation
    # recompute
    # amp
    def test_amp_run(self):
        print("Test for amp.")
        layer = SimpleLayer()
        layer = paddle.amp.decorate(layer, level="O2")
        set_dump_root_path(os.path.join(default_path, "amp"))
        layer = create_model(layer, dump_freq=2)
        layer.update_white_list_with_class(paddle.nn.Linear, mode="all")
        opt = paddle.optimizer.SGD(learning_rate=1e-3, parameters=layer.model.parameters())

        dataset = RandomDataset(train_step)
        loader = paddle.io.DataLoader(dataset)
        for step, inp in enumerate(loader()):
            inp = paddle.to_tensor(inp)
            inp.stop_gradient = False
            with paddle.amp.auto_cast(enable=True, level="O2"):
                out = recompute(layer, inp)
                loss = out.mean()
            layer.backward(loss)
            if (step + 1) % 2 == 0:
                opt.step()
                opt.clear_grad()
            layer.try_dump()

        assert check_report(layer.dump_path, layer.dump_path)
        assert check_params(layer.dump_path, layer.dump_path)


if __name__ == "__main__":
    unittest.main()
