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
from paddle.io import DataLoader, Dataset
import unittest
import paddle
import torch


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.data_list = [paddle.rand((100, 100)) for i in range(num_samples)]

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return self.num_samples


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


class TestDataloaderApi(unittest.TestCase):
    def setUp(self):
        # set args for test
        self.seed = 2023
        self.dump_per_step = 2
        self.dataset_size = 10
        self.batch_size = 1

        # set seed for paddle and torch
        paddle.seed(self.seed)
        torch.manual_seed(self.seed)

        # init model and dataloader
        self.layer = SimpleLayer()
        self.layer = create_model(self.layer)
        self.module = SimpleModule()
        self.module = create_model(self.module)
        self.dataset = RandomDataset(self.dataset_size)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, drop_last=True)


class TestCheckSuccess(TestDataloaderApi):
    def test(self):
        assign_weight(self.layer, self.module)

        # implement fwd and bwd separately
        for i, data in enumerate(self.dataloader()):
            paddle_input = data
            out = self.layer(paddle_input)
            loss = out.mean()
            self.layer.backward(loss)
            self.layer.try_dump(self.dump_per_step)

        for i, data in enumerate(self.dataloader()):
            # convert paddle tensor to torch tensor
            torch_input = torch.as_tensor(data.numpy().astype("float32"))
            out = self.module(torch_input)
            loss = out.mean()
            self.module.backward(loss)
            self.module.try_dump(self.dump_per_step)

        for i in range(0, len(self.dataloader), self.dump_per_step * self.batch_size):
            assert check_report(self.layer.dump_path + f"/step_{i}", self.module.dump_path + f"/step_{i}") == True
            assert check_params(self.layer.dump_path + f"/step_{i}", self.module.dump_path + f"/step_{i}") == True


class TestCheckFail(TestDataloaderApi):
    def test(self):

        # implement fwd and bwd separately
        for i, data in enumerate(self.dataloader()):
            paddle_input = data
            out = self.layer(paddle_input)
            loss = out.mean()
            self.layer.backward(loss)
            self.layer.try_dump(self.dump_per_step)

        for i, data in enumerate(self.dataloader()):
            # convert paddle tensor to torch tensor
            torch_input = torch.as_tensor(data.numpy().astype("float32"))
            out = self.module(torch_input)
            loss = out.mean()
            self.module.backward(loss)
            self.module.try_dump(self.dump_per_step)

        for i in range(0, len(self.dataloader), self.dump_per_step * self.batch_size):
            assert check_report(self.layer.dump_path + f"/step_{i}", self.module.dump_path + f"/step_{i}") == False
            assert check_params(self.layer.dump_path + f"/step_{i}", self.module.dump_path + f"/step_{i}") == False


if __name__ == "__main__":
    unittest.main()
