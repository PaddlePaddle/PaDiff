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
from paddle.io import Dataset, BatchSampler, DataLoader
import unittest
import paddle
import torch
    
class SimpleLayer(paddle.nn.Layer):
    def __init__(self, input_size):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(input_size, input_size)
        self.linear2 = paddle.nn.Linear(input_size, 10)
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x
    
class TestCheckApi(unittest.TestCase):
    def setUp(self):
        # set args for test
        self.seed = 2023
        self.dump_per_step = 2
        self.dataset_size = 10
        self.batch_size = 1
        self.forward_times = 10
        self.input_size = 100

        # set seed for paddle and torch
        paddle.seed(self.seed)
        torch.manual_seed(self.seed) 
        self.input = paddle.randn((self.batch_size, self.input_size))

        # init model and dataloader
        self.proxy_layer_amp = create_model(SimpleLayer(self.input_size), name="layer_amp")
        self.optimizer_amp = paddle.optimizer.SGD(learning_rate=0.01, parameters=self.proxy_layer_amp.model.parameters())
        self.proxy_layer_naive = create_model(SimpleLayer(self.input_size), name="layer_naive")
        self.optimizer_naive = paddle.optimizer.SGD(learning_rate=0.01, parameters=self.proxy_layer_naive.model.parameters())


class TestCheckSuccess(TestCheckApi):
    def test(self):
        assign_weight(self.proxy_layer_amp, self.proxy_layer_naive)
        for i in range(self.forward_times):
            
            # 因为使用scale会使得梯度值增大，要在scale下进行精度对齐，有待继续开发
            # scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            
            with paddle.amp.auto_cast(enable=True, custom_white_list={'elementwise_add'}):
                out = self.proxy_layer_amp(self.input)
                loss = out.mean()
            self.proxy_layer_amp.backward(loss)
            
            self.proxy_layer_amp.try_dump(self.dump_per_step)

            self.optimizer_amp.step()
            self.optimizer_amp.clear_grad()

        for i in range(self.forward_times):
            out = self.proxy_layer_naive(self.input)
            loss = out.mean()
            self.proxy_layer_naive.backward(loss)

            self.proxy_layer_naive.try_dump(self.dump_per_step)

            self.optimizer_naive.step()
            self.optimizer_naive.clear_grad()


        for i in range(0, self.forward_times, self.dump_per_step * self.batch_size):
            assert check_report(self.proxy_layer_amp.dump_path + f"/step_{i}", 
                                self.proxy_layer_naive.dump_path + f"/step_{i}",
                                cfg={"atol": 1e-4, "rtol": 1e-3, "compare_mode": "mean"}) == True
            assert check_params(self.proxy_layer_amp.dump_path + f"/step_{i}", 
                                self.proxy_layer_naive.dump_path + f"/step_{i}",
                                cfg={"atol": 1e-4, "rtol": 1e-3, "compare_mode": "mean"}) == True

class TestCheckFail(TestCheckApi):
    def test(self):
        for i in range(self.forward_times):
            with paddle.amp.auto_cast(enable=True, custom_white_list={'elementwise_add'}):
                out = self.proxy_layer_amp(self.input)
                loss = out.mean()
            self.proxy_layer_amp.backward(loss)
            self.proxy_layer_amp.try_dump(self.dump_per_step)

        for i in range(self.forward_times):
            out = self.proxy_layer_naive(self.input)
            loss = out.mean()
            self.proxy_layer_naive.backward(loss)
            self.proxy_layer_naive.try_dump(self.dump_per_step)

        for i in range(0, self.forward_times, self.dump_per_step * self.batch_size):
            assert check_report(self.proxy_layer_amp.dump_path + f"/step_{i}", 
                                self.proxy_layer_naive.dump_path + f"/step_{i}",
                                cfg={"atol": 1e-4, "rtol": 1e-3, "compare_mode": "mean"}) == False
            assert check_params(self.proxy_layer_amp.dump_path + f"/step_{i}", 
                                self.proxy_layer_naive.dump_path + f"/step_{i}",
                                cfg={"atol": 1e-4, "rtol": 1e-3, "compare_mode": "mean"}) == False

if __name__ == "__main__":
    unittest.main()