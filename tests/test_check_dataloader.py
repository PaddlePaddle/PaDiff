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

from padiff import check_dataloader

datas = [paddle.rand((100, 100)).numpy().astype("float32") for i in range(10)]


def paddle_loader():
    for data in datas:
        yield paddle.to_tensor(data)


def torch_loader():
    for data in datas:
        yield torch.as_tensor(data)


def torch_loader_with_diff():
    for idx, data in enumerate(datas):
        if idx == 2:
            yield torch.as_tensor(data + 1)
        else:
            yield torch.as_tensor(data)


class TestCaseName(unittest.TestCase):
    def test_check_dataloader(self):
        assert check_dataloader(paddle_loader(), torch_loader()) == True
        assert check_dataloader(paddle_loader(), torch_loader_with_diff()) == False


if __name__ == "__main__":
    unittest.main()
