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

import paddle


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


def warpped_fn(model, x):
    return model(x)


def run_paddle(warpped=False):
    model = SimplePaddle()
    inp = paddle.rand((100, 100))
    if warpped:
        out = warpped_fn(model, inp)
    else:
        out = model(inp)


if __name__ == "__main__":
    run_paddle()
    run_paddle(warpped=True)
