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

from .. import utils
import torch
import paddle

import warnings


class ActionPool:
    def __init__(self):
        self.pool = []

    def register(self, cls):
        name = cls.__name__
        self.pool.append(cls())
        sorted(self.pool, key=lambda x: x.priority, reverse=True)
        return cls

    def find_actions(self, torch_net, paddle_net):
        for act in self.pool:
            if act.match(torch_net, paddle_net):
                return act
        raise RuntimeError("No action is matched, not expected.")


global_actions = ActionPool()


def get_action(*args, **kargs):
    return global_actions.find_actions(*args, **kargs)


class Action:
    def match(self, torch_net, paddle_net):
        raise NotImplementedError("")

    def __call__(self, torch_item, paddle_item, cfg):
        raise NotImplementedError("")

    @property
    def priority(self):
        raise NotImplementedError("")


@global_actions.register
class EqualAction(Action):
    def match(self, torch_net, paddle_net):
        two_net = isinstance(torch_net, torch.nn.Module) and isinstance(paddle_net, paddle.nn.Layer)
        # if two_str, it should be two api layer
        two_str = isinstance(torch_net, str) and isinstance(paddle_net, str)
        if two_net or two_str:
            return True
        else:
            return False

    @property
    def priority(self):
        return 0

    def __call__(self, torch_item, paddle_item, cfg):
        """
        NOTE:
        """
        is_debug = cfg["debug"]
        torch_tensors = torch_item.compare_tensors()
        paddle_tensors = paddle_item.compare_tensors()
        for (tt,), (pt,) in zip(torch_tensors, paddle_tensors):
            if tt.numel() == 0 or pt.numel() == 0:
                warnings.warn("Found Tensor shape is [0], compare skipped!")
                continue
            try:
                utils.assert_tensor_equal(tt.detach().cpu().numpy(), pt.numpy(), cfg)
            except Exception as e:
                if is_debug:
                    print("Mean of inputs:")
                    print(torch_item.input[0].numpy().mean())
                    print(paddle_item.input[0].numpy().mean())
                    import pdb

                    pdb.set_trace()
                raise e
