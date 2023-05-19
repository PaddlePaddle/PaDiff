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

from ...utils import assert_tensor_equal
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

    def find_actions(self, base_model, raw_model):
        for act in self.pool:
            if act.match(base_model, raw_model):
                return act
        raise RuntimeError("No action is matched, not expected.")


global_actions = ActionPool()


def get_action(*args, **kargs):
    return global_actions.find_actions(*args, **kargs)


class Action:
    def match(self, base_model, raw_model):
        raise NotImplementedError("")

    def __call__(self, base_item, raw_item, cfg):
        raise NotImplementedError("")

    @property
    def priority(self):
        raise NotImplementedError("")


@global_actions.register
class EqualAction(Action):
    def match(self, base_model, raw_model):
        if (
            isinstance(base_model, torch.nn.Module)
            and isinstance(raw_model, paddle.nn.Layer)
            or isinstance(raw_model, torch.nn.Module)
            and isinstance(base_model, paddle.nn.Layer)
        ):
            return True
        return False

    @property
    def priority(self):
        return 0

    def __call__(self, base_item, raw_item, cfg):
        tensors_0 = base_item.tensors_for_compare()
        tensors_1 = raw_item.tensors_for_compare()
        for (t0,), (t1,) in zip(tensors_0, tensors_1):
            if t0.numel() == 0 or t1.numel() == 0:
                warnings.warn("Found Tensor.numel() is 0, compare skipped!")
                continue
            assert_tensor_equal(t0.detach().cpu().numpy(), t1.detach().cpu().numpy(), cfg)


@global_actions.register
class PPAction(Action):
    def match(self, base_model, raw_model):
        try:
            assert isinstance(base_model, paddle.nn.Layer)
            assert isinstance(raw_model, paddle.nn.Layer)
        except:
            return False
        return True

    @property
    def priority(self):
        return 1

    def __call__(self, base_item, raw_item, cfg):
        tensors_0 = base_item.tensors_for_compare()
        tensors_1 = raw_item.tensors_for_compare()
        for (t0,), (t1,) in zip(tensors_0, tensors_1):
            if t0.numel() == 0 or t1.numel() == 0:
                warnings.warn("Found Tensor.numel() is 0, compare skipped!")
                continue
            assert_tensor_equal(t0.detach().cpu().numpy(), t1.detach().cpu().numpy(), cfg)
