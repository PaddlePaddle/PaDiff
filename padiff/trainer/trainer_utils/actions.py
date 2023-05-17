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

    def find_actions(self, model_0, model_1):
        for act in self.pool:
            if act.match(model_0, model_1):
                return act
        raise RuntimeError("No action is matched, not expected.")


global_actions = ActionPool()


def get_action(*args, **kargs):
    return global_actions.find_actions(*args, **kargs)


class Action:
    def match(self, model_0, model_1):
        raise NotImplementedError("")

    def __call__(self, item_0, item_1, cfg):
        raise NotImplementedError("")

    @property
    def priority(self):
        raise NotImplementedError("")


@global_actions.register
class EqualAction(Action):
    def match(self, model_0, model_1):
        try:
            assert isinstance(model_0, paddle.nn.Layer)
            assert isinstance(model_1, torch.nn.Module)
        except:
            return False
        return True

    @property
    def priority(self):
        return 0

    def __call__(self, item_0, item_1, cfg):
        """
        NOTE:
        """
        is_debug = cfg["debug"]
        tensors_0 = item_0.tensors_for_compare()
        tensors_1 = item_1.tensors_for_compare()
        for (t0,), (t1,) in zip(tensors_0, tensors_1):
            if t0.numel() == 0 or t1.numel() == 0:
                warnings.warn("Found Tensor shape is [0], compare skipped!")
                continue
            try:
                utils.assert_tensor_equal(t0.detach().cpu().numpy(), t1.detach().cpu().numpy(), cfg)
            except Exception as e:
                if is_debug:
                    print("Mean of inputs:")
                    print(item_0.input[0].numpy().mean())
                    print(item_1.input[0].numpy().mean())
                    import pdb

                    pdb.set_trace()
                raise e


@global_actions.register
class PPAction(Action):
    def match(self, model_0, model_1):
        try:
            assert isinstance(model_0, paddle.nn.Layer)
            assert isinstance(model_1, paddle.nn.Layer)
        except:
            return False
        return True

    @property
    def priority(self):
        return 1

    def __call__(self, item_0, item_1, cfg):
        """
        NOTE:
        """
        is_debug = cfg["debug"]
        tensors_0 = item_0.tensors_for_compare()
        tensors_1 = item_1.tensors_for_compare()
        for (t0,), (t1,) in zip(tensors_0, tensors_1):
            if t0.numel() == 0 or t1.numel() == 0:
                warnings.warn("Found Tensor shape is [0], compare skipped!")
                continue
            try:
                utils.assert_tensor_equal(t0.detach().cpu().numpy(), t1.detach().cpu().numpy(), cfg)
            except Exception as e:
                if is_debug:
                    print("Mean of inputs:")
                    print(item_0.input[0].numpy().mean())
                    print(item_1.input[0].numpy().mean())
                    import pdb

                    pdb.set_trace()
                raise e
