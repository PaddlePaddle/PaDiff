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

from ..utils import assert_tensor_equal, load_numpy, logger
import numpy as np


class ActionPool:
    def __init__(self):
        self.pool = {}
        self._ordered_pool = []

    def register(self, name=None):
        def decorator(cls):
            final_name = cls.__name__ if name is None else name
            self.pool[final_name] = cls()
            self._ordered_pool = sorted(self.pool.values(), key=lambda x: x.priority, reverse=True)  # high -> low
            return cls

        if callable(name):
            cls = name
            name = None
            return decorator(cls)
        else:
            return decorator

    def get_action_by_name(self, name):
        if name not in self.pool:
            raise ValueError(f"Action '{name}' not registered. Available: {list(self.pool.keys())}")
        return self.pool[name]

    def find_actions(self, report_0, node_0, report_1, node_1, name=None):
        if name is not None:
            return self.get_action_by_name(name)

        for act in self._ordered_pool:
            if act.match(report_0, node_0, report_1, node_1):
                return act
        raise RuntimeError("No action is matched, not expected.")


global_actions = ActionPool()


def get_action(*args, **kargs):
    return global_actions.find_actions(*args, **kargs)


class Action:
    def match(self, report_0, node_0, report_1, node_1):
        raise NotImplementedError("")

    def __call__(self, base_item, raw_item, cfg):
        raise NotImplementedError("")

    @property
    def priority(self):
        raise NotImplementedError("")


@global_actions.register("equal")
class EqualAction(Action):
    def match(self, report_0, node_0, report_1, node_1):
        return True

    @property
    def priority(self):
        return 0

    def __call__(self, file_list_0, file_list_1, cfg):
        assert len(file_list_0) == len(
            file_list_1
        ), f"number of tensors for compare is not equal, {len(file_list_0)} vs {len(file_list_1)}"
        for info_0, info_1 in zip(file_list_0, file_list_1):
            tensor_0 = load_numpy(info_0["path"])
            tensor_1 = load_numpy(info_1["path"])
            if tensor_0.size == 0 or tensor_1.size == 0:
                if tensor_0.size != tensor_1.size:
                    raise RuntimeError("size of tensors is not equal")
                logger.warning("Found nparray.size == 0, compare skipped!")
                continue
            assert_tensor_equal(tensor_0, tensor_1, cfg)


@global_actions.register("loose_equal")
class LooseEqualAction(Action):
    def match(self, report_0, node_0, report_1, node_1):
        return True

    @property
    def priority(self):
        return 1

    def __call__(self, file_list_0, file_list_1, cfg):
        len_fl_0, len_fl_1 = len(file_list_0), len(file_list_1)
        if len_fl_0 != len_fl_1:
            logger.warning(f"number of tensors for compare is not equal, {len_fl_0} vs {len_fl_1}")

        min_len = min(len_fl_0, len_fl_1)

        num_success = 0
        for info_0, info_1 in zip(file_list_0[:min_len], file_list_1[:min_len]):
            tensor_0 = load_numpy(info_0["path"])
            tensor_1 = load_numpy(info_1["path"])
            if tensor_0.size == 0 or tensor_1.size == 0:
                logger.debug("Found empty tensor, compare skipped!")
                continue
            if tensor_0.shape != tensor_1.shape:
                logger.debug(f"Shape of tensors are not equal: {tensor_0.shape}!={tensor_1.shape}")
                if tensor_0.size == tensor_1.size:
                    logger.debug(f"Try to reshape them to {tensor_0.shape}")
                    tensor_1 = np.reshape(tensor_1, tensor_0.shape)
                else:
                    continue
            assert_tensor_equal(tensor_0, tensor_1, cfg)
            num_success += 1
        if min_len != 0 and num_success == 0:
            raise RuntimeError("All outputs for the layer have different shape!")
