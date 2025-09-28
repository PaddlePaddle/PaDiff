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

from .loader import global_json_laoder, global_yaml_loader
from ..utils import logger


global_compare_configs = {
    "atol": 0,
    "rtol": 1e-7,
    "compare_mode": "mean",
    "action_name": "equal",
    "check_mode": "fast",
}


def update_configs(cfg):
    global global_compare_configs
    assert isinstance(cfg, dict), "config should be dict"

    provided_keys = set(cfg.keys())
    valid_keys = set(global_compare_configs.keys())
    invalid_keys = provided_keys - valid_keys
    if invalid_keys:
        logger.warning(f"Invalid config keys ignored: {invalid_keys}. Valid keys are: {valid_keys}")

    valid_updates = {k: v for k, v in cfg.items() if k in valid_keys}
    global_compare_configs.update(valid_updates)
    return global_compare_configs


def parse_cfg(cfg):
    global global_compare_configs
    if cfg is None:
        return global_compare_configs
    return update_configs(cfg)
