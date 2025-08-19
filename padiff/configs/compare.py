# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


global_compare_configs = {
    "atol": 0,
    "rtol": 1e-7,
    "compare_mode": "mean",
    "act_name": "equal",
}


def update_configs(cfg):
    global global_compare_configs
    assert isinstance(cfg, dict)
    for k in cfg.keys():
        assert k in global_compare_configs
    global_compare_configs.update(cfg)
    return global_compare_configs


def parse_cfg(cfg):
    global global_compare_configs
    if cfg is None:
        return global_compare_configs
    return update_configs(cfg)
