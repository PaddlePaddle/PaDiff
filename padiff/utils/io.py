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

import os


def _get_all_rank_path(path):
    all_files = sorted(os.listdir(path))
    all_files_check = ["rank_" in file for file in all_files]
    if all(all_files_check):
        return [os.path.join(path, file) for file in all_files]
    else:
        return [path]


def _get_all_step_path(path):
    all_files = sorted(os.listdir(path))
    all_files_check = ["step_" in file for file in all_files]
    if all(all_files_check):
        return [os.path.join(path, file) for file in all_files]
    else:
        return [path]


def get_all_valid_path(report_path_0, report_path_1):
    all_steps_path_0 = _get_all_step_path(report_path_0)
    all_steps_path_1 = _get_all_step_path(report_path_1)
    assert len(all_steps_path_0) == len(all_steps_path_1)
    for step_path_0, step_path_1 in zip(all_steps_path_0, all_steps_path_1):
        all_ranks_path_0 = _get_all_rank_path(step_path_0)
        all_ranks_path_1 = _get_all_rank_path(step_path_1)
        assert len(all_ranks_path_0) == len(all_ranks_path_1)
        return all_ranks_path_0, all_ranks_path_1
    raise ValueError("reach illegal code, concat the developer")
