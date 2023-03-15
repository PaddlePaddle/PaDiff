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

from .actions import get_action
from .hooks import register_paddle_hooker, register_torch_hooker, paddle_api_hook, torch_api_hook
from .module_struct import copy_module_struct, print_struct_info, reorder_and_match_reports, debug_print_struct
from .report import Report, report_guard, current_paddle_report, current_torch_report

__all__ = [
    "get_action",
    "register_paddle_hooker",
    "register_torch_hooker",
    "paddle_api_hook",
    "torch_api_hook",
    "copy_module_struct",
    "print_struct_info",
    "reorder_and_match_reports",
    "debug_print_struct",
    "Report",
    "report_guard",
    "current_paddle_report",
    "current_torch_report",
]
