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

import contextlib
import json

from ...utils import set_seed
from .base import _context, _current_report
from .hook import register_hooker


@contextlib.contextmanager
def report_guard(report):
    global _global_report

    token = _current_report.set(report)
    old_global_report = _global_report
    _global_report = report
    try:
        yield
    finally:
        _current_report.reset(token)
        _global_report = old_global_report


@contextlib.contextmanager
def SyncStepGuard(diff_phase, report_path):
    _context.old_phase = _context.phase
    _context.old_base = _context.base

    try:
        with open(report_path + "/" + "report.json") as report_file:
            report = json.load(report_file)
        _context.phase = diff_phase
        _context.base = _context._split_by_net_id(report)
        yield
    finally:
        _context.phase = _context.old_phase
        _context.base = _context.old_base


@contextlib.contextmanager
def AlignmentGuard(model, seed=42):
    """Prepare the model environment for accuracy alignment."""
    model.model.train()
    model.toggle_dropout(enable=False)
    set_seed(seed)
    try:
        yield model
    finally:
        pass


@contextlib.contextmanager
def PaDiffGuard(model, seed=42):
    with contextlib.ExitStack() as stack:
        stack.enter_context(AlignmentGuard(model, seed=seed))
        stack.enter_context(report_guard(model.report))
        stack.enter_context(register_hooker(model))
        yield model
