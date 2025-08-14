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
import contextvars
import json
from typing import Dict, List, Optional
from ..utils import set_seed


# --- Core state management class ---
# This is an internal state shared by all Guards and should be placed first


class _SyncStepContext:
    def __init__(self):
        self.phase: str = ""
        self.base: Optional[Dict[int, List[Dict]]] = None
        self.old_phase: str = ""
        self.old_base: Optional[Dict[int, List[Dict]]] = None

    def _split_by_net_id(self, report):
        bucket = {}

        def _traversal(node, bucket):
            net_id = node["metas"]["net_id"]
            if net_id == -1:
                return
            if net_id not in bucket:
                bucket[net_id] = [node]
            else:
                bucket[net_id].append(node)

            for child in node["children"]:
                _traversal(child, bucket)

        for tree in report["tree"]:
            _traversal(tree, bucket)

        for key in bucket:
            bucket[key].sort(key=lambda x: x["metas"]["fwd_step"])

        return bucket


_context = _SyncStepContext()


_current_report = contextvars.ContextVar("current_report", default=None)


# --- Main Guard context manager ---
# Put the most important Guards that are most commonly used by users at the front


@contextlib.contextmanager
def report_guard(report):
    token = _current_report.set(report)
    try:
        yield
    finally:
        _current_report.reset(token)


@contextlib.contextmanager
def SyncStepGuard(diff_phase, report_path):
    _context.old_phase = _context.phase
    _context.old_base = _context.base

    try:
        with open(report_path + "/" + "report.json", "r") as report_file:
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


# --- Public utility functions for external calls ---
# These are "accessors" to the Guard's internal state and should be placed after the Guard it depends on


def current_report():
    return _current_report.get()


def single_step_state():
    return _context.phase


def find_base_report_node(net_id, step_idx):
    if _context.base is None:
        raise RuntimeError("SyncStepGuard context is not active.")
    return _context.base[net_id][step_idx]
