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

import contextvars

# --- Core state management class ---
# This is an internal state shared by all Guards and should be placed first


class _SyncStepContext:
    def __init__(self):
        self.phase: str = ""
        self.base: dict[int, list[dict]] | None = None
        self.old_phase: str = ""
        self.old_base: dict[int, list[dict]] | None = None

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
_global_report = None


# --- Public utility functions for external calls ---
# These are "accessors" to the Guard's internal state and should be placed after the Guard it depends on


def current_report():
    try:
        ctx_report = _current_report.get()
        if ctx_report is not None:
            return ctx_report
    except LookupError:
        pass
    return _global_report


def single_step_state():
    return _context.phase


def find_base_report_node(net_id, step_idx):
    if _context.base is None:
        raise RuntimeError("Neither SyncStepGuard or SingleStepGuard context is not active.")

    if net_id not in _context.base:
        raise RuntimeError(f"Cannot find net_id={net_id} in base report.")

    node_list = _context.base[net_id]
    if step_idx < 0 or step_idx >= len(node_list):
        raise RuntimeError(f"Index out of range: net_id={net_id}, step_idx={step_idx}, list length={len(node_list)}")

    return _context.base[net_id][step_idx]
