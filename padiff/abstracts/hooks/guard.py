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
from ..proxy import create_model
from ...tools import dump_report
from ...utils import logger


_global_report = None


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


class _CallsComplete(Exception):
    """A private exception used by PaDiffGuard to interrupt execution.

    This exception is raised by the internal calls_hook when the maximum number
    of calls (max_calls) has been reached. It is caught by PaDiffGuard
    to exit the context manager gracefully.
    """

    pass


@contextlib.contextmanager
def PaDiffGuard(
    model,
    name="model",
    auto_dump=True,
    load_weights_from=None,
    load_inputs_from=None,
    framework=None,
    verbose=False,
    seed=42,
    max_calls=1,
):
    # create_model
    if not hasattr(model, "report"):
        proxy_model = create_model(model, name=name)
    else:
        proxy_model = model

    # load init weights
    if load_weights_from:
        from ...tools import load_init_weights_from_dump

        load_init_weights_from_dump(load_weights_from, proxy_model, verbose=verbose)

    # load inputs
    loaded_inputs = None
    if load_inputs_from:
        from ...tools import load_first_input_from_dump

        assert framework is not None, "'framework' must be setted if 'load_inputs_from' is not None"
        loaded_inputs = load_first_input_from_dump(load_inputs_from, framework)
    report = proxy_model.report
    report._loaded_inputs = loaded_inputs

    # moniter number of calls
    calls_count = 0

    def calls_hook(m, input, output):
        nonlocal calls_count
        calls_count += 1
        if calls_count >= max_calls:
            raise _CallsComplete()

    try:
        # set hooks
        with contextlib.ExitStack() as stack:
            stack.enter_context(AlignmentGuard(proxy_model, seed=seed))
            stack.enter_context(report_guard(proxy_model.report))
            stack.enter_context(register_hooker(proxy_model))

            count_handle = proxy_model.register_forward_post_hook(calls_hook)
            stack.callback(count_handle.remove)

            # dump report
            if auto_dump:
                stack.callback(lambda: dump_report(proxy_model, proxy_model.dump_path))

            yield model

    except _CallsComplete:
        logger.info(f"PaDiffGuard: calls completed ({calls_count}/{max_calls})")
        # raise
        import sys

        sys.exit(0)
