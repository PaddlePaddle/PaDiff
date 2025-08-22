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
import os

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
def SingleStepGuard(diff_phase, base_dump_path):
    """
    A context manager for single-step alignment mode.
    Attention: The code of this class currently looks the same as 'SyncStepGuard' above,
    but for future extensibility considerations, it is still not merged with 'SyncStepGuard'.
    """
    old_phase = _context.phase
    old_base = _context.base

    try:
        logger.debug(f"SingleStepGuard: Attempting to initialize for {diff_phase} phase.")
        logger.debug(f"SingleStepGuard: base_dump_path = {base_dump_path}")

        if not os.path.exists(base_dump_path):
            logger.error(f"base_dump_path '{base_dump_path}' does not exist.")

        report_json_path = os.path.join(base_dump_path, "report.json")
        if not os.path.exists(report_json_path):
            logger.error(f"report.json not found at '{report_json_path}'.")

        _context.phase = diff_phase
        report_json_path = os.path.join(base_dump_path, "report.json")
        with open(report_json_path, "r") as f:
            base_report_data = json.load(f)
        _context.base = _context._split_by_net_id(base_report_data)

        yield

    except _CallsComplete:
        raise
    except Exception as e:
        logger.error(f"SingleStepGuard failed to initialize: {e}")
        raise
    finally:
        _context.phase = old_phase
        _context.base = old_base
        logger.debug("SingleStepGuard: Context state restored.")


@contextlib.contextmanager
def AlignmentGuard(model, seed=42):
    """Prepare the model environment for accuracy alignment."""
    logger.debug(f"AlignmentGuard: Initializing for {model}")
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

    def __init__(self, message="CallsComplete: maximum number of forward calls reached."):
        self.message = message
        super().__init__(self.message)


@contextlib.contextmanager
def PaDiffGuard(
    model,
    name="model",
    auto_dump=True,
    align_depth="inf",
    single_step_mode=None,  # None, "forward", "backward"
    load_init_weights=False,
    load_first_inputs=False,
    base_dump_path=None,
    framework=None,
    seed=42,
    max_calls=1,
):
    # create_model
    if not hasattr(model, "report"):
        proxy_model = create_model(model, name=name)
    else:
        proxy_model = model

    logger.debug(f"PaDiffGuard: depth of alignment is {align_depth}.")
    proxy_model.marker.update_black_list_with_depth(align_depth)

    if load_init_weights or load_first_inputs or (single_step_mode is not None):
        assert (
            base_dump_path is not None
        ), "'base_dump_path' should not be None, when loading of init weights or/and first inputs is needed or using single_step mode"

    # load init weights
    if load_init_weights:
        from ...tools import load_init_weights_from_dump

        load_init_weights_from_dump(base_dump_path, proxy_model)

    # load first inputs
    if load_first_inputs:
        from ...tools import load_first_input_from_dump

        assert framework is not None, "'framework' must be setted if 'load_first_inputs' is True"
        loaded_inputs = load_first_input_from_dump(base_dump_path, framework)
        proxy_model.report._loaded_inputs = loaded_inputs

    # moniter number of calls
    calls_count = 0

    def calls_hook(m, input, output):
        nonlocal calls_count
        calls_count += 1
        logger.debug(f"PaDiffGuard: forward call #{calls_count}")
        if calls_count >= max_calls:
            logger.warning(f"PaDiffGuard: max_calls={max_calls} reached, raising _CallsComplete")
            raise _CallsComplete()

    try:
        # set hooks
        with contextlib.ExitStack() as stack:
            stack.enter_context(AlignmentGuard(proxy_model, seed=seed))
            stack.enter_context(report_guard(proxy_model.report))
            stack.enter_context(register_hooker(proxy_model))

            if single_step_mode is not None:
                stack.enter_context(SingleStepGuard(single_step_mode, base_dump_path))

            count_handle = proxy_model.register_forward_post_hook(calls_hook)
            stack.callback(count_handle.remove)

            yield model

            # dump report
            if auto_dump:
                dump_report(proxy_model, proxy_model.dump_path)

    except _CallsComplete:
        logger.info(f"PaDiffGuard: calls completed ({calls_count}/{max_calls})")
        # dump report
        if auto_dump:
            try:
                dump_report(proxy_model, proxy_model.dump_path)
            except Exception:
                pass

        import sys

        sys.exit(0)

    except SystemExit as e:
        logger.info("PaDiffGuard: SystemExit received, skipping dump_report.")
        raise
