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
import sys
import functools
import inspect
import paddle
import torch

from ...utils import set_seed, logger, wrap_optimizer_step
from .base import _context, _current_report, _CallsComplete, current_report, get_calls_context
from .hook import register_hooker
from ..proxy import create_model
from ...tools import load_first_input_from_dump, load_init_weights_from_dump


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


@contextlib.contextmanager
def InputCaptureGuard(model, base_dump_path=None, framework=None, load_first_inputs=False):
    """
    Context manager to capture or inject first input. Cannot be implemented as a hook
    because hook registered through 'register_forward_pre_hook' can only capture args but not kwargs.
    """
    if hasattr(model, "_padiff_input_captured"):
        yield
        return

    original_forward = model.forward

    @functools.wraps(original_forward)
    def tracked_forward(*args, **kwargs):
        report = current_report()
        if not args and not kwargs:
            logger.warning("Skipped capturing or loading input: both args and kwargs are empty.")
            return original_forward(*args, **kwargs)

        if load_first_inputs and base_dump_path and framework and not hasattr(report, "_inputs_loaded"):
            assert framework is not None, "'framework' must be setted if 'load_first_inputs' is True"
            logger.info("Loading first input from dump")
            loaded_inputs = load_first_input_from_dump(base_dump_path, framework)
            if loaded_inputs is not None:
                loaded_args, loaded_kwargs = loaded_inputs
                try:
                    sig = inspect.signature(original_forward)
                    valid_arg_names = set(sig.parameters.keys())
                except Exception as e:
                    logger.warning(f"Failed to get forward signature: {e}. Using all keys.")
                    valid_arg_names = set(loaded_kwargs.keys())

                filtered_kwargs = {k: v for k, v in loaded_kwargs.items() if k in valid_arg_names}
                dropped_keys = set(loaded_kwargs.keys()) - set(filtered_kwargs.keys())
                if dropped_keys:
                    logger.debug(f"Dropped keys not in forward signature: {dropped_keys}")

                final_kwargs = {**kwargs, **filtered_kwargs}
                report._inputs_loaded = True
                return original_forward(*loaded_args, **final_kwargs)

        if report and not hasattr(report, "first_input_captured"):

            def serialize(x):
                if isinstance(x, (paddle.Tensor, torch.Tensor)):
                    return ("Tensor", x.detach().cpu().numpy())
                elif isinstance(x, dict):
                    return ("dict", {k: serialize(v) for k, v in x.items()})
                elif isinstance(x, (list, tuple)):
                    return (type(x).__name__, [serialize(item) for item in x])
                else:
                    return (type(x).__name__, str(x))

            serialized = {
                "args": [serialize(x) for x in args] if args else [],
                "kwargs": {k: serialize(v) for k, v in kwargs.items()},
            }
            if serialized["args"] or serialized["kwargs"]:
                report.first_input = serialized
                report.first_input_captured = True
                logger.info(f"Captured full input: args={len(args)}, kwargs={list(kwargs.keys())}")
            else:
                logger.warning("Skipped capturing input: serialized input is empty.")

        return original_forward(*args, **kwargs)

    model.forward = tracked_forward
    model._padiff_input_captured = True

    try:
        yield
    finally:
        model.forward = original_forward


@contextlib.contextmanager
def MaxCallsGuard(max_calls: int, model):
    if max_calls <= 0:
        yield
        return

    calls_context = get_calls_context()

    def pre_hook(m, input):
        if calls_context.is_exceeded():
            logger.warning(f"PaDiffGuard: max_calls={max_calls} reached, raising _CallsComplete")
            raise _CallsComplete()
        count = calls_context.increment()
        logger.info(f"MaxCallsGuard: forward start calling #{count}")

    handle = model.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def PaDiffGuard(
    model,
    optimizer=None,
    name="model",
    align_depth="inf",
    single_step_mode=None,  # None, "forward", "backward"
    load_init_weights=False,
    load_first_inputs=False,
    base_dump_path=None,
    framework=None,
    seed=42,
    max_calls=1,
    black_list=None,
    keys_mapping=None,
):
    # moniter number of calls
    calls_context = get_calls_context()
    reset_flag = calls_context.state["count"] == 0

    if reset_flag:
        # set max calls
        calls_context.set_limit(max_calls)

        logger.info(f"PaDiffGuard: creating proxy model.")
        proxy_model = create_model(model, name=name, reset_dir=reset_flag)
        model._padiff_proxy = proxy_model

        if optimizer is not None and not hasattr(optimizer, "_padiff_proxy_model"):
            logger.info(f"PaDiffGuard: wrapping optimizer.step().")
            optimizer._padiff_proxy_model = proxy_model
            wrap_optimizer_step(optimizer)

        if load_init_weights or load_first_inputs or (single_step_mode is not None):
            assert (
                base_dump_path is not None
            ), "'base_dump_path' should not be None, when loading of init weights or/and first inputs is needed or using single_step mode"

        # load init weights
        if load_init_weights:
            load_init_weights_from_dump(base_dump_path, proxy_model, keys_mapping)

        logger.debug(f"PaDiffGuard: depth of alignment is {align_depth}.")
        proxy_model.marker.update_black_list_with_depth(align_depth)
        proxy_model.update_black_list_with_name(black_list)

    else:
        proxy_model = model._padiff_proxy

    try:
        # set hooks
        with contextlib.ExitStack() as stack:
            # moniter number of calls
            if max_calls > 0:
                stack.enter_context(MaxCallsGuard(max_calls, proxy_model))

            stack.enter_context(AlignmentGuard(proxy_model, seed=seed))
            stack.enter_context(report_guard(proxy_model.report))
            # load first inputs
            if reset_flag:
                stack.enter_context(InputCaptureGuard(proxy_model.model, base_dump_path, framework, load_first_inputs))

            if single_step_mode is not None:
                stack.enter_context(SingleStepGuard(single_step_mode, base_dump_path))

            stack.enter_context(register_hooker(proxy_model))

            yield model

    except _CallsComplete:
        # dump
        proxy_model.dump_report(proxy_model.dump_path)
        proxy_model.dump_weights(proxy_model.dump_path)
        if optimizer is None:
            proxy_model.dump_grads(proxy_model.dump_path)

        sys.exit(0)

    except SystemExit as e:
        logger.info("PaDiffGuard: SystemExit received, skipping dump_report.")
        raise
