# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import shutil

import numpy as np
import paddle
import torch
from itertools import zip_longest
import os.path as osp
import traceback


try:
    from paddle.fluid.layers.utils import flatten, pack_sequence_as, map_structure
except:
    from paddle.utils import flatten, pack_sequence_as, map_structure


"""
    global infos
"""
global_options = None
log_path = os.path.join(sys.path[0], "padiff_log")

__reset_log_dir__ = False       # reset log_path only once


"""
    clone tensor
"""


def is_tensor(x):
    return isinstance(x, (paddle.Tensor, torch.Tensor))


def is_tensors(*x):
    ret = True
    for i in x:
        ret = ret and is_tensor(i)
    return ret


def is_require_grad(x):
    if hasattr(x, "requires_grad"):
        return x.requires_grad
    if hasattr(x, "stop_gradient"):
        return not x.stop_gradient
    return False


def set_require_grad(x):
    if hasattr(x, "requires_grad"):
        x.requires_grad = True
    if hasattr(x, "stop_gradient"):
        x.stop_gradient = False


def _clone_tensor(inp):
    """
    clone into cpu to save GPU memory.
    """
    if isinstance(inp, (torch.Tensor, paddle.Tensor)):
        if inp.numel() == 0:
            if isinstance(inp, torch.Tensor):
                return torch.tensor([], dtype=inp.dtype)
            else:
                return paddle.to_tensor([], dtype=inp.dtype)
        new_t = inp.detach().cpu().clone()
        if is_require_grad(inp):
            set_require_grad(new_t)
        return new_t
    else:
        return inp


def clone_structure(inputs):
    """
    Clone a nested structure.
    """
    return map_structure(_clone_tensor, inputs)


def clone_tensors(inputs):
    tensors = [_clone_tensor(t) for (t,) in for_each_tensor(inputs)]
    return tensors

"""
    traversal tools
"""


def for_each_tensor(*structure):
    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)
    entries = filter(lambda x: is_tensors(*x), entries)
    for tensors in entries:
        yield tensors


def for_each_grad_tensor(*structure):
    def filter_fn(ts):
        return is_tensors(*ts) and is_require_grad(ts[0])

    for ts in filter(filter_fn, for_each_tensor(*structure)):
        yield ts


def map_structure_and_replace_key(func, structure1, structure2):
    """
    Apply `func` to each entry in `structure` and return a new structure.
    """
    flat_structure = [flatten(s) for s in structure1]
    entries = zip(*flat_structure)
    return pack_sequence_as(structure2, [func(*x) for x in entries])


"""
    tensor compare or compute
"""


def max_diff(output1, output2):
    _max_diff = 0
    for (tensor1,), (tensor2,) in zip(for_each_tensor(output1), for_each_tensor(output2)):
        if tensor2.numel() == 0 or tensor1.numel() == 0:
            continue
        temp = np.abs(tensor2.detach().cpu().numpy() - tensor1.detach().cpu().numpy()).max()
        if temp > _max_diff:
            _max_diff = temp

    return _max_diff


def assert_tensor_equal(tensor1, tensor2, cfg):
    """
    return None or raise Error.
    """
    atol = cfg["atol"]
    rtol = cfg["rtol"]
    compare_mode = cfg["compare_mode"]

    if compare_mode == "mean":
        np.testing.assert_allclose(tensor1.mean(), tensor2.mean(), atol=atol, rtol=rtol)
    elif compare_mode == "strict":
        np.testing.assert_allclose(tensor1, tensor2, atol=atol, rtol=rtol)


def tensors_mean(inp, mode):
    if isinstance(inp, torch.Tensor) or isinstance(inp, paddle.Tensor):
        return inp.mean()

    if mode == "torch":
        means = []
        for t in for_each_tensor(inp):
            means.append(t[0].to(torch.float32).mean())
        loss = torch.stack(means).mean()
        return loss
    elif mode == "paddle":
        means = []
        for t in for_each_tensor(inp):
            means.append(t[0].astype("float32").mean())
        loss = paddle.stack(means).mean()
        return loss
    else:
        raise RuntimeError("unrecognized mode `{}`, expected: `torch` or `paddle`".format(mode))


"""
    init tools
"""

def init_options(options):
    default_options = {
        "atol": 0,
        "rtol": 1e-7,
        "compare_mode": "mean",

        "auto_init": True,
        "diff_phase": "both",
        "single_step": False,
        "steps": 1,
        "use_loss": False,
        "use_opt": False,
    }

    default_options.update(options)
    options.update(default_options)

    if not options["single_step"] and options["diff_phase"] == "backward":
        options["diff_phase"] = "both"
        log("  Not in single_step mode, diff_phase `backward` is not supported, set to `both` instead.")

    if options["diff_phase"] == "forward":
        if options["use_opt"]:
            options["use_opt"] = False
            log("  Diff_phase is `forward`, optimizer will not be used.")
        if options["steps"] > 1:
            options["steps"] = 1
            log("  Diff_phase is `forward`, steps is set to `1`.")

    if options["steps"] > 1 and options["use_opt"] == False:
        options["steps"] = 1
        log("  Steps is set to `1`, because optimizers are not given.")

    log("Your options:")
    print("{")
    for key in options.keys():
        if key in ["atol", "rtol", "auto_init", "compare_mode", "single_step", "steps", "use_loss", "use_opt"]:
            print("  {}: `{}`".format(key, options[key]))
    print("}")

    global global_options
    global_options = options


"""
    process files
"""

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


"""
    log utils
"""

def log_file(filename, mode, info):
    global __reset_log_dir__
    if not __reset_log_dir__:
        reset_dir(log_path)
        __reset_log_dir__ = True

    filepath = os.path.join(sys.path[0], "diff_log", filename)
    with open(filepath, mode) as f:
        f.write(info)

    return filepath


def log(*args):
    print("[AutoDiff]", *args)


"""
    stack tools
"""

def _is_system_package(filename):
    exclude = [
        "lib/python",
        "/usr/local",
        osp.dirname(paddle.__file__),
        osp.dirname(torch.__file__),
        osp.dirname(__file__),  # exclude padiff
    ]
    for pattern in exclude:
        if pattern in filename:
            return True
    return False


def extract_frame_summary():
    """
    extract the current call stack by traceback module.
    gather the call information and put them into ReportItem to helper locate the error.

    frame_summary:
        line: line of the code
        lineno: line number of the file
        filename: file name of the stack
        name: the function name.
    """
    frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
    last_user_fs = None
    for fs in frame_summarys:
        if not _is_system_package(fs.filename):
            last_user_fs = fs
            break
    assert last_user_fs is not None, "Error happend, can't return None."
    return last_user_fs, frame_summarys


"""
    check dataloader
"""


def check_dataloader(first_loader, second_loader, **kwargs):
    def get_numpy(data):
        if isinstance(data, (paddle.Tensor, torch.Tensor)):
            return data.detach().cpu().numpy()
        return data

    options = {
        "atol": 0,
        "rtol": 1e-7,
        "compare_mode": "mean",
    }
    options.update(kwargs)

    for data_0, data_1 in zip_longest(first_loader, second_loader, fillvalue=None):
        if data_0 is None or data_1 is None:
            raise RuntimeError("Given dataloader return difference number of datas.")
        try:
            assert_tensor_equal(get_numpy(data_0), get_numpy(data_1), options)
        except Exception as e:
            log("check dataloader failed!!!")
            print(f"{type(e).__name__ + ':  ' + str(e)}")
            return False
    return True


class Counter:
    def __init__(self):
        self.clear()

    def clear(self):
        self.id = 0

    def get_id(self):
        ret = self.id
        self.id += 1
        return ret