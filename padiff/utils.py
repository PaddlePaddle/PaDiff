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

try:
    from paddle.fluid.layers.utils import flatten, pack_sequence_as, map_structure
except:
    from paddle.utils import flatten, pack_sequence_as, map_structure
from .file_loader import global_yaml_loader as yamls


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


def max_diff(paddle_output, torch_output):
    _max_diff = 0
    for (pt,), (tt,) in zip(for_each_tensor(paddle_output), for_each_tensor(torch_output)):
        if tt.numel() == 0 or pt.numel() == 0:
            continue
        temp = np.abs(tt.detach().cpu().numpy() - pt.detach().numpy()).max()
        if temp > _max_diff:
            _max_diff = temp

    return _max_diff


def assert_tensor_equal(tensor1, tensor2, options):
    """if equal: return None
    else: raise Error and Error Message.
    """
    atol = options["atol"]
    rtol = options["rtol"]
    compare_mode = options["compare_mode"]

    if compare_mode == "mean":
        np.testing.assert_allclose(tensor1.mean(), tensor2.mean(), atol=atol, rtol=rtol)
    elif compare_mode == "strict":
        np.testing.assert_allclose(tensor1, tensor2, atol=atol, rtol=rtol)


def tensors_mean(inp, mode):
    """
    TODO(wuzhanfei): This function is used to calcu loss in same way for paddle layer and torch module
    need to support real opt later
    """
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
        "diff_phase": "both",
        "compare_mode": "mean",
        "single_step": False,
        "debug": False,
        "use_loss": False,
        "use_opt": False,
        "steps": 1,
        "curent_model_idx": None,
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
        if key in ["atol", "rtol", "compare_mode", "single_step", "steps", "use_loss", "use_opt"]:
            print("  {}: `{}`".format(key, options[key]))
    print("}")

    yamls.options = options


def init_path_info(models):
    def _set_path_info(model, path):
        for name, child in model.named_children():
            path.append(name)
            setattr(child.model, "path_info", ".".join(path))
            _set_path_info(child, path)
            path.pop()

    for model in models:
        setattr(model.model, "path_info", model.name)
        _set_path_info(model, [model.name])


def remove_inplace(models):
    """
    Set `inplace` tag to `False` for torch module
    """

    for model in models:
        if model.model_type == "torch":
            for submodel in model.submodels():
                if hasattr(submodel, "inplace"):
                    submodel.inplace = False


"""
    log utils
"""

diff_log_path = os.path.join(sys.path[0], "diff_log")
__reset_log_dir__ = False


def reset_log_dir():
    if os.path.exists(diff_log_path):
        shutil.rmtree(diff_log_path)
    os.makedirs(diff_log_path)


def log_file(filename, mode, info):
    global __reset_log_dir__
    if not __reset_log_dir__:
        reset_log_dir()
        __reset_log_dir__ = True

    filepath = os.path.join(sys.path[0], "diff_log", filename)
    with open(filepath, mode) as f:
        f.write(info)

    return filepath


def log(*args):
    print("[AutoDiff]", *args)


def weight_struct_info(models, submodels):
    lines = 0
    infos = []

    for idx in range(2):
        model = models[idx]
        submodel = submodels[idx]
        title = f"{model.name}\n" + "=" * 40 + "\n"
        retval = weight_struct_string(model, mark=submodel, prefix=[" " * 4])
        info = title + "\n".join(retval)
        infos.append(info)
        lines += len(retval)

    retstr = ""
    if lines > 100:
        file_names = [f"weight_{models[idx].name}.log" for idx in range(2)]
        for idx in range(2):
            info = infos[idx]
            log_file(file_names[idx], "w", info)
        retstr += (
            f"Weight diff log saved to `{diff_log_path}/{file_names[0]}` and `{diff_log_path}/{file_names[1]}`.\n"
        )
        retstr += "Please view the reports and checkout the layers which is marked with `<---  *** HERE ***` !\n"
    else:
        for info in infos:
            retstr += info
            retstr += "\n"

    retstr += "\nNOTICE: submodel will be marked with `(skip)` because: \n"
    retstr += "    1. This submodel is contained by layer_map.\n"
    retstr += "    2. This submodel has no parameter, so padiff think it is a wrap layer.\n"

    retstr += "\nHint:\n"
    retstr += "    1. Check the definition order of params in submodel is the same.\n"
    retstr += "    2. Check the corresponding submodel have the same style:\n"
    retstr += "       param <=> param, buffer <=> buffer, embedding <=> embedding ...\n"
    retstr += "       cases like param <=> buffer, param <=> embedding are not allowed.\n"
    retstr += "    3. If can not change model codes, try to use a `LayerMap`\n"
    retstr += "       which can solve most problems.\n"
    retstr += "    0. Visit `https://github.com/PaddlePaddle/PaDiff` to find more infomation.\n"

    return retstr


def weight_struct_string(model, mark=None, prefix=[]):
    cur_str = ""
    for i, s in enumerate(prefix):
        if i == len(prefix) - 1:
            cur_str += s
        else:
            if s == " |--- ":
                cur_str += " |    "
            elif s == " +--- ":
                cur_str += "      "
            else:
                cur_str += s

    cur_str += str(model.class_name)

    if not hasattr(model.model, "no_skip"):
        cur_str += "  (skip)"

    if os.getenv("PADIFF_PATH_LOG") == "ON" and hasattr(model.model, "path_info"):
        cur_str += "  (" + model.path_info + ")"

    if mark.model is model.model:
        cur_str += "    <---  *** HERE ***"

    ret_strs = [cur_str]

    children = list(model.children())
    for i, child in enumerate(children):
        pre = " |--- "
        if i == len(children) - 1:
            pre = " +--- "
        prefix.append(pre)
        retval = weight_struct_string(child, mark, prefix)
        ret_strs.extend(retval)
        prefix.pop()

    return ret_strs


def debug_print(model, mark=None, prefix=[]):
    retval = weight_struct_string(model, mark=None, prefix=[])
    print("\n".join(retval))


"""
    stack tools
"""


import os.path as osp
import traceback


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
