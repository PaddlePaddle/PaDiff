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

import json
import collections.abc

import numpy as np
import paddle
import torch

import os.path as osp
import traceback


def set_seed(seed=42):
    np.random.seed(seed)
    paddle.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_numpy_from_tensor(tensor):
    if tensor.dtype == torch.bfloat16:
        np_array = tensor.cpu().detach().float().numpy()
    elif tensor.dtype == paddle.bfloat16:
        np_array = tensor.cpu().detach().astype("float32").numpy()
    else:
        np_array = tensor.cpu().detach().numpy()
    return np_array


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


def _clone_tensor(inp):  # to cpu
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
    return map_structure(_clone_tensor, inputs)


def clone_tensors(inputs):
    tensors = [_clone_tensor(t) for (t,) in for_each_tensor(inputs)]
    return tensors


def is_sequence(x):
    """
    Check if x is a sequence (list, tuple, etc.) but not string or bytes.
    """
    if isinstance(x, str) or isinstance(x, bytes):
        return False
    return isinstance(x, collections.abc.Sequence) or isinstance(x, collections.abc.MutableSequence)


def to_sequence(x):
    return list(x) if is_sequence(x) else [x]


def traverse(structure, on_leaf, on_container=None):
    """
    Traverse a nested structure and apply `on_leaf` to each leaf.
    Optionally apply `on_container` to containers (dict, list, etc.).

    Args:
        structure: The nested structure to traverse.
        on_leaf: Function to apply to each leaf (e.g., Tensor).
        on_container: Optional function to apply to container after children are processed.

    Returns:
        A new structure with leaves transformed.
    """
    if structure is None:
        return structure

    # basic type
    if isinstance(structure, (str, int, float, bool)):
        return on_leaf(structure)

    # Tensor
    if isinstance(structure, (paddle.Tensor, torch.Tensor)):
        return on_leaf(structure)

    # dict
    if isinstance(structure, dict):
        new_dict = type(structure)()
        for k in sorted(structure.keys()):
            new_dict[k] = traverse(structure[k], on_leaf, on_container)
        return on_container(new_dict) if on_container else new_dict

    # list
    if isinstance(structure, list):
        result = [traverse(item, on_leaf, on_container) for item in structure]
        return on_container(result) if on_container else result

    # tuple
    if isinstance(structure, tuple):
        # namedtuple
        if hasattr(structure, "_fields"):
            result = type(structure)(
                *[traverse(getattr(structure, field), on_leaf, on_container) for field in structure._fields]
            )
            return on_container(result) if on_container else result
        else:  # tuple
            result = tuple(traverse(item, on_leaf, on_container) for item in structure)
            return on_container(result) if on_container else result

    # others like, ModelOutput, CausalLMOutputWithPast, DynamicCache
    if hasattr(structure, "__dict__"):
        try:
            new_obj = type(structure)()
            for k in sorted(structure.__dict__.keys()):
                if not k.startswith("_"):
                    v = getattr(structure, k)
                    setattr(new_obj, k, traverse(v, on_leaf, on_container))
            return on_container(new_obj) if on_container else new_obj
        except:
            pass

    # others
    if hasattr(structure, "__class__"):
        try:
            new_obj = type(structure)()
            for k in dir(structure):
                if not k.startswith("_") and not callable(getattr(structure, k)):
                    try:
                        v = getattr(structure, k)
                        setattr(new_obj, k, traverse(v, on_leaf, on_container))
                    except:
                        pass
            return on_container(new_obj) if on_container else new_obj
        except:
            pass

    # default: treat as leaves
    return on_leaf(structure)


def flatten(structure):
    flat_list = []

    def on_leaf(x):
        flat_list.append(x)
        return None

    traverse(structure, on_leaf)
    return flat_list


def map_structure(func, structure):
    return traverse(structure, on_leaf=func)


def pack_sequence_as(structure, flat_sequence):
    flat_iter = iter(flat_sequence)
    return traverse(structure, on_leaf=lambda x: next(flat_iter), on_container=lambda x: x)


"""
    traversal tools
"""


def for_each_tensor(*structure):
    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)
    entries = filter(lambda x: is_tensors(*x), entries)
    yield from entries


def for_each_grad_tensor(*structure):
    def filter_fn(ts):
        return is_tensors(*ts) and is_require_grad(ts[0])

    for ts in filter(filter_fn, for_each_tensor(*structure)):
        yield ts


def for_each_grad_tensor_no_require(*structure):
    for ts in for_each_tensor(*structure):
        if is_tensors(*ts):
            yield ts


def map_structure_and_replace_key(func, structure1, structure2):
    """
    Apply `func` to each entry in `structure` and return a new structure.
    deprecated: func(*x) will cause more than 1 tensor to be passed in, but 'inner()' of 'replace_forward_output' only wants one input
    """
    flat_structure = [flatten(s) for s in structure1]
    entries = zip(*flat_structure)
    return pack_sequence_as(structure2, [func(*x) for x in entries])


"""
    tensor compare or compute
"""


def assert_tensor_equal(tensor1, tensor2, cfg):
    """
    return None or raise Error.
    """
    atol = cfg.get("atol", 0)
    rtol = cfg.get("rtol", 1e-7)
    compare_mode = cfg.get("compare_mode", "mean")

    if compare_mode == "mean":
        np.testing.assert_allclose(tensor1.mean(), tensor2.mean(), atol=atol, rtol=rtol)
    elif compare_mode == "strict":
        np.testing.assert_allclose(tensor1, tensor2, atol=atol, rtol=rtol)
    elif compare_mode == "abs_mean":
        np.testing.assert_allclose(abs(tensor1).mean(), abs(tensor2).mean(), atol=atol, rtol=rtol)
    else:
        raise RuntimeError(f"Invalid compare_mode {compare_mode}")


"""
    tools for recording frame stack
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


def frames_to_string(frames, indent=0):
    indent = " " * indent
    lines = []
    for f in frames:
        lines.append(f"{indent}File {f.filename}: {f.lineno}    {f.name}\n{indent}{indent}{f.line}")
    return "\n".join(lines)


"""
    load tools
"""


def load_numpy(path):
    if path is None:
        return None
    return np.load(path)


def load_json(path, report_name):
    with open(path + "/" + report_name) as f:
        retval = json.load(f)
    return retval
