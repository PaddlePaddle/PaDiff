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
import warnings
from collections import namedtuple
from itertools import zip_longest

import numpy as np
import paddle
import torch
from paddle.fluid.layers.utils import flatten, pack_sequence_as, map_structure


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


def map_for_each_weight(fn, layer, module):
    """
    Automatically fill weights by randn.
    """
    for paddle_sublayer, torch_submodule in zip_longest(layer.sublayers(True), module.modules(), fillvalue=None):
        if paddle_sublayer is None or torch_submodule is None:
            raise RuntimeError("Torch and Paddle return difference number of sublayers. Check your model.")
        for (name, paddle_param), torch_param in zip(
            paddle_sublayer.named_parameters("", False),
            torch_submodule.parameters(False),
        ):
            fn(paddle_sublayer, torch_submodule, name, paddle_param, torch_param)


def map_for_each_sublayer(fn, layer, module):
    """
    Automatically fill weights by randn.
    """
    for paddle_sublayer, torch_submodule in zip(layer.sublayers(True), module.modules()):
        fn(paddle_sublayer, torch_submodule)


def map_structure_and_replace_key(func, structure1, structure2):
    """
    Apply `func` to each entry in `structure` and return a new structure.
    """
    flat_structure = [flatten(s) for s in structure1]
    entries = zip(*flat_structure)
    return pack_sequence_as(structure2, [func(*x) for x in entries])


def _clone_tensor(inp):
    """
    clone into cpu to save GPU memory.
    """
    if isinstance(inp, (torch.Tensor, paddle.Tensor)):
        new_t = inp.detach().cpu().clone()
        if is_require_grad(inp):
            set_require_grad(new_t)
        return new_t
    else:
        return inp


def clone_tensors(inputs):
    """
    Clone the tensors in inputs. for example:
    Inputs = [2, Tensor1, "sdf", Tensor2]
    Return = [Tensor1_cloned, Tensor2_cloned]
    """
    cloned_inputs = []
    for (t,) in for_each_tensor(inputs):
        cloned_inputs.append(_clone_tensor(t))
    return cloned_inputs


def clone_structure(inputs):
    """
    Clone a nested structure.
    """
    return map_structure(_clone_tensor, inputs)


def is_sublayer(father_net, child_net):
    """
    return True if child_net is the DIRECTL children of father_net.
    """

    def _is_sublayer(father_net, child_net, layer_type):

        for child in father_net.children():
            check_list = child if isinstance(child, layer_type["layer_list"]) else [child]
            for l in check_list:
                if isinstance(l, layer_type["sequential"]):
                    if _is_sublayer(l, child_net, layer_type):
                        return True
                else:
                    if id(l) == id(child_net):
                        return True
        return False

    if isinstance(father_net, torch.nn.Module) and isinstance(child_net, torch.nn.Module):
        layer_type = {
            "sequential": torch.nn.Sequential,
            "layer_list": torch.nn.ModuleList,
        }
        if _is_sublayer(father_net, child_net, layer_type):
            return True
        return False
    elif isinstance(father_net, paddle.nn.Layer) and isinstance(child_net, paddle.nn.Layer):
        layer_type = {
            "sequential": paddle.nn.Sequential,
            "layer_list": paddle.nn.LayerList,
        }
        if _is_sublayer(father_net, child_net, layer_type):
            return True
        return False
    else:
        raise RuntimeError("father net is not Module / Layer")


def traversal_layers(net, layer_mapping):
    for child in net.children():
        if not (isinstance(child, torch.nn.Sequential) or isinstance(child, paddle.nn.Sequential)):
            yield child
        ignore_sublayer = False
        for key, val in layer_mapping.items():
            if id(child) == id(key) or id(child) == id(val):
                ignore_sublayer = True
        if not ignore_sublayer:
            for sublayer in traversal_layers(child, layer_mapping):
                yield sublayer


class TableView:
    """
    A search speedup wrapper class.
    """

    def __init__(self, data, key=None):
        self.data = data
        self.view = {}
        assert callable(key), "Key must be callable with a paramter: x -> key."
        for item in self.data:
            if key(item) not in self.view:
                self.view[key(item)] = [item]
            else:
                warnings.warn("Warning: duplicate key is found, use list + pop strategy.")
                self.view[key(item)].append(item)

    def __getitem__(self, key):
        assert key in self.view, "{} is not found in index.".format(key)
        ret = self.view[key].pop(0)  # pop for sorting.
        return ret

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        return key in self.view


class TreeView:
    """
    This class is used to organize ReportItems in order of backward process

    wrap items as a tree structure:
    [1, 2, 3, 4, 5, 6]
    the last item is the root of the layers.
    if the child is 2 and 5, then we can construct a tree:
      6
      |---------|
      2         5
      |         |
    [1,2]    [3,4,5]   <--- recursive construct.
    """

    def __init__(self, data):
        """data is the forward items. the last one is the root layer."""
        Node = namedtuple("Node", ["value", "children"])

        def _construct_tree(begin_idx, end_idx):
            if end_idx < begin_idx:
                raise RuntimeError("[{}, {}] is invalid.".format(begin_idx, end_idx))
            root = Node(value=data[end_idx], children=[])
            last = begin_idx
            for i in range(begin_idx, end_idx):
                if is_sublayer(root.value.net, data[i].net):
                    root.children.append(_construct_tree(last, i))
                    last = i + 1
            return root

        self.root = _construct_tree(0, len(data) - 1)
        self.data = data

    def __len__(self):
        return len(self.data)

    def traversal_forward(self):
        """
        with the order of:
        child1, child2, child3 ... childn, root
        """

        def _traversal_forward(root):
            for child in root.children:
                for item in _traversal_forward(child):
                    yield item
            yield root.value

        for item in _traversal_forward(self.root):
            yield item

    def traversal_backward(self):
        """
        with the order of:
        childn, childn-1, child... child1, root
        """

        def _traversal_backward(root):
            for child in root.children[::-1]:
                for item in _traversal_backward(child):
                    yield item
            yield root.value

        for item in _traversal_backward(self.root):
            yield item


diff_log_path = os.path.join(sys.path[0], "diff_log")


def reset_log_dir():
    if os.path.exists(diff_log_path):
        shutil.rmtree(diff_log_path)
    os.makedirs(diff_log_path)


def clean_log_dir():
    if not os.listdir(diff_log_path):
        os.rmdir(diff_log_path)


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
            means.append(t[0].mean())
        loss = torch.stack(means).mean()
        return loss
    elif mode == "paddle":
        means = []
        for t in for_each_tensor(inp):
            means.append(t[0].mean())
        loss = paddle.stack(means).mean()
        return loss
    else:
        raise RuntimeError("unrecognized mode `{}`, expected: `torch` or `paddle`".format(mode))


def max_diff(paddle_output, torch_output):
    p_values = []
    t_values = []
    for t in for_each_tensor(paddle_output):
        p_values.append(t[0])
    for t in for_each_tensor(torch_output):
        t_values.append(t[0])

    _max_diff = 0
    for (pt, tt) in zip(p_values, t_values):
        temp = np.abs(tt.detach().numpy() - pt.detach().numpy()).max()
        if temp > _max_diff:
            _max_diff = temp

    return _max_diff


def log(*args):
    print("[AutoDiff]", *args)


def compare_tensor(tensor1, tensor2, atol=0, rtol=1e-7, compare_mode="mean"):
    """
    compare tensor and return bool
    """
    if tensor1 is None and tensor2 is None:
        return True
    if compare_mode == "strict":
        return np.allclose(tensor1, tensor2, atol=atol, rtol=rtol)
    elif compare_mode == "mean":
        return np.allclose(tensor1.mean(), tensor2.mean(), atol=atol, rtol=rtol)
    else:
        raise RuntimeError("compare_mode `{}` is not supported, use `strict` or `mean` instead".format(compare_mode))


def assert_tensor_equal(tensor1, tensor2, atol=0, rtol=1e-7, compare_mode="mean"):
    """if equal: return None
    else: raise Error and Error Message.
    """
    if tensor1 is None and tensor2 is None:
        return True
    if compare_mode == "mean":
        np.testing.assert_allclose(tensor1.mean(), tensor2.mean(), atol=atol, rtol=rtol)
    elif compare_mode == "strict":
        np.testing.assert_allclose(tensor1, tensor2, atol=atol, rtol=rtol)


def init_options(options):
    default_options = {
        "atol": 0,
        "rtol": 1e-7,
        "diff_phase": "both",
        "compare_mode": "mean",
        "single_step": False,
        "debug": False,
        "cmd": False,
    }

    for key in default_options.keys():
        if key not in options.keys():
            options[key] = default_options[key]

    if options["single_step"]:
        options["diff_phase"] = "forward"
        log("In single_step mode, diff_phase is set to `forward`.")

    log("Your options:")
    print("{")
    for key in options.keys():
        print("  {}: `{}`".format(key, options[key]))
    print("}")


def modify_layer_mapping(layer_mapping):
    remove_keys = []
    for key in layer_mapping.keys():
        if not isinstance(key, paddle.nn.Layer):
            remove_keys.append(key)
    for key in remove_keys:
        layer_mapping[layer_mapping[key]] = key
        layer_mapping.pop(key)
