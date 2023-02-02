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
from collections import namedtuple, Iterable
from itertools import zip_longest

import numpy as np
import paddle
import torch
from paddle.fluid.layers.utils import flatten, pack_sequence_as, map_structure


# process Tensor


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


# traversal tools


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


# Report tools


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


# logs

diff_log_path = os.path.join(sys.path[0], "diff_log")


def reset_log_dir():
    if os.path.exists(diff_log_path):
        shutil.rmtree(diff_log_path)
    os.makedirs(diff_log_path)


def clean_log_dir():
    if not os.listdir(diff_log_path):
        os.rmdir(diff_log_path)


def log(*args):
    print("[AutoDiff]", *args)


# tensor compute


def max_diff(paddle_output, torch_output):
    _max_diff = 0
    for (pt,), (tt,) in zip(for_each_tensor(paddle_output), for_each_tensor(torch_output)):
        temp = np.abs(tt.detach().numpy() - pt.detach().numpy()).max()
        if temp > _max_diff:
            _max_diff = temp

    return _max_diff


def compare_tensor_ret_bool(tensor1, tensor2, atol=0, rtol=1e-7, compare_mode="mean"):
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


def assert_tensor_equal(tensor1, tensor2, options):
    """if equal: return None
    else: raise Error and Error Message.
    """
    atol = options["atol"]
    rtol = options["rtol"]
    compare_mode = options["compare_mode"]
    if tensor1 is None and tensor2 is None:
        return True
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


# init tools


def init_options(options):
    default_options = {
        "atol": 0,
        "rtol": 1e-7,
        "diff_phase": "both",
        "compare_mode": "mean",
        "single_step": False,
        "debug": False,
        "cmd": False,
        "loss_fn": False,
        "opt": False,
        "multi_steps": False,
    }

    default_options.update(options)
    options.update(default_options)

    log("Your options:")

    if options["single_step"]:
        options["diff_phase"] = "forward"
        log("  In single_step mode, diff_phase will be set to `forward`.")

    if options["diff_phase"] == "forward" and options["opt"]:
        log("  Diff_phase is `forward`, optimizer will not be used.")

    print("{")
    for key in options.keys():
        if key not in ["debug", "cmd", "loss_fn", "opt", "multi_steps"]:
            print("  {}: `{}`".format(key, options[key]))
    print("}")


def init_LayerMap(layer, module, layer_map):
    if layer_map is None:
        layer_map = LayerMap()
    elif isinstance(layer_map, dict):
        new_map = LayerMap()
        new_map.map = layer_map
        layer_map = new_map
    else:
        assert isinstance(layer_map, LayerMap), "Invalid Argument."

    layer_map.ignore_class(layer)
    layer_map.ignore_class(module)

    return layer_map


class LayerMap(object):
    def __init__(self):
        self._layer_one2one = {}  # key: paddle.nn.Layer, value: torch.nn.Module
        self._layer_ignore = set()  # ignore layer in this set
        self._layer_ignore_sublayer = set()  # ignore sublayer of layers in this set (do not include themselves)

        self._ignore_cls = (  # these classes will be ignored
            paddle.nn.Sequential,
            torch.nn.Sequential,
        )

    @staticmethod
    def modify_layer_map(layer_map):
        swap_keys = []
        for key in layer_map.keys():
            if not isinstance(key, paddle.nn.Layer):
                swap_keys.append(key)
        for key in swap_keys:
            layer_map[layer_map[key]] = key
            layer_map.pop(key)

    @property
    def map(self):
        return self._layer_one2one

    @map.setter
    def map(self, inp):
        assert isinstance(inp, dict), "LayerMap.map wants `dict` obj as input"
        LayerMap.modify_layer_map(inp)
        self._layer_one2one.update(inp)
        self._layer_ignore_sublayer.update(set(inp.keys()))
        self._layer_ignore_sublayer.update(set(inp.values()))

    def ignore(self, inp):
        if isinstance(inp, Iterable):
            self._layer_ignore.update(set(inp))
        elif isinstance(inp, (paddle.nn.Layer, torch.nn.Module)):
            self._layer_ignore.add(inp)
        else:
            raise RuntimeError("Unexpect input type for LayerMap.ignore: {}".format(type(inp)))

    def ignore_recursively(self, layers):
        self._layer_ignore_sublayer.update(set(layers))
        self._layer_ignore.update(set(layers))

    def ignore_class(self, layer):
        ignored = set()
        for sublayer in self.layers(layer):
            if isinstance(sublayer, self._ignore_cls):
                ignored.add(sublayer)
        self._layer_ignore.update(ignored)

    def _traversal_layers(self, net):
        for child in net.children():
            if child not in self._layer_ignore:
                yield child
            if child not in self._layer_ignore_sublayer:
                for sublayer in self._traversal_layers(child):
                    yield sublayer

    def special_init_layers(self):
        return self.map.items()

    def weight_init_layers(self, layer):
        # layers in layer_map should be inited in `special_init`, so they will be skipped here
        layers = [layer]
        if isinstance(layer, paddle.nn.Layer):
            layers.extend(filter(lambda x: x not in self.map.keys(), self._traversal_layers(layer)))
        elif isinstance(layer, torch.nn.Module):
            layers.extend(filter(lambda x: x not in self.map.values(), self._traversal_layers(layer)))
        else:
            raise RuntimeError("Invalid model type: {}".format(type(layer)))
        return layers

    def layers(self, layer):
        layers = [layer]
        layers.extend(self._traversal_layers(layer))
        return layers
