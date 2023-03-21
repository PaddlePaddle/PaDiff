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
from collections import Iterable
from itertools import zip_longest

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


"""
    log utils
"""

diff_log_path = os.path.join(sys.path[0], "diff_log")
__reset_log_dir__ = False


def reset_log_dir():
    if os.path.exists(diff_log_path):
        shutil.rmtree(diff_log_path)
    os.makedirs(diff_log_path)


def clean_log_dir():
    if not os.listdir(diff_log_path):
        os.rmdir(diff_log_path)


def log_file(filename, mode, info):
    global __reset_log_dir__
    if not __reset_log_dir__:
        reset_log_dir()
        __reset_log_dir__ = True

    filepath = os.path.join(sys.path[0], "diff_log", filename)
    with open(filepath, mode) as f:
        f.write(info)


def log(*args):
    print("[AutoDiff]", *args)


def model_repr_info(model):
    extra_lines = []
    extra_repr = model.extra_repr()
    if extra_repr:
        extra_lines = extra_repr.split("\n")
    if len(extra_lines) == 1:
        repr_info = extra_lines[0]
    else:
        repr_info = ""

    retstr = model.__class__.__name__ + "(" + repr_info + ")"
    return retstr


def weight_struct_info(layer, module, paddle_sublayer, torch_submodule):
    t_title = "Torch Model\n" + "=" * 25 + "\n"
    t_retval = print_weight_struct(module, mark=torch_submodule, prefix=[" " * 4])
    t_info = t_title + "\n".join(t_retval)

    p_title = "Paddle Model\n" + "=" * 25 + "\n"
    p_retval = print_weight_struct(layer, mark=paddle_sublayer, prefix=[" " * 4])
    p_info = p_title + "\n".join(p_retval)

    retstr = ""

    if len(p_retval) + len(t_retval) > 100:
        log_file("paddle_weight_check.log", "w", p_info)
        log_file("torch_weight_check.log", "w", t_info)
        retstr += f"Model Struct saved to `{diff_log_path + '/torch_weight_check.log'}` and `{diff_log_path + '/paddle_weight_check.log'}`.\n"
        retstr += "Please view the reports and checkout the layers which is marked with `<---  *** HERE ***` !\n"
    else:
        retstr += t_info
        retstr += "\n"
        retstr += p_info
        retstr += "\n"

    retstr += "\nHint:\n"
    retstr += "      1. check the init order of param or layer in definition is the same.\n"
    retstr += "      2. try to use `LayerMap` to skip the diff in models, you can find the instructions at `https://github.com/PaddlePaddle/PaDiff`.\n"

    return retstr


def print_weight_struct(net, mark=None, prefix=[]):
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

    cur_str += str(net.__class__.__name__)
    if mark is net:
        cur_str += "    <---  *** HERE ***"

    ret_strs = [cur_str]

    children = list(net.children())
    for i, child in enumerate(children):
        pre = " |--- "
        if i == len(children) - 1:
            pre = " +--- "
        prefix.append(pre)
        retval = print_weight_struct(child, mark, prefix)
        ret_strs.extend(retval)
        prefix.pop()

    return ret_strs


def debug_print(net, mark=None, prefix=[]):
    retval = print_weight_struct(net, mark=None, prefix=[])
    print("\n".join(retval))


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
        "cmd": False,
        "use_loss": False,
        "use_opt": False,
        "steps": 1,
    }

    default_options.update(options)
    options.update(default_options)

    if options["single_step"]:
        options["steps"] = 1
        log("  In single_step mode, steps will be set to `1`.")
        options["use_opt"] = False
        log("  In single_step mode, optimizer will not be used.")
    elif options["diff_phase"] == "backward":
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


"""
    LayerMap
"""


def is_wrap_layer(layer):
    if isinstance(layer, paddle.nn.Layer):
        no_param = len(list(layer.parameters(include_sublayers=False))) == 0
        no_buffer = len(list(layer.buffers(include_sublayers=False))) == 0
    elif isinstance(layer, torch.nn.Module):
        no_param = len(list(layer.parameters(recurse=False))) == 0
        no_buffer = len(list(layer.buffers(recurse=False))) == 0
    return no_param and no_buffer


class LayerMap(object):
    def __init__(self):
        self._layer_one2one = {}  # key: torch.nn.Module, value: paddle.nn.Layer
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
            if not isinstance(key, torch.nn.Module):
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

    def ignore_class(self, layer, ign_cls=None):
        ignored = set()
        if ign_cls == None:
            ign_cls = self._ignore_cls
        for sublayer in self.layers_skip_ignore(layer):
            if isinstance(sublayer, ign_cls):
                ignored.add(sublayer)
        self._layer_ignore.update(ignored)

    def _traversal_layers_with_ignore_add_path(self, net, path):
        for name, child in net.named_children():
            path.append(name)
            if (child not in self._layer_ignore and not is_wrap_layer(child)) or (
                child in self.map.keys() or child in self.map.values()
            ):
                if not hasattr(child, "padiff_path"):
                    setattr(child, "padiff_path", ".".join(path))
                yield child
            if child not in self._layer_ignore_sublayer:
                for sublayer in self._traversal_layers_with_ignore_add_path(child, path):
                    yield sublayer
            path.pop()

    def _traversal_layers_with_ignore(self, net):
        for child in net.children():
            if (child not in self._layer_ignore and not is_wrap_layer(child)) or (
                child in self.map.keys() or child in self.map.values()
            ):
                yield child
            if child not in self._layer_ignore_sublayer:
                for sublayer in self._traversal_layers_with_ignore(child):
                    yield sublayer

    def special_init_layers(self):
        return self.map.items()

    def weight_init_layers(self, layer):
        # layers in layer_map should be inited in `special_init`, so they will be skipped here
        layers = [layer]
        path = [layer.__class__.__name__]
        if isinstance(layer, paddle.nn.Layer):
            layers.extend(
                filter(lambda x: x not in self.map.values(), self._traversal_layers_with_ignore_add_path(layer, path))
            )
        elif isinstance(layer, torch.nn.Module):
            layers.extend(
                filter(lambda x: x not in self.map.keys(), self._traversal_layers_with_ignore_add_path(layer, path))
            )
        else:
            raise RuntimeError("Invalid model type: {}".format(type(layer)))
        return layers

    def layers_skip_ignore(self, layer):
        # NOTICE: root level always in return vals, though it could be a wrap layer
        layers = [layer]
        layers.extend(self._traversal_layers_with_ignore(layer))
        return layers


def auto_LayerMap(layer, module):
    """
    This function will try to find components which support special init, and add them to layer_map automatically.

    NOTICE: auto_LayerMap suppose that all sublayers/submodules are defined in same order, if not, auto_LayerMap may not work correctly.
    """

    from .special_init import global_special_init_pool as init_pool
    from .special_init import build_name
    from itertools import zip_longest

    def _traversal_layers(net, path, registered):
        for name, child in net.named_children():
            path.append(name)
            if child.__class__.__name__ in registered:
                yield (child, ".".join(path))
            if child.__class__.__name__ not in registered:
                for sublayer, ret_path in _traversal_layers(child, path, registered):
                    yield (sublayer, ret_path)
            path.pop()

    paddle_layers = list(_traversal_layers(layer, [layer.__class__.__name__], init_pool.registered_paddle_layers))
    torch_modules = list(_traversal_layers(module, [module.__class__.__name__], init_pool.registered_torch_modules))

    layer_map = LayerMap()

    log("auto_LayerMap Start searching...")
    for paddle_info, torch_info in zip_longest(paddle_layers, torch_modules, fillvalue=None):
        if paddle_info is None or torch_info is None:
            log(
                "The number of registered paddle sublayer and torch submodule is not the same! Check your model struct first !!!"
            )
            log("auto_LayerMap FAILED!!!")
            return None
        paddle_layer, paddle_path = paddle_info
        torch_module, torch_path = torch_info
        paddle_name = paddle_layer.__class__.__name__
        torch_name = torch_module.__class__.__name__
        name = build_name(paddle_name, torch_name)
        if name in init_pool.funcs.keys():
            layer_map.map = {torch_module: paddle_layer}
            print(f"Add:    paddle `{paddle_name}` at `{paddle_path}` <==> torch `{torch_name}` at `{torch_path}`.")
        else:
            log("When generating LayerMap in order, find that paddle sublayer can not matchs torch submodule.")
            log(f"    paddle: `{paddle_name}` at `{paddle_path}`")
            log(f"    torch:  `{torch_name}` at `{torch_path}`")
            log("auto_LayerMap FAILED!!!")
            return None
    log("auto_LayerMap SUCCESS!!!")
    return layer_map
