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


# for api -> Layer

import sys, os
import inspect
from functools import partial
from contextlib import contextmanager
from types import ModuleType
import importlib
from importlib.machinery import ModuleSpec
from importlib.abc import MetaPathFinder, Loader

from importlib.machinery import SourceFileLoader, ExtensionFileLoader, PathFinder

SKIP_NAMES = {
    "flatten": "paddle.fluid.layers.utils",
    "wrap": "torch.fx",
    "map_structure": "paddle.fluid.layers.utils",
}


@contextmanager
def ReleaseFinder():
    myfinder = sys.meta_path.pop(0)
    yield
    sys.meta_path = [myfinder] + sys.meta_path


class PaDiffFinder(MetaPathFinder):
    def module_filter(self, name):
        wanted = (
            "paddle",
            "torch",
        )
        for m in wanted:
            if name.startswith(m + ".") or name == m:
                return True, m
        return False, None

    def find_spec(self, fullname, path, target=None):
        if fullname in sys.modules.keys():
            return None
        found, module_type = self.module_filter(fullname)
        if found:
            spec = self.sys_find_spec(fullname, path, target)
            # ExtensionFileLoader loader .so and other lib file
            # if loader is None, python may use a _NamespaceLoader or other way to init
            if spec is not None and not isinstance(spec.loader, (ExtensionFileLoader)) and spec.loader is not None:
                spec._module_type = module_type
                spec.loader = PaDiffLoader(spec.loader)
            return spec
        return None

    def sys_find_spec(self, fullname, path, target=None):
        for finder in sys.meta_path:
            if isinstance(finder, PaDiffFinder):
                continue
            try:
                find_spec = finder.find_spec
            except AttributeError:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None:
                spec._finder = finder
                spec._fullname = fullname
                return spec

        return None


class PaDiffLoader(Loader):
    def __init__(self, _loader):
        self._loader = _loader

    def exec_module(self, module):
        _module_type = module.__spec__._module_type

        self._loader.exec_module(module)

        setattr(module, "_module_type", _module_type)
        setattr(module, "_wrapped_cache", {})
        setattr(module, "_in_api_flag", False)

        # try:
        #     if hasattr(module, '__getattribute__'):
        #         get_method = module.__getattribute__
        #         setattr(module, '__origin_getattribute__', get_method)
        #     setattr(module, '__getattribute__', __my_getattribute__)
        # except:
        #     print(str(module) + "getattribute set failed")

    def create_module(self, spec):
        # return PaDiffModule(spec)
        return None


class PaDiffModule(ModuleType):
    def __init__(self, spec):
        self._module_type = spec._module_type
        self._wrapped_cache = {}
        self._in_api_flag = False
        self._importing = True


def __my_getattribute__(self, name):
    if name in self._wrapped_cache.keys():
        return self._wrapped_cache[name]

    try:
        obj = self._orig_getattribute(name)
    except:
        obj = self.__dict__[name]

    if inspect.ismodule(obj):
        return obj

    # a function, and not in api
    if not self._in_api_flag and inspect.isfunction(obj):
        if name in SKIP_NAMES.keys() and SKIP_NAMES[name] == self._real.__name__:
            self._wrapped_cache[name] = obj
            return obj

        # only when this api is called, a Layer/Module is built
        def wrapped(*args, **kwargs):
            self._in_api_flag = True

            out = obj(*args, **kwargs)

            import paddle, torch

            # from paddle.fluid.layers.utils import flatten
            # only transform apis which ret Tensor
            if out is None or all(
                [not isinstance(x, (paddle.Tensor, torch.Tensor)) for x in paddle.fluid.layers.utils.flatten(out)]
            ):
                self._in_api_flag = False
                return out

            # get Layer and register hook
            if self._module_type == "paddle":

                from .Trainer import paddle_layer_hook

                class PaddleApi(paddle.nn.Layer):
                    def __init__(self, func):
                        super(PaddleApi, self).__init__()
                        self._func = func
                        self._name = name

                    def forward(self, *args, **kwargs):
                        return self._func(*args, **kwargs)

                layer = PaddleApi(obj)
                # need idx to support single step, set idx -1 here to skip api in single step mode
                handle = layer.register_forward_post_hook(partial(paddle_layer_hook, idx=-1))

            elif self._module_type == "torch":

                from .Trainer import torch_layer_hook

                class TorchApi(torch.nn.Module):
                    def __init__(self, func):
                        super(TorchApi, self).__init__()
                        self.func = func
                        self._name = name

                    def forward(self, *args, **kwargs):
                        return self.func(*args, **kwargs)

                layer = TorchApi(obj)
                handle = layer.register_forward_hook(partial(torch_layer_hook, idx=-1))

            else:
                raise RuntimeError("Import Err: module_type not in (paddle, torch)")

            # call forward
            sys.stdout = open(os.devnull, "w")
            out = layer(*args, **kwargs)
            sys.stdout = sys.__stdout__

            # remove hook
            handle.remove()

            self._in_api_flag = False
            return out

        self._wrapped_cache[name] = wrapped
        return self._wrapped_cache[name]

    return obj


sys.meta_path = [PaDiffFinder()] + sys.meta_path


import paddle
import torch

from .auto_diff import auto_diff
from .utils import LayerMap

__all__ = [
    "auto_diff",
    "LayerMap",
]

__version__ = "0.1.0"

from . import configs
