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
from types import ModuleType, MethodType
import importlib
from importlib.machinery import ModuleSpec
from importlib.abc import MetaPathFinder, Loader

from importlib.machinery import SourceFileLoader, ExtensionFileLoader, PathFinder

SKIP_NAMES = {
    "flatten": "paddle.fluid.layers.utils",
    "wrap": "torch.fx",
    "map_structure": "paddle.fluid.layers.utils",
}

WANT_WRAP = (
    "paddle",
    "paddle.nn.functional",
    "torch",
    "torch.nn.functional",
    "torch.spectial",
)


def module_filter(name):
    if name in WANT_WRAP:
        return True, name.partition(".")[0]
    return False, None


class PaDiffFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in sys.modules.keys():
            return None
        found, module_type = module_filter(fullname)
        if found:
            spec = self.sys_find_spec(fullname, path, target)
            # ExtensionFileLoader loader .so and other lib file
            # if loader is None, python may use a _NamespaceLoader or other way to init
            if spec is not None and not isinstance(spec.loader, (ExtensionFileLoader)) and spec.loader is not None:
                spec._module_type = module_type
                spec.loader = PaDiffLoader(spec.loader)
            return spec
        return None

    # find module by using sys defined finders
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


def wrap_func(fullname, func):
    def wrapped(*args, **kwargs):

        if fullname.startswith("paddle"):

            from .hooks import paddle_api_hook

            class PaddleApi(paddle.nn.Layer):
                def __init__(self, func):
                    super(PaddleApi, self).__init__()
                    self._func = func
                    self.__name__ = fullname
                    self.__api__ = True

                def forward(self, *args, **kwargs):
                    return self._func(*args, **kwargs)

            layer = PaddleApi(func)
            # need idx to support single step, set idx -1 here to skip api in single step mode
            handle = layer.register_forward_post_hook(partial(paddle_api_hook, idx=-1))

        elif fullname.startswith("torch"):

            from .hooks import torch_api_hook

            class TorchApi(torch.nn.Module):
                def __init__(self, func):
                    super(TorchApi, self).__init__()
                    self.func = func
                    self.__name__ = fullname
                    self.__api__ = True

                def forward(self, *args, **kwargs):
                    return self.func(*args, **kwargs)

            layer = TorchApi(func)
            handle = layer.register_forward_hook(partial(torch_api_hook, idx=-1))

        else:
            raise RuntimeError("Import Err: module_type not in (paddle, torch)")

        out = layer(*args, **kwargs)

        handle.remove()

        return out

    return wrapped


class PaDiffLoader(Loader):
    def __init__(self, _loader):
        self._loader = _loader

    def exec_module(self, module):
        self._loader.exec_module(module)

        if hasattr(module, "__all__"):
            for name in module.__all__:
                if name.startswith("_"):
                    continue
                obj = module.__dict__[name]
                if inspect.isfunction(obj):
                    module.__dict__[name] = wrap_func(module.__name__ + "." + name, obj)
        else:
            for k, v in module.__dict__.items():
                if k.startswith("_"):
                    continue
                if inspect.isfunction(v):
                    module.__dict__[k] = wrap_func(module.__name__ + "." + k, v)
                elif inspect.isbuiltin(v) and module.__name__.startswith("torch"):
                    module.__dict__[k] = wrap_func(module.__name__ + "." + k, v)

    def create_module(self, spec):
        # return PaDiffModule(spec)
        return None


for name in WANT_WRAP:
    if name in sys.modules.keys():
        module = sys.modules[name]
        if hasattr(module, "__all__"):
            for name in module.__all__:
                if name.startswith("_"):
                    continue
                obj = module.__dict__[name]
                if inspect.isfunction(obj):
                    module.__dict__[name] = wrap_func(module.__name__ + "." + name, obj)
        else:
            for k, v in module.__dict__.items():
                if k.startswith("_"):
                    continue
                if inspect.isfunction(v):
                    module.__dict__[k] = wrap_func(k, v)
                elif inspect.isbuiltin(v) and module.__name__.startswith("torch"):
                    module.__dict__[k] = wrap_func(k, v)


sys.meta_path = [PaDiffFinder()] + sys.meta_path

__version__ = "0.1.0"

import paddle
import torch

from . import configs
from .utils import LayerMap
from .auto_diff import auto_diff

__all__ = [
    "auto_diff",
    "LayerMap",
]
