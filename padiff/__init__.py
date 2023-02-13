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


SKIP_NAMES = {
    "flatten": "paddle.fluid.layers.utils",
    "wrap": "torch.fx",
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
        found, _module_type = self.module_filter(fullname)
        if found:
            with ReleaseFinder():
                _real = importlib.import_module(fullname)
                sys.modules.pop(fullname)
                spec = ModuleSpec(fullname, PaDiffLoader())
                spec._real = _real
                spec._module_type = _module_type
                return spec
        return None


class PaDiffLoader(Loader):
    def exec_module(self, module):
        pass

    def create_module(self, spec):
        return PaDiffModule(spec)


class PaDiffModule(ModuleType):
    def __init__(self, spec):
        self.name = spec.name
        self._real = spec._real
        self._module_type = spec._module_type

        self._wrapped_cache = {}

        self._in_api_flag = False

    def __getattr__(self, name: str):
        if name in self._wrapped_cache.keys():
            return self._wrapped_cache[name]
        try:
            obj = self._real.__getattribute__(name)
        except:
            return self._real.__getattr__(name)

        if inspect.ismodule(obj):
            spec = ModuleSpec(self.name + "." + name, PaDiffLoader())
            spec._real = obj
            spec._module_type = self._module_type
            obj = PaDiffModule(spec)
            self._wrapped_cache[name] = obj
            return self._wrapped_cache[name]

        # a function, and not in api
        if not self._in_api_flag and inspect.isfunction(obj):
            if name in SKIP_NAMES.keys() and SKIP_NAMES[name] == self.name:
                self._wrapped_cache[name] = obj
                return self._wrapped_cache[name]

            # only when this api is called, a Layer/Module is built
            def wrapped(*args, **kwargs):
                from .Trainer import paddle_layer_hook, torch_layer_hook

                self._in_api_flag = True

                out = obj(*args, **kwargs)

                # only transform apis which ret Tensor
                if out is None or all(
                    [not isinstance(x, (paddle.Tensor, torch.Tensor)) for x in paddle.fluid.layers.utils.flatten(out)]
                ):
                    self._in_api_flag = False
                    return out

                # get Layer and register hook
                if self._module_type == "paddle":

                    class PaddleApi(paddle.nn.Layer):
                        def __init__(self, func):
                            super(PaddleApi, self).__init__()
                            self._func = func

                        def forward(self, *args, **kwargs):
                            return self._func(*args, **kwargs)

                    layer = PaddleApi(obj)
                    # need idx to support single step, set idx -1 here to skip api in single step mode
                    handle = layer.register_forward_post_hook(partial(paddle_layer_hook, idx=-1))

                elif self._module_type == "torch":

                    class TorchApi(torch.nn.Module):
                        def __init__(self, func):
                            super(TorchApi, self).__init__()
                            self.func = func

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


from .auto_diff import auto_diff
from .utils import LayerMap

__all__ = [
    "auto_diff",
    "LayerMap",
]

__version__ = "0.1.0"

from . import configs

import paddle
import torch
