# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import torch
from itertools import zip_longest, chain

from .utils import log
from .special_init import build_name, global_special_init_pool as init_pool


class LayerMap(object):
    def __init__(self):
        self._layer_one2one = {}  # key: torch.nn.Module, value: paddle.nn.Layer
        self._layer_ignore = set()  # ignore layer in this set
        self._layer_ignore_sublayer = set()  # ignore sublayer of layers in this set (do not include themselves)

    @property
    def map(self):
        return self._layer_one2one

    @map.setter
    def map(self, inp):
        assert isinstance(inp, dict), "LayerMap.map wants `dict` obj as input"
        new_inp = {}
        for k, v in inp.items():
            new_inp[v] = k
        self._layer_one2one.update(new_inp)
        self._layer_ignore_sublayer.update(set(inp.keys()))
        self._layer_ignore_sublayer.update(set(inp.values()))

    def ignore(self, inp):
        if isinstance(inp, (list, tuple)):
            self._layer_ignore.update(set(inp))
        elif isinstance(inp, (paddle.nn.Layer, torch.nn.Module)):
            self._layer_ignore.add(inp)
        else:
            raise RuntimeError("Unexpect input type for LayerMap.ignore: {}".format(type(inp)))

    def ignore_recursively(self, layers):
        if isinstance(layers, (paddle.nn.Layer, torch.nn.Module)):
            layers = [layers]
        self._layer_ignore_sublayer.update(set(layers))
        self._layer_ignore.update(set(layers))

    def ignore_class(self, layer, ign_cls):
        ignored = set()
        for sublayer in self.layers_skip_ignore(layer):
            if isinstance(sublayer, ign_cls):
                ignored.add(sublayer)
        self._layer_ignore.update(ignored)

    def _traversal_layers_with_ignore(self, model):
        for child_padiff_model in model.children():
            child = child_padiff_model.model
            if (child not in self._layer_ignore and not is_wrap_layer(child_padiff_model)) or (
                child in self.map.keys() or child in self.map.values()
            ):
                if not hasattr(child, "no_skip"):
                    setattr(child, "no_skip", True)
                yield child_padiff_model
            if child not in self._layer_ignore_sublayer:
                for sublayer in self._traversal_layers_with_ignore(child_padiff_model):
                    yield sublayer

    def _traversal_layers_for_model_struct(self, model):
        # any in self._layer_ignore_sublayer should be returned
        # to check whether an api should be record
        for child_padiff_model in model.children():
            child = child_padiff_model.model
            if (child not in self._layer_ignore and not is_wrap_layer(child_padiff_model)) or (
                child in self._layer_ignore_sublayer
            ):
                if not hasattr(child, "no_skip"):
                    setattr(child, "no_skip", True)
                yield child_padiff_model
            if child not in self._layer_ignore_sublayer:
                for sublayer in self._traversal_layers_for_model_struct(child_padiff_model):
                    yield sublayer

    def special_init_layers(self):
        # TODO: return padiff model
        return self.map.items()

    def layers_in_map(self):
        return chain(self.map.keys(), self.map.values())

    def weight_init_layers(self, layer):
        # layers in layer_map should be inited in `special_init`, so they will be skipped here
        layers = [layer]
        layers.extend(filter(lambda x: x.model not in self.layers_in_map(), self._traversal_layers_with_ignore(layer)))
        return layers

    def layers_skip_ignore(self, layer):
        layers = [layer]
        layers.extend(self._traversal_layers_with_ignore(layer))
        return layers

    def struct_hook_layers(self, layer):
        layers = [layer]
        layers.extend(self._traversal_layers_for_model_struct(layer))
        return layers

    def auto(self, layer, module):
        """
        This function will try to find components which support special init, and add them to layer_map automatically.

        NOTICE: LayerMap.auto suppose that all sublayers/submodules are defined in same order, if not, this method may not work correctly.
        """

        def _traversal_layers(model, path, registered):
            for name, child in model.named_children():
                path.append(name)
                if child.__class__.__name__ in registered and child not in self._layer_ignore:
                    yield (child, ".".join(path))
                if child.__class__.__name__ not in registered and child not in self._layer_ignore_sublayer:
                    for sublayer, ret_path in _traversal_layers(child, path, registered):
                        yield (sublayer, ret_path)
                path.pop()

        paddle_layers = list(_traversal_layers(layer, [layer.__class__.__name__], init_pool.registered_paddle_layers))
        torch_modules = list(
            _traversal_layers(module, [module.__class__.__name__], init_pool.registered_torch_modules)
        )

        _map = {}

        log("auto update LayerMap start searching...\n")

        for paddle_info, torch_info in zip_longest(paddle_layers, torch_modules, fillvalue=None):
            if paddle_info is None or torch_info is None:
                print(
                    "\nError: The number of registered paddle sublayer and torch submodule is not the same! Check your model struct first!"
                )
                log("auto update LayerMap FAILED!!!\n")
                return

            paddle_layer, paddle_path = paddle_info
            torch_module, torch_path = torch_info
            paddle_name = paddle_layer.__class__.__name__
            torch_name = torch_module.__class__.__name__
            name = build_name(paddle_name, torch_name)
            if name in init_pool.funcs.keys():
                _map.update({torch_module: paddle_layer})
                print(
                    f"++++    paddle `{paddle_name}` at `{paddle_path}` <==> torch `{torch_name}` at `{torch_path}`."
                )
            else:
                print(
                    "\nError: When generating LayerMap in order, find that paddle sublayer can not matchs torch submodule."
                )
                print(f"    paddle: `{paddle_name}` at `{paddle_path}`")
                print(f"    torch:  `{torch_name}` at `{torch_path}`")
                log("auto update LayerMap FAILED!!!\n")
                return
        print()
        log("auto update LayerMap SUCCESS!!!\n")

        self.map = _map


def is_wrap_layer(model):
    return len(list(model.parameters(recursively=False))) == 0


def init_LayerMap(layer_map):
    if layer_map is None:
        layer_map = LayerMap()
    elif isinstance(layer_map, dict):
        new_map = LayerMap()
        new_map.map = layer_map
        layer_map = new_map
    else:
        assert isinstance(layer_map, LayerMap), "Invalid Argument."
    return layer_map
