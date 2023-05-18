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
from .abstracts import ProxyModel


class LayerMap(object):
    def __init__(self):
        self._layer_map = {}  # key: component of base_model, value: component of src_model
        self._ignored_layers = set()  # ignore layer in this set
        self._sublayer_ignored_layers = set()  # ignore sublayer of layers in this set (do not include themselves)

    @property
    def map(self):
        return self._layer_map

    @map.setter
    def map(self, inputs):
        assert isinstance(inputs, dict), "LayerMap.map wants `dict` obj as input"
        # user should give {src_model: base_model} as input, (because src_model is the first param)
        # but we want {base_model: src_model}, so swap the kv pairs
        new_inputs = {}
        for k, v in inputs.items():
            new_inputs[v] = k
        self._layer_map.update(new_inputs)
        self._sublayer_ignored_layers.update(set(inputs.keys()))
        self._sublayer_ignored_layers.update(set(inputs.values()))

    def ignore(self, inputs):
        if isinstance(inputs, (list, tuple)):
            self._ignored_layers.update(set(inputs))
        elif isinstance(inputs, (paddle.nn.Layer, torch.nn.Module)):
            self._ignored_layers.add(inputs)
        else:
            raise RuntimeError("Unexpected input type for LayerMap.ignore: {}".format(type(inputs)))

    def ignore_recursively(self, layers):
        if isinstance(layers, (paddle.nn.Layer, torch.nn.Module)):
            layers = [layers]
        self._sublayer_ignored_layers.update(set(layers))
        self._ignored_layers.update(set(layers))

    def ignore_class(self, layer, ign_cls):
        ignored = set()
        for sublayer in self.layers_skip_ignore(layer):
            if isinstance(sublayer, ign_cls):
                ignored.add(sublayer)
        self._ignored_layers.update(ignored)

    def special_init_layers(self):
        # TODO: return proxy_model
        map_items = self.map.items()

        def new_generator():
            for k, v in map_items:
                yield ProxyModel.create_from(v), ProxyModel.create_from(k)

        return new_generator()

    def layers_in_map(self):
        return chain(self.map.keys(), self.map.values())

    def layers_skip_ignore(self, layer):
        layers = [layer]
        layers.extend(self._traversal_layers_with_ignore(layer))
        return layers

    def weight_init_layers(self, layer):
        # layers in layer_map should be inited in `special_init`, so they will be skipped here
        layers = filter(lambda x: x.model not in self.layers_in_map(), self.layers_skip_ignore(layer))
        return layers

    def struct_hook_layers(self, layer):
        layers = [layer]
        layers.extend(self._traversal_layers_for_model_struct(layer))
        return layers

    def _traversal_layers_with_ignore(self, model):
        for child_model in model.children():
            child = child_model.model
            if (child not in self._ignored_layers and not is_wrap_layer(child_model)) or child in self.layers_in_map():
                if not hasattr(child, "no_skip"):
                    setattr(child, "no_skip", True)
                yield child_model
            if child not in self._sublayer_ignored_layers:
                for sublayer in self._traversal_layers_with_ignore(child_model):
                    yield sublayer

    def _traversal_layers_for_model_struct(self, model):
        # any in self._sublayer_ignored_layers should be returned
        # for checking whether an api should be record (is under a layer in _sublayer_ignored_layers)
        for child_model in model.children():
            child = child_model.model
            if (
                child not in self._ignored_layers and not is_wrap_layer(child_model)
            ) or child in self._sublayer_ignored_layers:
                if not hasattr(child, "no_skip"):
                    setattr(child, "no_skip", True)
                yield child_model
            if child not in self._sublayer_ignored_layers:
                for sublayer in self._traversal_layers_for_model_struct(child_model):
                    yield sublayer

    def auto(self, src_model, base_model):
        """
        This method will try to find components which support special init, and add them to layer_map automatically.
        NOTICE: LayerMap.auto suppose that all sublayers/submodules are defined in same order, if not, this method may not work correctly.
        """

        def _traversal_layers(model, path, registered):
            for name, child in model.named_children():
                path.append(name)
                if child.fullname in registered and child.model not in self._ignored_layers:
                    yield (child, ".".join(path))
                if child.fullname not in registered and child.model not in self._sublayer_ignored_layers:
                    for sublayer, ret_path in _traversal_layers(child, path, registered):
                        yield (sublayer, ret_path)
                path.pop()

        # ProxyModel.create_from will do assert check for models
        src_model = ProxyModel.create_from(src_model)
        base_model = ProxyModel.create_from(base_model)

        src_submodels = list(_traversal_layers(src_model, [src_model.class_name], init_pool.registered_src_models))
        base_submodels = list(_traversal_layers(base_model, [base_model.class_name], init_pool.registered_base_models))

        _map = {}

        log("auto update LayerMap start searching...\n")

        for src_info, base_info in zip_longest(src_submodels, base_submodels, fillvalue=None):
            if src_info is None or base_info is None:
                print(
                    "\nError: The number of submodels which need special init is not the same! Check your model struct first!"
                )
                log("auto update LayerMap FAILED!!!\n")
                return False

            src_model, src_path = src_info
            base_model, base_path = base_info
            name = build_name(src_model.model_type, src_model.class_name, base_model.model_type, base_model.class_name)
            if name in init_pool.funcs.keys():
                _map.update({src_model.model: base_model.model})
                print(
                    f"++++    src_model `{src_model.fullname}` at `{src_path}` <==> base_model `{base_model.fullname}` at `{base_path}`    ++++"
                )
            else:
                print("\nError: When generating LayerMap in order, find that src_model can not matchs base_model.")
                print(f"    src_model: `{src_model.fullname}` at `{src_path}`")
                print(f"    base_model:  `{base_model.fullname}` at `{base_path}`")
                log("auto update LayerMap FAILED!!!\n")
                return False
        print()
        log("auto update LayerMap SUCCESS!!!\n")

        self.map = _map
        return True

    @staticmethod
    def create_from(layer_map=None):
        if layer_map is None:
            layer_map = LayerMap()
        elif isinstance(layer_map, dict):
            new_map = LayerMap()
            new_map.map = layer_map
            layer_map = new_map
        else:
            assert isinstance(layer_map, LayerMap), "Invalid Argument."
        return layer_map


def is_wrap_layer(model):
    return len(list(model.parameters(recursively=False))) == 0
