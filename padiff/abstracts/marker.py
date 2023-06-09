import paddle
import torch
from types import MethodType

class Marker:
    def __init__(self, proxy_model):
        self.proxy_model = proxy_model

        self.black_list = set()
        self.black_list_recursively = set()
        self.white_list = set()
        self.white_list_recursively = set()
        self.use_white_list = False

        self.layer_map = []

    def update_black_list(self, layers, mode="all"):
        assert mode in ("self", "sublayers", "all")
        if isinstance(layers, (paddle.nn.Layer, torch.nn.Module)):
            layers = [layers]
        if mode in ("self", "all"):
            self.black_list.update(set(layers))
        if mode in ("sublayers", "all"):
            self.black_list_recursively.update(set(layers))

    def update_white_list(self, layers, mode="self"):
        assert mode in ("self", "sublayers", "all")
        if isinstance(layers, (paddle.nn.Layer, torch.nn.Module)):
            layers = [layers]
        if mode in ("self", "all"):
            self.white_list.update(set(layers))
        if mode in ("sublayers", "all"):
            self.white_list_recursively.update(set(layers))
        self.use_white_list = True
    
    def set_layer_map(self, layer_map):
        self.layer_map = layer_map

    def update_black_list_with_class(self, layer_class, recursively=True):
        pass

    def update_white_list_with_class(self, layer_class, recursively=False):
        pass

    def traversal_layers(self, include_self=True):
        for model in traversal_layers(self.proxy_model, self, include_self):
            yield model
    
    def traversal_for_hook(self):
        for model in traversal_for_hook(self.proxy_model, self):
            yield model


def is_wrap_layer(model):
    return len(list(model.parameters(recursively=False))) == 0


def traversal_prototype(fn0, fn1):
    def inner(model, marker):
        for child in model.children():
            if fn0(child, marker):
                yield child
            if fn1(child, marker):
                for sublayer in inner(child, marker):
                    yield sublayer
    return inner


traversal_all = traversal_prototype(
    fn0 = lambda model, marker: True,
    fn1 = lambda model, marker: True,
)
traversal_with_black_list = traversal_prototype(
    fn0 = lambda model, marker: model.model not in marker.black_list and not is_wrap_layer(model),
    fn1 = lambda model, marker: model.model not in marker.black_list_recursively,
)
traversal_layers_for_model_struct = traversal_prototype(
    fn0 = lambda model, marker: (model.model not in marker.black_list and not is_wrap_layer(model)) or model.model in marker.black_list_recursively,
    fn1 = lambda model, marker: model.model not in marker.black_list_recursively
)

def traversal_with_white_list(model, marker):
    for child in model.children():
        if child.model in marker.white_list:
            yield child
        if child.model in marker.white_list_recursively:
            for sublayer in traversal_all(child, marker):
                yield sublayer
        else:
            for sublayer in traversal_with_white_list(child, marker):
                yield sublayer


def traversal_layers(model, marker, include_self=True):
    if include_self:
        yield model
    if marker.use_white_list:
        for mod in traversal_with_white_list(model, marker):
            yield mod
    else:
        for mod in traversal_with_black_list(model, marker):
            yield mod

def traversal_for_hook(model, marker):
    yield model
    if marker.use_white_list:
        for mod in traversal_with_white_list(model, marker):
            yield mod
    else:
        for mod in traversal_layers_for_model_struct(model, marker):
            yield mod