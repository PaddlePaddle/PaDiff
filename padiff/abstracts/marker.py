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
        self.init_traversal_tools()

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

    def init_traversal_tools(self):
        self.traversal_all = traversal_prototype( 
            self,
            fn0 = lambda x, y: True,
            fn1 = lambda x, y: True,
        )
        self.traversal_with_black_list = traversal_prototype(
            self,
            fn0 = lambda root, child: child.model not in root.black_list and not child.is_wrap_layer(),
            fn1 = lambda root, child: child.model not in root.black_list_recursively,
        )
        self.traversal_layers_for_model_struct = traversal_prototype(
            self,
            fn0 = lambda root, child: (child.model not in root.black_list and not is_wrap_layer(child)) or child.model in root.black_list_recursively,
            fn1 = lambda root, child: child.model not in root.black_list_recursively
        )
        
        def _traversal_with_white_list(model):
            for child in model.children():
                if child.model in self.white_list:
                    yield child
                if child.model in self.white_list_recursively:
                    for sublayer in self.traversal_all(child):
                        yield sublayer
                else:
                    for sublayer in _traversal_with_white_list(child):
                        yield sublayer

        self.traversal_with_white_list = _traversal_with_white_list

    def traversal_layers(self, model, include_self=True):
        if include_self:
            yield model
        if self.use_white_list:
            for mod in self.traversal_with_white_list(model):
                yield mod
        else:
            for mod in self.traversal_with_black_list(model):
                yield mod

    def traversal_for_hook(self, model):
        yield model
        if self.use_white_list:
            for mod in self.traversal_with_white_list(model):
                yield mod
        else:
            for mod in self.traversal_layers_for_model_struct(model):
                yield mod


def is_wrap_layer(model):
    return len(list(model.parameters(recursively=False))) == 0


def traversal_prototype(marker, fn0, fn1):
    def inner(model):
        for child in model.children():
            if fn0(marker, child):
                yield child
            if fn1(marker, child):
                for sublayer in inner(child):
                    yield sublayer
    return inner
