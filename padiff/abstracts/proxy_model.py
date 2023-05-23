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
import re

from .proxy_parameter import ProxyParam


class ProxyModel:
    def __init__(self, model, name, model_type):
        self.model = model
        self.model_type = model_type
        self.name = name

    @staticmethod
    def create_from(model, name=None):
        if name is None:
            name = model.__class__.__name__
        if isinstance(model, ProxyModel):
            return model
        elif isinstance(model, paddle.nn.Layer):
            return PaddleModel(model, name)
        elif isinstance(model, torch.nn.Module):
            return TorchModel(model, name)
        else:
            raise RuntimeError(f"Can not create ProxyModel from {type(model)}")

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def model_repr_info(self):
        model = self.model
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

    @property
    def path_info(self):
        return self.model.path_info

    @property
    def class_name(self):
        return self.model.__class__.__name__

    @property
    def fullname(self):
        return f"{self.model_type}::{self.class_name}"

    def __str__(self, *args, **kwargs):
        return f"Model({self.fullname})"

    def __repr__(self, *args, **kwargs):
        return f"Model({self.fullname})"

    def parameters(self):
        raise NotImplementedError()

    def named_parameters(self):
        raise NotImplementedError()

    # child sublayers, do not include self
    def children(self):
        raise NotImplementedError()

    def named_children(self):
        raise NotImplementedError()

    # get sublayers recursively include self
    def submodels(self):
        raise NotImplementedError()

    def named_submodels(self):
        raise NotImplementedError()

    def get_device(self):
        raise NotImplementedError()

    def to_cpu(self):
        raise NotImplementedError()

    def to(self, device):
        raise NotImplementedError()

    def register_forward_pre_hook(self, hook):
        raise NotImplementedError()

    def register_forward_post_hook(self, hook):
        raise NotImplementedError()


class PaddleModel(ProxyModel):
    def __init__(self, model, name):
        super(PaddleModel, self).__init__(model, name, "paddle")

    def parameters(self, recursively):
        origin_iter = self.model.parameters(include_sublayers=recursively)
        return deco_iter(origin_iter, ProxyParam.create_from)

    def named_parameters(self, recursively):
        origin_iter = self.model.named_parameters(include_sublayers=recursively)
        return deco_iter(origin_iter, ProxyParam.create_from)

    def children(self):
        origin_iter = self.model.children()
        return deco_iter(origin_iter, ProxyModel.create_from)

    def named_children(self):
        origin_iter = self.model.named_children()
        return deco_iter(origin_iter, ProxyModel.create_from)

    def submodels(self):
        origin_iter = self.model.sublayers(True)
        return deco_iter(origin_iter, ProxyModel.create_from)

    def named_submodels(self):
        origin_iter = self.model.named_sublayers(include_self=True)
        return deco_iter(origin_iter, ProxyModel.create_from)

    def get_device(self):
        place_str = str(self.model.parameters()[0].place)
        return re.match(r"Place\((.*)\)", place_str).group(1)

    def to_cpu(self):
        self.model.to("cpu")
        paddle.device.cuda.empty_cache()

    def to(self, device):
        self.model.to(device)

    def register_forward_pre_hook(self, hook):
        return self.model.register_forward_pre_hook(hook)

    def register_forward_post_hook(self, hook):
        return self.model.register_forward_post_hook(hook)


class TorchModel(ProxyModel):
    def __init__(self, model, name):
        super(TorchModel, self).__init__(model, name, "torch")

    def parameters(self, recursively):
        origin_iter = self.model.parameters(recurse=recursively)
        return deco_iter(origin_iter, ProxyParam.create_from)

    def named_parameters(self, recursively):
        origin_iter = self.model.named_parameters(recurse=recursively)
        return deco_iter(origin_iter, ProxyParam.create_from)

    def children(self):
        origin_iter = self.model.children()
        return deco_iter(origin_iter, ProxyModel.create_from)

    def named_children(self):
        origin_iter = self.model.named_children()
        return deco_iter(origin_iter, ProxyModel.create_from)

    def submodels(self):
        origin_iter = self.model.modules()
        return deco_iter(origin_iter, ProxyModel.create_from)

    def named_submodels(self):
        origin_iter = self.model.named_modules(remove_duplicate=True)
        return deco_iter(origin_iter, ProxyModel.create_from)

    def get_device(self):
        return next(self.model.parameters()).device

    def to_cpu(self):
        self.model.to("cpu")
        torch.cuda.empty_cache()

    def to(self, device):
        self.model.to(device)

    def register_forward_pre_hook(self, hook):
        return self.model.register_forward_pre_hook(hook)

    def register_forward_post_hook(self, hook):
        return self.model.register_forward_hook(hook)


def deco_iter(iterator, fn):
    def new_fn(obj):
        try:
            return fn(obj)
        except:
            return obj

    def new_generator():
        for obj in iterator:
            if isinstance(obj, (tuple, list)):
                yield tuple(map(new_fn, obj))
            else:
                yield new_fn(obj)

    return new_generator()
