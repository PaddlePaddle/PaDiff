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

from itertools import zip_longest

import numpy

from .utils import (
    log,
    map_for_each_sublayer,
    LayerMap,
    weight_struct_info,
    PadiffModel,
)
from .file_loader import global_yaml_loader as yamls

from .special_init import global_special_init_pool as init_pool
from .special_init import build_name


def process_each_weight(process, model_0, model_1, layer_map=LayerMap()):
    """
    Apply process for each pair of parameters in layer(paddle) and module(torch)

    Args:
        process (function): process applied to parameters
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
        layer_map (dict, optional): manually map paddle layer and torch module.
    """

    def _process_runner(
        process,
        submodels,
        param_names,
        params,
    ):
        settings = yamls.get_weight_settings(submodels[0], submodels[1], param_names[0])

        process(
            submodels,
            param_names,
            params,
            settings,
        )

    submodels_0 = layer_map.weight_init_layers(model_0)
    submodels_1 = layer_map.weight_init_layers(model_1)

    for submodel_0, submodel_1 in zip_longest(submodels_0, submodels_1, fillvalue=None):
        if submodel_0 is None or submodel_1 is None:
            raise RuntimeError("Torch and Paddle return difference number of sublayers. Check your model.")
        for (param_name_0, param_0), (param_name_1, param_1) in zip(
            submodel_0.named_parameters(recursively=False),
            submodel_1.named_parameters(recursively=False),
        ):
            try:
                _process_runner(
                    process,
                    (submodel_0, submodel_1),
                    (param_name_0, param_name_1),
                    (param_0, param_1),
                )
            except Exception as e:
                err_str = f"Error occured between:\n"
                err_str += f"    Model[0] {submodel_0.fullname}: {submodel_0.model_repr_info()}\n"
                err_str += f"            {submodel_0.padiff_path + '.' + param_name_0}\n"
                err_str += f"    Model[1] {submodel_1.fullname}: {submodel_1.model_repr_info()}\n"
                err_str += f"            {submodel_1.padiff_path + '.' + param_name_1}\n"
                err_str += f"{type(e).__name__ + ':  ' + str(e)}\n"
                err_str += weight_struct_info(model_0, model_1, submodel_0, submodel_1)
                raise RuntimeError(err_str)


def shape_check(
    submodels,
    param_names,
    params,
    settings,
):
    shape_0 = params[0].shape()
    shape_1 = params[1].shape()
    if settings["transpose"]:
        shape_1.reverse()
    assert (
        shape_0 == shape_1
    ), f"Shape of param `{param_names[0]}` in first model and param `{param_names[1]}` in second model is not the same. {shape_0} vs {shape_1}\n"


def _assign_weight(
    submodels,
    param_names,
    params,
    settings,
):
    shape_check(
        submodels,
        param_names,
        params,
        settings,
    )
    np_value = params[1].numpy()
    if settings["transpose"]:
        np_value = numpy.transpose(np_value)

    params[0].set_data(np_value)


def assign_weight(target_model, source_model, layer_map=LayerMap()):
    """
    Init weights of layer(paddle) and module(torch) with same value

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
    """
    assert isinstance(target_model, PadiffModel), "The first param of assign_weight should be a PadiffModel"
    assert isinstance(source_model, PadiffModel), "The second param of assign_weight should be a PadiffModel"

    # TODO: special init is not nessesary for current requirement, so just skip here
    # need update later
    if target_model.model_type == "paddle" and source_model.model_type == "torch":
        for torch_submodule, paddle_sublayer in layer_map.special_init_layers():
            paddle_layer_name = paddle_sublayer.__class__.__name__
            torch_module_name = torch_submodule.__class__.__name__
            key_name = build_name(paddle_layer_name, torch_module_name)
            if key_name not in init_pool.funcs.keys():
                log(
                    "*** Special init paddle layer `{}` and torch module `{}` is not supported ***".format(
                        paddle_layer_name, torch_module_name
                    )
                )
                log("    Checkout the parameters are inited by yourself")
                log("    ,or you can register your init method!")
            else:
                try:
                    init_pool.funcs[key_name](paddle_sublayer, torch_submodule)
                except Exception as e:
                    print(
                        f"Special init paddle layer `{paddle_layer_name}` and torch module `{torch_module_name}` failed."
                    )
                    print(type(e).__name__ + ":  " + str(e))
                    log("Assign weight Failed !!!")
                    return False

    try:
        process_each_weight(_assign_weight, target_model, source_model, layer_map)
        log("Assign weight success !!!")
        return True
    except Exception as e:
        log("Assign weight Failed !!!\n")
        print(type(e).__name__ + ":  " + str(e))
        return False


def remove_inplace(models):
    """
    Set `inplace` tag to `False` for torch module

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
    """

    def _remove_inplace(model_1, model_2):
        if hasattr(model_1, "inplace"):
            model_1.inplace = False
        if hasattr(model_2, "inplace"):
            model_2.inplace = False

    map_for_each_sublayer(_remove_inplace, models[0], models[1])
