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
    log_file,
    diff_log_path,
    weight_struct_info,
    assert_tensor_equal,
)
from .layer_map import LayerMap
from .abstracts import create_proxy_model, ProxyModel
from .file_loader import global_yaml_loader as yamls
from .special_init import global_special_init_pool as init_pool
from .special_init import build_name


def process_each_weight(process, models, layer_map):
    def _process_runner(process, submodels, param_names, params):
        settings = yamls.get_weight_settings(submodels[0], submodels[1], param_names[0])
        process(submodels, param_names, params, settings)

    submodels_0 = layer_map.weight_init_layers(models[0])
    submodels_1 = layer_map.weight_init_layers(models[1])

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
                err_str += f"            {submodel_0.path_info + '.' + param_name_0}\n"
                err_str += f"    Model[1] {submodel_1.fullname}: {submodel_1.model_repr_info()}\n"
                err_str += f"            {submodel_1.path_info + '.' + param_name_1}\n"
                err_str += f"{type(e).__name__ + ':  ' + str(e)}\n"
                err_str += weight_struct_info(models, (submodel_0, submodel_1))
                raise RuntimeError(err_str)


# this interface is exposed, so it takes two models as inputs
def assign_weight(target_model, source_model, layer_map={}):
    """
    Init weights of layer(paddle) and module(torch) with same value

    Args:
        layer (paddle.nn.Layer): input paddle layer
        module (torch.nn.Module): input torch module
    """

    if not isinstance(target_model, ProxyModel):
        target_model = create_proxy_model(target_model)
    if not isinstance(source_model, ProxyModel):
        source_model = create_proxy_model(source_model)

    layer_map = LayerMap.create_from(layer_map)
    models = (target_model, source_model)

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

    def _assign_weight(submodels, param_names, params, settings):
        check_shape(models, param_names, params, settings)
        np_value = params[1].numpy()
        if settings["transpose"]:
            np_value = numpy.transpose(np_value)

        params[0].set_data(np_value)

    try:
        process_each_weight(_assign_weight, models, layer_map)
        log("Assign weight success !!!")
        return True
    except Exception as e:
        log("Assign weight Failed !!!\n")
        print(type(e).__name__ + ":  " + str(e))
        return False


def check_shape(models, param_names, params, settings):
    shape_0 = params[0].shape()
    shape_1 = params[1].shape()
    if settings["transpose"]:
        shape_1.reverse()
    assert (
        shape_0 == shape_1
    ), f"Shape of param `{param_names[0]}` in {models[0].name} and param `{param_names[1]}` in {models[1].name} is not the same. {shape_0} vs {shape_1}\n"


def check_weight(models, options, layer_map):
    _weight_check = True

    def _check_weight(submodels, param_names, params, settings):
        check_shape(models, param_names, params, settings)

        np_value_0 = params[0].numpy()
        np_value_1 = params[1].numpy()

        if settings["transpose"]:
            np_value_1 = numpy.transpose(np_value_1)

        # check weight
        try:
            assert_tensor_equal(np_value_0, np_value_1, settings)
        except Exception as e:
            nonlocal _weight_check
            _weight_check = False
            info = (
                "=" * 25 + "\n" + "After training, weight value is different.\n"
                "between Model[0] `{}`, Model[1] `{}` \n"
                "Model[0] param path:\n    {}\n"
                "Model[1] param path:\n    {}\n"
                "{}\n\n".format(
                    models[0].model_repr_info(),
                    models[1].model_repr_info(),
                    submodels[0].path_info + "." + param_names[0],
                    submodels[1].path_info + "." + param_names[1],
                    type(e).__name__ + ":  " + str(e),
                )
            )
            log_file("weight_diff.log", "a", info)

    try:
        process_each_weight(_check_weight, models, layer_map)
    except Exception as e:
        log("Err occurs when compare weight!!!\n")
        print(type(e).__name__ + ":  " + str(e))
        return False

    if _weight_check == False:
        log(f"Diff found in model weights after optimizer step, check report `{diff_log_path + '/weight_diff.log'}`.")
    else:
        log("weight compared.")

    return _weight_check


def check_grad(models, options, layer_map):
    _grad_check = True

    def _check_grad(submodels, param_names, params, settings):
        check_shape(models, param_names, params, settings)

        # grad() returns numpy value here
        grad_0 = params[0].grad()
        grad_1 = params[1].grad()

        # check grad
        try:
            if grad_0 is None and grad_1 is None:
                return
            elif grad_0 is None and grad_1 is not None:
                raise RuntimeError(
                    f"Found grad in {model[0].name} is `None`, when grad in {model[1].name} exists. Please check the grad value."
                )
            elif grad_0 is not None and grad_1 is None:
                raise RuntimeError(
                    f"Found grad in {model[1].name} is `None`, when grad in {model[0].name} exists. Please check the grad value."
                )

            if settings["transpose"]:
                grad_1 = numpy.transpose(grad_1)

            assert_tensor_equal(grad_0, grad_1, settings)
        except Exception as e:
            nonlocal _grad_check
            _grad_check = False
            info = (
                "=" * 25 + "\n" + "After training, grad value is different.\n"
                "between Model[0] `{}`, Model[1] `{}` \n"
                "Model[0] grad path:\n    {}\n"
                "Model[1] grad path:\n    {}\n"
                "{}\n\n".format(
                    models[0].model_repr_info(),
                    models[1].model_repr_info(),
                    submodels[0].path_info + "." + param_names[0],
                    submodels[1].path_info + "." + param_names[1],
                    type(e).__name__ + ":  " + str(e),
                )
            )
            log_file("grad_diff.log", "a", info)

    try:
        process_each_weight(_check_grad, models, layer_map)
    except Exception as e:
        log("Err occurs when compare grad!!!\n")
        print(type(e).__name__ + ":  " + str(e))
        return False

    if _grad_check == False:
        log(f"Diff found in model grad after backward, check report `{diff_log_path + '/grad_diff.log'}`.")
    else:
        log("grad compared.")

    return _grad_check
