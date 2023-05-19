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


import paddle
import torch
from .utils import log, init_options, init_path_info
from .abstracts import ProxyModel
from .layer_map import LayerMap
from .weights import assign_weight
from .trainer import Trainer


paddle.set_printoptions(precision=10)
torch.set_printoptions(precision=10)


def auto_diff(base_model, raw_model, inputs, loss_fns=None, optimizers=None, layer_map=None, **kwargs):
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        base_model: paddle.nn.Layer or torch.nn.Module, provides the baseline of data precision。
        raw_model: paddle.nn.Layer or torch.nn.Module, which need to compare with base_model.
        inputs: input data for models, it should be a list of dict.
        loss_fns (list, optional): list of loss function for models.
        optimizers (list, optional): list of optimizers for models.
        layer_map (class LayerMap, optional): manually map paddle layer and torch module.
        kwargs: other options, view `https://github.com/PaddlePaddle/PaDiff` to learn more infomations
    Returns:
        True for success, False for failed.
    """

    options = kwargs
    init_options(options)

    models = (base_model, raw_model)

    # ProxyModel.create_from will do assert check for models
    if "model_names" in options:
        assert len(options["model_names"]) == 2
        assert options["model_names"][0] != options["model_names"][1], "Can not use same name for two model."
        models = [ProxyModel.create_from(x, name) for x, name in zip(models, options["model_names"])]
    else:
        names = [base_model.__class__.__name__ + "(base_model)", raw_model.__class__.__name__ + "(raw_model)"]
        log(f"Model_names not found, use default names instead:")
        print(f"             `{names[0]}`")
        print(f"             `{names[1]}`")
        models = [ProxyModel.create_from(x, name) for x, name in zip(models, names)]

    assert isinstance(inputs, (tuple, list)), "Invalid Argument."

    for input in inputs:
        assert isinstance(input, dict), "Invalid Argument."

    if loss_fns is not None:
        options["use_loss"] = True
        assert len(loss_fns) == 2
        for loss in loss_fns:
            assert callable(loss), "Invalid loss function"

    if optimizers is not None:
        options["use_opt"] = True
        assert len(optimizers) == 2
        for opt in optimizers:
            assert isinstance(opt, (paddle.optimizer.Optimizer, torch.optim.Optimizer)) or callable(
                opt
            ), "Invalid optimizer"

    layer_map = LayerMap.create_from(layer_map)
    init_path_info(models)
    trainer = Trainer(models, loss_fns, optimizers, layer_map, options)
    if options["auto_init"] and not assign_weight(base_model, raw_model, layer_map):
        return False

    ret = trainer.train(inputs)

    if ret:
        log("SUCCESS !!!\n")
    else:
        log("FAILED !!!\n")

    return ret
