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
from .utils import log, init_options, init_padiff_path
from .padiff_abstracts import padiff_model
from .layer_map import init_LayerMap
from .weights import assign_weight
from .trainer import Trainer


paddle.set_printoptions(precision=10)
torch.set_printoptions(precision=10)


def auto_diff(
    model_0, model_1, example_inp, auto_weights=True, options={}, layer_map=None, loss_fn=None, optimizer=None, steps=1
):
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        TODO update annotations
        example_inp (paddle_input, torch_input): input data for paddle layer and torch module.
            paddle_input and torch_input should be dict and send into model like `module(**input)`.
        auto_weights (boolean, optional): uniformly init the parameters of models
        options (dict, optional):
            atol, compare_mode
        layer_map (class LayerMap, optional): manually map paddle layer and torch module.
    Returns:
        True for success, False for failed.
    """
    assert isinstance(model_0, (paddle.nn.Layer, torch.nn.Module))
    assert isinstance(model_1, (paddle.nn.Layer, torch.nn.Module))

    models = (model_0, model_1)
    if "model_names" in options:
        assert len(options["model_names"]) == 2
        assert options["model_names"][0] != options["model_names"][1], "Can not use same name for two model."
        models = [padiff_model(x, name) for x, name in zip(models, options["model_names"])]
    else:
        if model_0.__class__.__name__ != model_1.__class__.__name__:
            names = [x.__class__.__name__ for x in models]
        else:
            names = [x.__class__.__name__ + f"_{idx}" for x, idx in enumerate(models)]
        log(f"*** model_names not provided, use `{names[0]}` and `{names[1]}` as default ***")
        models = [padiff_model(x, name) for x, name in zip(models, names)]

    assert isinstance(example_inp, (tuple, list)), "Invalid Argument."

    for inputs in example_inp:
        assert isinstance(inputs, dict), "Invalid Argument."

    if loss_fn is not None:
        options["use_loss"] = True
        assert len(loss_fn) == 2
        for loss in loss_fn:
            assert callable(loss), "Invalid loss function"

    if optimizer is not None:
        options["use_opt"] = True
        assert len(optimizer) == 2
        for opt in optimizer:
            assert isinstance(opt, (paddle.optimizer.Optimizer, torch.optim.Optimizer)) or callable(
                opt
            ), "Invalid optimizer"

    # prepare models and options
    options["steps"] = steps
    init_options(options)
    layer_map = init_LayerMap(layer_map)
    init_padiff_path(models)
    trainer = Trainer(models, loss_fn, optimizer, layer_map, options)
    if auto_weights and not assign_weight(models[0], models[1], layer_map):
        return False

    ret = trainer.train(example_inp)

    if ret:
        log("SUCCESS !!!\n")
    else:
        log("FAILED !!!\n")

    return ret
