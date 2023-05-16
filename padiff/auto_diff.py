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
from .utils import log, init_options, init_LayerMap, init_padiff_path, padiff_model
from .weights import assign_weight
from .trainer import Trainer


paddle.set_printoptions(precision=10)
torch.set_printoptions(precision=10)


def auto_diff(
    models, names, example_inp, auto_weights=True, options={}, layer_map=None, loss_fn=None, optimizer=None, steps=1
):
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        models : list of PadiffModel
        example_inp (paddle_input, torch_input): input data for paddle layer and torch module.
            paddle_input and torch_input should be dict and send into model like `module(**input)`.
        auto_weights (boolean, optional): uniformly init the parameters of models
        options (dict, optional):
            atol, compare_mode
        layer_map (class LayerMap, optional): manually map paddle layer and torch module.
    Returns:
        True for success, False for failed.
    """

    # checkout inputs
    assert len(models) == 2, "Need input 2 models."

    if names is not None:
        assert len(models) == len(names)
        models = [padiff_model(x, name) for x, name in zip(models, names)]
    else:
        raise RuntimeError()

    assert isinstance(example_inp, (tuple, list)), "Invalid Argument."

    for inputs in example_inp:
        assert isinstance(inputs, dict), "Invalid Argument."

    if loss_fn is not None:
        options["use_loss"] = True
        assert len(loss_fn) == len(models)

        for loss in loss_fn:
            assert callable(loss), "Invalid loss function"

    if optimizer is not None:
        options["use_opt"] = True
        assert len(optimizer) == len(models)

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

    # TODO(linjieccc): pytest failed if log clean is enabled
    # clean_log_dir()
    return ret
