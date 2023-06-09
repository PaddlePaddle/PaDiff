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
from .diff_utils import init_options, OptimizerHelper
from ..utils import log
from ..abstracts import ProxyModel, create_model
from ..weight_init import assign_weight



paddle.set_printoptions(precision=10)
torch.set_printoptions(precision=10)


def auto_diff(base_model, raw_model, inputs, loss_fns=None, optimizers=None, **kwargs):
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

    if not isinstance(base_model, ProxyModel) or not isinstance(raw_model, ProxyModel):
        names = [base_model.__class__.__name__ + "(base_model)", raw_model.__class__.__name__ + "(raw_model)"]
        models = [create_model(x, name) for x, name in zip(models, names)]

    assert isinstance(inputs, (tuple, list)), "Invalid Argument."

    for input_ in inputs:
        assert isinstance(input_, dict), "Invalid Argument."

    if loss_fns is not None:
        options["use_loss"] = True
        assert len(loss_fns) == 2
        for loss in loss_fns:
            assert callable(loss), "Invalid loss function"
    else:
        loss_fns = [None, None]

    if optimizers is not None:
        options["use_opt"] = True
        assert len(optimizers) == 2
        for opt in optimizers:
            assert isinstance(opt, (paddle.optimizer.Optimizer, torch.optim.Optimizer)) or callable(
                opt
            ), "Invalid optimizer"
        optimizers = [OptimizerHelper(opt) for opt in optimizers]
    else:
        optimizers = [None, None]

    init_options(options)
    cfg = {}
    for key in ("atol", "rtol", "compare_mode"):
        cfg[key] = options[key]
        del options[key]

    if options["auto_init"] and not assign_weight(base_model, raw_model):
        return False

    ##########
    # TODO run model -> dump -> compare
    ##########

    run_pipeline((base_model, raw_model), inputs, loss_fns, optimizers, options, cfg)



    return
