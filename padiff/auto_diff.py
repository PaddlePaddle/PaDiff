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
from .report import Report, check_forward_and_backward
from .utils import (
    log,
    reset_log_dir,
    init_options,
    modify_layer_mapping,
)
from .weights import check_weight_grad
from .yaml_loader import global_yaml_loader as yamls
from .cmd import PaDiff_Cmd
from .Trainer import Trainer


paddle.set_printoptions(precision=10)
torch.set_printoptions(precision=10)


def auto_diff(
    layer, module, example_inp, auto_weights=True, options={}, layer_mapping={}, loss_fn=None, optimizer=None, steps=1
):
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        layer (paddle.nn.Layer): paddle layer that needs compare
        module (torch.nn.Module): torch module that needs compare
        example_inp (paddle_input, torch_input): input data for paddle layer and torch module.
            paddle_input and torch_input should be dict and send into net like `module(**input)`.
        auto_weights (boolean, optional): uniformly init the parameters of models
        options (dict, optional):
            atol, compare_mode
        layer_mapping (dict, optional): manually map paddle layer and torch module.
    Returns:
        True for success, False for failed.
    """

    # checkout inputs
    assert isinstance(layer, paddle.nn.Layer), "Invalid Argument."
    assert isinstance(module, torch.nn.Module), "Invalid Argument."
    assert isinstance(example_inp, (tuple, list)), "Invalid Argument."
    log("Start auto_diff, may need a while to generate reports...")

    paddle_input, torch_input = example_inp
    assert isinstance(paddle_input, dict), "Invalid Argument."
    assert isinstance(torch_input, dict), "Invalid Argument."
    paddle.set_device("cpu")
    module = module.to("cpu")

    if loss_fn is not None:
        paddle_loss, torch_loss = loss_fn
        assert callable(paddle_loss), "Invalid loss function"
        assert callable(torch_loss), "Invalid loss function"
        options["loss_fn"] = True

    if optimizer is not None:
        paddle_opt, torch_opt = optimizer
        assert isinstance(paddle_opt, paddle.optimizer.Optimizer), "Invalid Optimizer."
        assert isinstance(torch_opt, torch.optim.Optimizer), "Invalid Optimizer."
        options["opt"] = True

    # prepare models and options
    trainer = Trainer(layer, module, loss_fn, optimizer)
    _preprocess(trainer, auto_weights, options, layer_mapping)

    if steps > 1:
        if options["diff_phase"] == "forward" or options["opt"] == False:
            steps = 1
            log("Notice: diff_phase is `forward` or require optimizer, steps is set to `1`.")

    # collect reports and analys
    for step_id in range(steps):
        log(f"=================Train Step {step_id}=================")
        paddle_report = Report("paddle")
        torch_report = Report("torch")
        trainer.set_report(paddle_report, torch_report)

        trainer.clear_grad()
        trainer.train_step(example_inp, options, layer_mapping)

        weight_check, grad_check = check_weight_grad(
            trainer.layer, trainer.module, layer_mapping=layer_mapping, options=options
        )
        ret = check_forward_and_backward(torch_report, paddle_report, options)
        ret = ret and weight_check and grad_check

        if ret == False:
            log(f"\nDiff found in step {step_id}")
            break

    if options["cmd"]:
        PaDiff_Cmd(paddle_report, torch_report, options).cmdloop()

    # TODO(linjieccc): pytest failed if log clean is enabled
    # clean_log_dir()
    return ret


def _preprocess(trainer, auto_weights, options, layer_mapping):
    reset_log_dir()
    init_options(options)
    modify_layer_mapping(layer_mapping)
    yamls.options = options
    if auto_weights:
        trainer.assign_weight_(options=options, layer_mapping=layer_mapping)
