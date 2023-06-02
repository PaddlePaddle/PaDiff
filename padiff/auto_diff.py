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
        options["model_names"] = names

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

    init_options(options)
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


#=================================================================================



from .Runner import Runner
from .OptimizerHelper import OptimizerHelper
from .Checker import Checker

from .trainer_utils import Report
from ..utils import log


class Trainer:
    def __init__(self, models, loss_fn, opt, layer_map, options):
        self.models = models
        self.model_types = [x.model_type for x in models]
        self.runner = Runner(models, loss_fn, layer_map, options)
        self.optimizer_helper = OptimizerHelper(opt, options)
        self.options = options
        self.steps = options["steps"]
        self.layer_map = layer_map

    def do_run(self, reports, inputs):
        self.runner.set_report(reports)
        self.runner.run_step(inputs)
        setattr(reports[0].stack.root, "model_name", self.models[0].name)
        setattr(reports[1].stack.root, "model_name", self.models[1].name)

    def do_check_fwd_bwd(self, reports):
        ret = Checker.check_forward_and_backward(reports, self.options)
        return ret

    def do_check_grad(self):
        ret = Checker.check_grad(self.models, options=self.options, layer_map=self.layer_map)
        return ret

    def do_check_weight(self):
        ret = Checker.check_weight(self.models, options=self.options, layer_map=self.layer_map)
        return ret

    def do_optimizer(self):
        self.optimizer_helper.step()

    def train(self, inputs):
        if self.options["single_step"]:
            return self.run_single_step(inputs)
        else:
            return self.run_normal(inputs)

    # run pipeline should be a part of auto_diff
    # not used in new design
    def run_normal(self, inputs):
        for step_id in range(self.options["steps"]):
            log(f"=================Train Step {step_id}=================")
            reports = [Report(self.model_types[x]) for x in range(2)]
            self.do_run(reports, inputs)

            ret = self.do_check_fwd_bwd(reports)
            if ret == False:
                return False

            if self.options["diff_phase"] == "forward":
                log("Diff phase is forward, weight and grad check skipped.")
            else:
                ret = self.do_check_grad()
                if ret == False:
                    return False

                self.do_optimizer()
                ret = self.do_check_weight()
                if ret == False:
                    return False
        return True

    def run_single_step(self, inputs):
        diff_phase = self.options["diff_phase"]
        for step_id in range(self.options["steps"]):
            log(f"=================Train Step {step_id}=================")

            if diff_phase == "forward" or diff_phase == "both":
                log(f"diff phase is {diff_phase}, run single_step forward part.")
                self.options["diff_phase"] = "forward"

                reports = [Report(self.model_types[x]) for x in range(2)]
                self.do_run(reports, inputs)

                ret = self.do_check_fwd_bwd(reports)
                if ret == False:
                    log("Diff found at sinle_step mode `forward` part!")
                    return False

            if diff_phase == "backward" or diff_phase == "both":
                log(f"diff phase is {diff_phase}, run single_step backward part.")
                self.options["diff_phase"] = "backward"

                reports = [Report(self.model_types[x]) for x in range(2)]
                self.do_run(reports, inputs)

                ret = self.do_check_fwd_bwd(reports)
                if ret == False:
                    log("Diff found at sinle_step mode `backward` part!")
                    return False

                ret = self.do_check_grad()
                if ret == False:
                    log("Diff found at sinle_step mode `backward` part!")
                    return False

                self.do_optimizer()
                ret = self.do_check_weight()
                if ret == False:
                    log("Diff found at sinle_step mode `backward` part!")
                    return False

        self.options["diff_phase"] = diff_phase
        return True
