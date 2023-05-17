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

    def do_run(self, reports, example_inp):
        self.runner.set_report(reports)
        self.runner.forward_step(example_inp)
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

    def train(self, example_inp):
        if self.options["single_step"]:
            return self.run_single_step(example_inp)
        else:
            return self.run_normal(example_inp)

    def run_normal(self, example_inp):
        for step_id in range(self.options["steps"]):
            log(f"=================Train Step {step_id}=================")
            reports = [Report(self.model_types[x]) for x in range(2)]
            self.do_run(reports, example_inp)

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

    def run_single_step(self, example_inp):
        diff_phase = self.options["diff_phase"]
        for step_id in range(self.options["steps"]):
            log(f"=================Train Step {step_id}=================")

            if diff_phase == "forward" or diff_phase == "both":
                log(f"diff phase is {diff_phase}, run single_step forward part.")
                self.options["diff_phase"] = "forward"

                reports = [Report(self.model_types[x]) for x in range(2)]
                self.do_run(reports, example_inp)

                ret = self.do_check_fwd_bwd(reports)
                if ret == False:
                    log("Diff found at sinle_step mode `forward` part!")
                    return False

            if diff_phase == "backward" or diff_phase == "both":
                log(f"diff phase is {diff_phase}, run single_step backward part.")
                self.options["diff_phase"] = "backward"

                reports = [Report(self.model_types[x]) for x in range(2)]
                self.do_run(reports, example_inp)

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
