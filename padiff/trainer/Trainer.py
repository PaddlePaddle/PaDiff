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
    def __init__(self, layer, module, loss_fn, opt, layer_map, options):
        self.runner = Runner(layer, module, loss_fn, layer_map, options)
        self.optimizer_helper = OptimizerHelper(opt, options)
        self.options = options
        self.steps = options["steps"]
        self.layer_map = layer_map

    def do_run(self, paddle_report, torch_report, example_inp):
        self.runner.set_report(paddle_report, torch_report)
        self.runner.forward_step(example_inp)

    def do_check_fwd_bwd(self, paddle_report, torch_report):
        ret = Checker.check_forward_and_backward(torch_report, paddle_report, self.options)
        return ret

    def do_check_grad(self):
        ret = Checker.check_grad(self.runner.layer, self.runner.module, options=self.options, layer_map=self.layer_map)
        return ret

    def do_check_weight(self):
        ret = Checker.check_weight(
            self.runner.layer, self.runner.module, options=self.options, layer_map=self.layer_map
        )
        return ret

    def do_optimizer(self):
        self.optimizer_helper.step()

    def train(self, example_inp):
        if self.options["single_step"]:
            return self.run_single_step(example_inp)
        else:
            return self.run_normal(example_inp)

    def run_normal(self, example_inp):
        ret = True
        for step_id in range(self.options["steps"]):
            log(f"=================Train Step {step_id}=================")
            paddle_report = Report("paddle")
            torch_report = Report("torch")
            self.do_run(paddle_report, torch_report, example_inp)
            ret = self.do_check_fwd_bwd(paddle_report, torch_report) and ret
            if ret == False:
                return False

            if self.options["diff_phase"] == "forward":
                log("Diff phase is forward, weight and grad check skipped.")
            else:
                ret = self.do_check_grad() and ret
                self.do_optimizer()
                ret = self.do_check_weight() and ret

                if ret == False:
                    return False
        return True

    def run_single_step(self, example_inp):
        ret = True
        diff_phase = self.options["diff_phase"]
        for step_id in range(self.options["steps"]):
            log(f"=================Train Step {step_id}=================")
            if diff_phase == "forward" or diff_phase == "both":
                log(f"diff phase is {diff_phase}, run single_step forward part.")
                self.options["diff_phase"] = "forward"
                paddle_report = Report("paddle")
                torch_report = Report("torch")
                self.do_run(paddle_report, torch_report, example_inp)
                ret = self.do_check_fwd_bwd(paddle_report, torch_report) and ret
                if ret == False:
                    log("Diff found at sinle_step mode `forward` part!")
                    return False

            if diff_phase == "backward" or diff_phase == "both":
                log(f"diff phase is {diff_phase}, run single_step backward part.")
                self.options["diff_phase"] = "backward"

                paddle_report = Report("paddle")
                torch_report = Report("torch")
                self.do_run(paddle_report, torch_report, example_inp)

                ret = self.do_check_fwd_bwd(paddle_report, torch_report) and ret
                if ret == False:
                    log("Diff found at sinle_step mode `backward` part!")
                    return False

                ret = self.do_check_grad() and ret
                self.do_optimizer()
                ret = self.do_check_weight() and ret

                if ret == False:
                    log("Diff found at sinle_step mode `backward` part!")
                    return False

        self.options["diff_phase"] = diff_phase
        return True
