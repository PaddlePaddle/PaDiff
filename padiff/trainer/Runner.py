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

from .trainer_utils import register_hooker, report_guard
from ..utils import (
    log,
    max_diff,
    tensors_mean,
    remove_inplace,
)

import os


class Runner(object):
    def __init__(self, models, loss_fn, layer_map, options):
        self.models = models
        self.options = options
        self.loss_fn = loss_fn
        self.layer_map = layer_map
        self.reports = [None, None]

        remove_inplace(models)

        if os.getenv("PADIFF_CUDA_MEMORY") != "OFF":
            self.devices = [model.get_device() for model in self.models]
            for model in self.models:
                model.to_cpu()

    def set_report(self, reports):
        reports[0].layer_map = self.layer_map
        reports[1].layer_map = self.layer_map
        self.reports = reports

    def run_step(self, inputs):
        def run_model(model_idx):
            with register_hooker(self, model_idx):
                try:
                    output = self.models[model_idx](**(inputs[model_idx]))
                    if self.options["use_loss"]:
                        loss = self.loss_fn[model_idx](output)
                        self.reports[model_idx].set_loss(loss)
                    else:
                        loss = tensors_mean(output, self.models[model_idx].model_type)
                    if self.options["diff_phase"] == "both" or self.options["diff_phase"] == "backward":
                        loss.backward()
                    return output
                except Exception as e:
                    raise RuntimeError(
                        "Exception is thrown while running forward of {}, please check the legality of module.\n{}".format(
                            self.models[model_idx].name, type(e).__name__ + ":  " + str(e)
                        )
                    )

        with report_guard(self.reports):
            base_output = run_model(model_idx=0)
            raw_output = run_model(model_idx=1)

        if not self.options["single_step"]:
            log("Max elementwise output diff is {}".format(max_diff(base_output, raw_output)))
