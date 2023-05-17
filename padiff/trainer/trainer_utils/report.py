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

import contextlib

from .. import utils
from .module_struct import (
    LayerStack,
)

"""
    Report definition
"""


class Counter:
    def __init__(self):
        self.clear()

    def clear(self):
        self.id = 0

    def get_id(self):
        ret = self.id
        self.id += 1
        return ret


class ReportItem:
    def __init__(self, type, step, input, output, net, net_id, frame_info, frames):
        assert type in [
            "forward",
            "backward",
        ], "type can only be one of ['forward', 'backward']"
        self.type = type
        self.step = step
        """
        self.input is a tuple: (tensor, ...)
        """
        # self.input = clone_tensors(input)
        self.input = input
        self.output = output

        self.net = net
        self.net_str = net.__name__ if hasattr(net, "__api__") else net.__class__.__name__
        self.net_id = net_id
        self.fwd_item = None
        self.bwd_item = None
        self.frame_info = frame_info
        self.frames = frames
        self.input_grads = self._gen_input_grads()

    def set_forward(self, fwd):
        assert self.type == "backward", "can't set forward for non-backward item."
        fwd.bwd_item = self
        self.fwd_item = fwd

    def _gen_input_grads(self):
        if self.type == "forward":
            return None
        assert self.input is not None, "Backward while input is None, not expected."

        return [None for i in utils.for_each_grad_tensor(self.input)]

    def set_input_grads(self, nth, value):
        assert nth < len(self.input_grads)
        self.input_grads[nth] = value

    def print_stacks(self):
        def print_frames(fs, indent=8):
            indent = " " * indent
            for f in fs:
                print(
                    "{} File {}: {}    {}\n{}{}{}".format(indent, f.filename, f.lineno, f.name, indent, indent, f.line)
                )

        print_frames(self.frames)

    def stacks(self):
        return self.frames

    def tensors_for_compare(self):
        if self.type == "forward":
            return utils.for_each_tensor(self.output)
        if self.type == "backward":
            return utils.for_each_tensor(self.input_grads)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strings = []
        strings.append("ReportItem: \n    type={}".format(self.type))
        strings.append("    step_idx: {}".format(self.step))
        strings.append("    net: {}\n".format(self.net_str))
        return "\n".join(strings)


class Report:
    def __init__(self, name):
        self.name = name
        self.items = []
        self.counter = None
        self.loss = None
        self.stack = LayerStack(name)

        # self.layer_map is used to confirm whether an API report is needed
        # if api belongs to an layer which is ignored, we do not need it's report
        # layer_map is set in Trainer.set_report
        self.layer_map = None

    def put_item(self, type, input, output, net, net_id, frame_info, frames):
        step = self.counter.get_id()
        self.items.append(
            ReportItem(
                type=type,
                step=step,
                input=input,
                output=output,
                net=net,
                net_id=net_id,
                frame_info=frame_info,
                frames=frames,
            )
        )
        return self.items[-1]

    def get_fwd_items(self):
        sorted(self.items, key=lambda x: x.step)
        return list(filter(lambda x: x.type == "forward", self.items))

    def find_item(self, p_report, net_id, type_):
        tlist = list(filter(lambda x: x.type == type_ and x.net_id == net_id, self.items))
        plist = list(filter(lambda x: x.type == type_ and x.net_id == net_id, p_report.items))
        return tlist[len(plist) - 1]

    def set_loss(self, loss):
        self.loss = loss.detach().cpu().clone()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        sorted(self.items, key=lambda x: x.step)
        strings = []
        strings.append("Report name is: " + self.name)
        for item in self.items:
            strings.append("    " + str(item.step) + ": [{}]".format(item.net_str))
        return "\n".join(strings)


"""
    report_guard
"""

global_reports = [None, None]
global_torch_counter = Counter()
global_paddle_counter = Counter()


@contextlib.contextmanager
def report_guard(reports):
    global global_reports
    old_reports = global_reports
    try:
        global_reports = reports

        reports[0].counter = global_torch_counter
        reports[1].counter = global_paddle_counter

        reports[0].counter.clear()
        reports[1].counter.clear()

        yield

    finally:
        global_reports = old_reports
        reports[0].counter = None
        reports[1].counter = None


def current_reports():
    global global_reports
    if global_reports is None:
        return None

    return global_reports
