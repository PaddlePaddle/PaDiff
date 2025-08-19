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

import os

from ..utils import Counter, for_each_grad_tensor, for_each_tensor


class LayerStack:
    def __init__(self):
        super().__init__()
        self.stack = []
        self.root = []

    def _push(self, value):
        self.stack.append(value)

    def _pop(self):
        return self.stack.pop()

    def _top(self):
        if len(self.stack) == 0:
            return None
        return self.stack[-1]

    def _empty(self):
        return len(self.stack) == 0

    def push_layer(self, module):
        net = NetWrapper(module)
        if not self._empty():
            net.father = self._top()
            self._top().children.append(net)
        else:
            self.root.append(net)
        self._push(net)

    def pop_layer(self, module):
        assert id(self._top().net) == id(module)
        return self._pop()

    def push_api(self, api, fwd, bwd):
        if hasattr(api, "__api__"):
            net = NetWrapper(api)
            net.layer_type = "api"
            if not self._empty():
                self._top().children.append(net)
                net.father = self._top()
            net.set_report(fwd, bwd)
        else:
            net = self._top()
            net.set_report(fwd, bwd)


class NetWrapper:
    def __init__(self, net):
        self.net = net
        self.net_str = net.__name__ if hasattr(net, "__api__") else net.__class__.__name__
        self.children = []
        self.father = None

        self.layer_type = "net"  # "api" | "in map" | "net"

        self.fwd_report = None
        self.bwd_report = None

    def set_report(self, fwd, bwd):
        self.fwd_report = fwd
        self.bwd_report = bwd

    def pprint(self):
        def _tree_print(root, mark=None, prefix=[]):
            cur_str = ""
            for i, s in enumerate(prefix):
                if i == len(prefix) - 1:
                    cur_str += s
                else:
                    if s == " |--- ":
                        cur_str += " |    "
                    elif s == " +--- ":
                        cur_str += "      "
                    else:
                        cur_str += s

            cur_str += str(root)
            if os.getenv("PADIFF_PATH_LOG") == "ON" and hasattr(root.net, "route"):
                cur_str += "  (" + root.net.route + ")"
            if mark is root:
                cur_str += "    <---  *** HERE ***"

            ret_strs = [cur_str]
            for i, child in enumerate(root.children):
                pre = " |--- "
                if i == len(root.children) - 1:
                    pre = " +--- "
                prefix.append(pre)
                retval = _tree_print(child, mark, prefix)
                ret_strs.extend(retval)
                prefix.pop()

            return ret_strs

        ret = _tree_print(self)
        print("\n".join(ret))

    def __str__(self):
        return f"({self.layer_type}){self.net_str}"


class Report:
    def __init__(self, marker):
        self.items = []
        self.counter = Counter()
        self.marker = marker
        self.stack = LayerStack()

    def put_item(self, type_, input_, output, net, net_id, frames):
        step = self.counter.get_id()
        self.items.append(
            ReportItem(
                type_=type_,
                step=step,  # report order of layers/apis
                input_=input_,
                output=output,
                net=net,
                net_id=net_id,  # traversal order of sublayers
                frames=frames,
            )
        )
        return self.items[-1]

    def __str__(self):
        sorted(self.items, key=lambda x: x.step)
        strings = []
        strings.append("Report:")
        for item in self.items:
            strings.append("    " + str(item.step) + f": [{item.net_str}]")
        return "\n".join(strings)


class ReportItem:
    def __init__(self, type_, step, input_, output, net, net_id, frames):
        assert type_ in [
            "forward",
            "backward",
        ], f"type can only be one of ['forward', 'backward'], but{type_}"
        self.type = type_  # fwd or bwd
        self.step = step  # report order
        self.input = input_  # layer input (same for fwd or bwd)
        self.output = output  # layer output (same for fwd or bwd)

        self.net = net  # the layer ,if it is an api, this should be a str layer which is generated in hooks
        self.net_str = net.__name__ if hasattr(net, "__api__") else net.__class__.__name__
        self.net_id = net_id  # sublayer order
        self.fwd_item = None  # bound to another reportitem, if self.type is "backward"
        self.bwd_item = None  # bound to another reportitem, if self.type is "forward"
        self.input_grads = self._gen_input_grads()
        self.frames = frames

    def set_forward(self, fwd):
        assert self.type == "backward", "can't set forward for non-backward item."
        fwd.bwd_item = self
        self.fwd_item = fwd

    def _gen_input_grads(self):
        if self.type == "forward":
            return None
        assert self.input is not None, "Backward while input is None, not expected."

        return [None for i in for_each_grad_tensor(self.input)]

    def set_input_grads(self, nth, value):
        assert nth < len(self.input_grads)
        self.input_grads[nth] = value

    def tensors_for_compare(self):
        if self.type == "forward":
            return [t for (t,) in for_each_tensor(self.output)]
        if self.type == "backward":
            return [t for (t,) in for_each_grad_tensor(self.input_grads)]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strings = []
        strings.append(f"ReportItem: \n    type={self.type}")
        strings.append(f"    step_idx: {self.step}")
        strings.append(f"    net: {self.net_str}\n")
        return "\n".join(strings)
