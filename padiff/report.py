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

from .actions import get_action
from .stack_info import print_frames
from .utils import (
    TableView,
    TreeView,
    clone_structure,
    for_each_grad_tensor,
    for_each_tensor,
    log,
    assert_tensor_equal,
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
        if self.type == "forward":
            # we only clone output in forward step.
            self.output = clone_structure(output)
        else:
            self.output = output
        self.net = net
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

        return [None for i in for_each_grad_tensor(self.input)]

    def set_input_grads(self, nth, value):
        assert nth < len(self.input_grads)
        self.input_grads[nth] = value

    def print_stacks(self):
        print_frames(self.frames)

    def stacks(self):
        return self.frames

    def compare_tensors(self):
        if self.type == "forward":
            return for_each_tensor(self.output)
        if self.type == "backward":
            return for_each_tensor(self.input_grads)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strings = []
        strings.append("ReportItem: \n    type={}".format(self.type))
        strings.append("    step_idx: {}".format(self.step))
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

    def find_item(self, p_report, net_id):
        tlist = list(filter(lambda x: x.type == "forward" and x.net_id == net_id, self.items))
        plist = list(filter(lambda x: x.type == "forward" and x.net_id == net_id, p_report.items))
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
            strings.append("    " + str(item.step) + ": [{}]".format(type(item.net)))
        return "\n".join(strings)


"""
    LayerStack
"""


class LayerStack(object):
    """
    this class is used to build module structure
    """

    def __init__(self, type_):
        super(LayerStack, self).__init__()
        self.type = type_
        self.stack = []

        self.root = None

    def _push(self, value):
        self.stack.append(value)

    def _pop(self):
        return self.stack.pop()

    def _top(self):
        return self.stack[-1]

    def _empty(self):
        return len(self.stack) == 0

    def push_layer(self, module):
        net = NetWrap(module)
        if not self._empty():
            net.father = self._top()
            self._top().sons.append(net)
        else:
            if self.root is None:
                self.root = net
            else:
                raise RuntimeError("found multy root layers!")
        self._push(net)

    def pop_layer(self, module):
        assert id(self._top().net) == id(module)
        self._pop()

    def push_api(self, api, fwd, bwd):
        # an api
        if hasattr(api, "__api__"):
            net = NetWrap(api)
            net.is_api = True
            net.leaf_num = 1
            if not self._empty():
                self._top().sons.append(net)
            net.set_report(fwd, bwd)
            for N in self.stack:
                N.leafs.append(net)

        # a layer in one2one dict
        elif id(api) == id(self._top().net):
            net = self._top()
            net.is_layer_mapped = True
            net.set_report(fwd, bwd)
            for N in self.stack[:-1]:
                N.leafs.append(net)

        else:
            raise RuntimeError("api hook err when `push_api` !")


class NetWrap(object):
    def __init__(self, net):
        self.net = net
        self.sons = []
        self.father = None

        # leafs under this net
        self.leafs = []

        self.is_api = False
        self.is_layer_mapped = False

        # if is_leaf, report should exist
        self.is_leaf = False
        self.fwd_report = None
        self.bwd_report = None

    def set_report(self, fwd, bwd):
        self.fwd_report = fwd
        self.bwd_report = bwd
        self.is_leaf = True

    def __str__(self):
        if self.is_api:
            return "(api) " + self.net.__name__
        elif self.is_layer_mapped:
            return "(net in map) " + self.net.__class__.__name__
        else:
            return "(net) " + self.net.__class__.__name__


"""
    report_guard
"""

global_torch_report = None
global_paddle_report = None
global_torch_counter = Counter()
global_paddle_counter = Counter()


@contextlib.contextmanager
def report_guard(torch_report, paddle_report):
    global global_torch_report, global_paddle_report
    old_t = global_torch_report
    old_p = global_paddle_report
    try:
        global_torch_report = torch_report
        global_paddle_report = paddle_report

        torch_report.counter = global_torch_counter
        paddle_report.counter = global_paddle_counter

        torch_report.counter.clear()
        paddle_report.counter.clear()

        yield

    finally:
        global_torch_report = old_t
        global_paddle_report = old_p
        torch_report.counter = None
        paddle_report.counter = None


def current_paddle_report():
    if global_paddle_report is None:
        return None
        raise RuntimeError(
            "Please call `current_paddle_report()` within contextmanager `report_guard(Report(), Report())`."
        )
    return global_paddle_report


def current_torch_report():
    if global_torch_report is None:
        return None
        raise RuntimeError(
            "Please call `current_torch_report()` within contextmanager `report_guard(Report(), Report())`."
        )
    return global_torch_report


"""
    report analys
"""


def print_info(paddle_item, torch_item, exc, step_idx, grad=False):
    log("FAILED !!!")
    if grad:
        log(
            "    Diff found in `Backward Stage` in step: {}, net_id is {} vs {}".format(
                step_idx, paddle_item.net_id, torch_item.net_id
            )
        )
    else:
        log(
            "    Diff found in `Forward  Stage` in step: {}, net_id is {} vs {}".format(
                step_idx, paddle_item.net_id, torch_item.net_id
            )
        )
    log("    Type of layer is  : {} vs {}".format(type(torch_item.net), type(paddle_item.net)))
    print(str(exc))

    print("\n\nPaddle Stacks:")
    print("=========================")
    paddle_item.print_stacks()
    print("Torch  Stacks:")
    print("=========================")
    torch_item.print_stacks()


def _check_forward_and_backward(torch_rep, paddle_rep, cfg):
    """
    TODO(@xiongkun):
    More abundant printing methods can be supported later，For example, interactive printing mode，Tree Printing mode，Currently, only list printing is supported.
    """
    torch_fwd_items = torch_rep.get_fwd_items()
    paddle_fwd_items = paddle_rep.get_fwd_items()

    # temp use
    torch_fwd_items = list(filter(lambda x: x.net_id != -1, torch_fwd_items))
    paddle_fwd_items = list(filter(lambda x: x.net_id != -1, paddle_fwd_items))

    torch_fwd_items = TableView(torch_fwd_items, lambda x: x.net_id)
    paddle_tree_view = TreeView(paddle_fwd_items)

    assert len(torch_fwd_items) == len(
        paddle_fwd_items
    ), "Difference length of torch_fwd_items and paddle_items, make sure the paddle layer and torch module have the same valid sublayer."

    backward_items = []
    # forward check
    for idx, paddle_item in enumerate(paddle_tree_view.traversal_forward()):
        assert paddle_item.net_id in torch_fwd_items, "Torch has no corresponding module for {}".format(
            type(paddle_item.net)
        )
        torch_item = torch_fwd_items[paddle_item.net_id]
        assert torch_item.type == paddle_item.type and paddle_item.type == "forward"
        act = get_action(torch_item.net, paddle_item.net)
        try:
            backward_items.append([torch_item.bwd_item, paddle_item.bwd_item])
            act(torch_item, paddle_item, cfg)
        except Exception as e:
            if cfg["single_step"]:
                log("Under single_step mode:")
            print_info(paddle_item, torch_item, e, idx, grad=False)
            return False

    log("forward {} steps compared.".format(len(paddle_fwd_items)))

    # loss check
    if cfg["loss_fn"]:
        try:
            assert_tensor_equal(paddle_rep.loss, torch_rep.loss, cfg)
            log("loss compared.")
        except Exception as e:
            log("*** Diff found in loss, Checkout your loss function! ***")
            log("loss compare:\n")
            print("{}".format(str(e)))
            return False

    if cfg["diff_phase"] == "forward":
        log("Diff_phase is `forward`. Backward compare skipped.")
        log("SUCCESS !!!")
        return True

    # backward check
    # backward_map map from id(paddle_backward_item) to torch_backward_item
    backward_map = TableView(backward_items, lambda x: id(x[1]))
    """
    TODO(xiongkun): the order is problematic because we consider the tree structure as a chain structure.
          so, always the root layer is calculated first. but we want the first layer with diff.

    """
    for idx, paddle_item in enumerate(paddle_tree_view.traversal_backward()):
        torch_item, paddle_item = backward_map[id(paddle_item.bwd_item)]
        assert torch_item.type == paddle_item.type and paddle_item.type == "backward"
        act = get_action(torch_item.net, paddle_item.net)
        try:
            act(torch_item, paddle_item, cfg)
        except Exception as e:
            print_info(paddle_item, torch_item, e, idx, grad=True)
            return False

    log("bacward {} steps compared.".format(len(backward_items)))

    # total status
    log("SUCCESS !!!")
    return True


def check_forward_and_backward(torch_rep, paddle_rep, cfg):
    breakpoint()
    tree_print(torch_rep.stack.root, [])
    tree_print(paddle_rep.stack.root, [])
    return True


# used to print module as a tree
def tree_print(root, prefix):
    for i, s in enumerate(prefix):
        if i == len(prefix) - 1:
            print(s, end="")
        else:
            if s == "|--- ":
                print("|    ", end="")
            elif s == "+--- ":
                print("     ", end="")

    print(str(root))

    for i, child in enumerate(root.sons):
        pre = "|--- "
        if i == len(root.sons) - 1:
            pre = "+--- "
        prefix.append(pre)
        tree_print(child, prefix)
        prefix.pop()


# del all wrap layers
def del_wrap_layers(root):
    def copy_node_attrs(root):
        retval = NetWrap(root.net)
        for attr in ("is_api", "is_layer_mapped", "is_leaf", "fwd_report", "bwd_report"):
            val = getattr(root, attr)
            setattr(retval, attr, val)
        return retval

    # base cases
    # if len(root.leafs) == 0: a leaf layer
    if len(root.leafs) == 0:
        retval = copy_node_attrs(root)
        return retval
    # if len(root.leafs) == 1: a wrap layer
    if len(root.leafs) == 1:
        retval = copy_node_attrs(root.leafs[0])
        return retval

    # about ret:
    # 1. ret is a leaf layer
    # 2. ret is a normal layer who called multi apis

    # about retval to return:
    # 1. root is a wrapper, so, return retval directly
    #    this case is imposible: should return in base case
    # 2. root is not a wrapper, needs:
    #    create new node and reset sons, leafs
    #    reset fathers for nodes returned

    retval = copy_node_attrs(root)

    for child in root.sons:
        ret = del_wrap_layers(child)
        ret.father = retval
        retval.sons.append(ret)
        retval.leafs.extend(ret.leafs)

    return retval
