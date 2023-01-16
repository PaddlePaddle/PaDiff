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

from cmd import Cmd
from .utils import TableView, TreeView, assert_tensor_equal, for_each_tensor
import os

HELP_USAGE = 0
HELP_INFO = 1


class PaDiff_Cmd(Cmd):
    prompt = "(PaDiff) "

    def __init__(self, paddle_report, torch_report, options):
        Cmd.__init__(self)

        # "usage: ``, info: ",
        self.helps = {
            "compare": [
                "[usage]: `compare forward|backward|{int} input|output|grad`",
                "[info]: compare ReportItems. If compare backward, the second param is not nessesary; if compare {int}, that means you want to compare the {int}th ReportItem in forward process (with forward order)",
            ],
            "options": [
                "[usage]: `options` or `options set {name} {value}`",
                "[info]: print user options, or set options",
            ],
            "info": ["[usage]: `info`", "[info]: print model info"],
            "eval": ["[usage]: `eval {python command}`", "[info]: eval python command"],
            "q": ["[usage]: `q`", "[info]: quit cmd"],
            "clear": ["[usage]: `clear`", "[info]: clear screen"],
            "item": ["[usage]: `item forward|backward {int} {stack|net|net_id|step_id}`", "[info]: show item info"],
        }

        self.paddle_report = paddle_report
        self.torch_report = torch_report
        self.options = options

        self.p2t_dict = {}
        self.fwd_order = []
        self.bwd_order = []

        torch_fwd_items = torch_report.get_fwd_items()
        paddle_fwd_items = paddle_report.get_fwd_items()

        torch_table_view = TableView(torch_fwd_items, lambda x: x.net_id)
        paddle_tree_view = TreeView(paddle_fwd_items)

        for idx, paddle_item in enumerate(paddle_tree_view.traversal_forward()):
            self.fwd_order.append(paddle_item)
            torch_item = torch_table_view[paddle_item.net_id]
            assert torch_item.type == paddle_item.type and paddle_item.type == "forward"
            self.p2t_dict[paddle_item] = torch_item

        if options["diff_phase"] == "forward":
            return

        for idx, paddle_item in enumerate(paddle_tree_view.traversal_backward()):
            self.bwd_order.append(paddle_item.bwd_item)
            torch_item = self.p2t_dict[paddle_item.fwd_item].bwd_item
            assert torch_item.type == paddle_item.type and paddle_item.type == "backward"
            self.p2t_dict[paddle_item.bwd_item] = torch_item

    # core function
    def do_compare(self, line):
        words = line.split(" ")
        if len(words) < 2:
            print("Too few params.")
            return
        res = True
        if words[0] == "forward":
            res = self._compare(self.fwd_order, words[1])
        elif words[0] == "backward":
            res = self._compare(self.bwd_order, words[1])
        elif self._is_int(words[0]):
            res = self._compare([self.fwd_order[int(words[0])]], words[1])
        else:
            print("{} can not be recognized.".format(words[0]))
            self.do_help("compare")

        if res:
            print("Compare complete, no diff found.")

    # info & logs
    def do_item(self, line):
        words = line.split(" ")
        if len(words) < 3 or words[0] not in ["forward", "backward"] or not self._is_int(words[1]):
            print("Param err!")
            self.do_help("item")
            return

        if words[0] == "forward":
            item = self.fwd_order[int(words[1])]
        else:
            item = self.bwd_order[int(words[1])]

        if words[2] == "stack":
            item.print_stacks()
        elif words[2] == "net":
            print(str(item.net))
        elif words[2] == "net_id":
            print(item.net_id)
        elif words[2] == "step_id":
            print(item.step_id)
        else:
            print("Not support {}".format(words[2]))

    def do_options(self, line):
        words = line.split(" ")
        if len(words) > 2 and words[0] == "set":
            if words[1] in ["compare_mode", "diff_phase"]:
                self.options[words[1]] = words[2]
            elif words[1] in ["atol", "rtol"]:
                self.options[words[1]] = float(words[2])
            elif words[1] in ["single_step"]:
                self.options[words[1]] = bool(words[2])
            else:
                print("Can not set {}".format(words[2]))
        else:
            print("{")
            for key in self.options:
                print("  {}: {}".format(key, self.options[key]))
            print("}")

    def do_info(self, line):
        print("diff_phase: {}".format(self.options["diff_phase"]))
        print("forward steps: {}".format(len(self.fwd_order)))
        if self.options["diff_phase"] == "both":
            print("backward steps: {}".format(len(self.bwd_order)))
        # TODO: show model in graph?

    def do_help(self, line):
        words = line.split(" ")
        if len(line) == 0:
            print("Available commands:")
            for key in self.helps.keys():
                print("  {}".format(key))
        else:
            for w in words:
                if w in self.helps.keys():
                    print("{}\n{}\n{}".format(w, self.helps[w][HELP_USAGE], self.helps[w][HELP_INFO]))
                else:
                    print("Command {} not exist.")

    # for debug
    def do_eval(self, line):
        try:
            eval(line)
        except:
            print("execution Failed")

    # basic
    def do_q(self, line):
        return True

    def do_clear(self, line):
        os.system("clear")

    def preloop(self):
        print("\nEnter cmd ...\n")

    def emptyline(self):
        pass

    def default(self, line):
        print("Command not found.")

    # utils
    def _compare(self, items, mode):
        for idx, p_item in enumerate(items):
            t_item = self.p2t_dict[p_item]
            pts = self._get_tensors(p_item, mode)
            tts = self._get_tensors(t_item, mode)
            for (tt,), (pt,) in zip(tts, pts):
                res = self._compare_and_show_message(tt.detach().numpy(), pt.numpy())
                if res == False:
                    print("At idx:`{}` in mode `{}`, diff found, compare stopped.\n".format(idx, mode))
                    return False
        return True

    def _get_tensors(self, RepItem, mode):
        if mode == "input":
            return for_each_tensor(RepItem.input)
        elif mode == "output":
            return for_each_tensor(RepItem.output)
        elif mode == "grad":
            return for_each_tensor(RepItem.input_grads)
        else:
            print("{} not available".format(mode))

    def _is_int(self, string):
        try:
            int(string)
            return True
        except:
            return False

    def _compare_and_show_message(self, t1, t2):
        try:
            assert_tensor_equal(t1, t2, self.options["atol"], self.options["rtol"], self.options["compare_mode"])
            return True
        except Exception as e:
            print(str(e), "\n")
            return False
