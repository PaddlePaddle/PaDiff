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


from .actions import get_action
from .checker_utils import clone_dict_tree, print_runtime_info, reorder_and_match_sublayers, load_json
from ..utils import log
import json


def check_runtime(report_path_0, report_path_1, cfg):
    report_0 = load_json(report_path_0)
    report_1 = load_json(report_path_1)

    reports = [report_0, report_1]
    roots = [clone_dict_tree(rep["tree"]) for rep in reports]

    # forward check
    res = check_forward(roots, reports, cfg)
    if res == False:
        return False
    log("forward stage compared.")

    # backward check
    res = check_backward(roots, reports, cfg)
    if res == False:
        return False
    log("backward stage compared.")

    return True


def check_forward(nodes, reports, cfg):
    act = get_action(reports[0], nodes[0], reports[1], nodes[1])
    try:
        act(nodes[0]["fwd_outputs"], nodes[1]["fwd_outputs"], cfg)
        return True
    except Exception as e:
        compare_info = e
        if len(nodes[0].children) == 0 or len(nodes[1].children) == 0:
            print_runtime_info(nodes, reports, e, "Forward")
            return False

    # reorder current level
    try:
        if not nodes[1]["reordered"]:
            reorder_and_match_sublayers(nodes, reports)
    except Exception as e:
        log(f"While checking forward, diff found at base_model {nodes[0]["name"]} vs raw_model {nodes[1]["name"]}")
        log("Call `reorder_and_match_sublayers` for more detailed infos, but error occurs again:")
        print(type(e).__name__ + ":  " + str(e))
        log("Compare detail:")
        print_runtime_info(nodes, reports, compare_info, "Forward")
        return False

    for child_0, child_1 in zip(nodes[0]["children"], nodes[1]["children"]):
        res = check_forward((child_0, child_1), reports, cfg)
        if res == False:
            return False

    # sublayers is compared ok, but diff found at father layer
    log(
        f"Sublayers of {nodes[0]["name"]} and {nodes[1]["name"]} are corresponded, but diff found at their output!"
    )
    print_runtime_info(nodes, reports, compare_info, "Forward")
    return False


def check_backward(nodes, reports, cfg):
    act = get_action(reports[0], nodes[0], reports[1], nodes[1])
    try:
        act(nodes[0]["bwd_outputs"], nodes[1]["bwd_outputs"], cfg)
        return True
    except Exception as e:
        compare_info = e
        if len(nodes[0].children) == 0 or len(nodes[1].children) == 0:
            print_runtime_info(nodes, reports, e, "Backward")
            return False

    # reorder current level
    try:
        if not nodes[1]["reordered"]:
            reorder_and_match_sublayers(nodes, reports)
    except Exception as e:
        log(f"While checking backward, diff found at base_model {nodes[0]["name"]} vs raw_model {nodes[1]["name"]}")
        log("Call `reorder_and_match_sublayers` for more detailed infos, but error occurs again:")
        print(type(e).__name__ + ":  " + str(e))
        log("Compare detail:")
        print_runtime_info(nodes, reports, compare_info, "Backward")
        return False

    for child_0, child_1 in reversed(zip(nodes[0]["children"], nodes[1]["children"])):
        res = check_forward((child_0, child_1), reports, cfg)
        if res == False:
            return False

    # sublayers is compared ok, but diff found at father layer
    log(
        f"Grad of sublayer {nodes[0]["name"]} and {nodes[1]["name"]} are corresponded, but current grad found diff!"
    )
    print_runtime_info(nodes, reports, compare_info, "Backward")
    return False
