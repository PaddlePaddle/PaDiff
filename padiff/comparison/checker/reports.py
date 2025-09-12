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

from itertools import zip_longest
from ..actions import get_action
from ...utils import (
    clone_dict_tree,
    get_all_valid_path,
    load_json,
    print_report_info,
    reorder_and_match_sublayers,
    logger,
    struct_info_log,
)
from ...configs import parse_cfg
from ...abstracts import global_special_init_pool as init_pool


def check_report(report_path_0, report_path_1, cfg=None, diff_phase="both"):
    assert diff_phase in ("forward", "backward", "both"), f"Illegal diff_phase {diff_phase}"

    cfg = parse_cfg(cfg)
    logger.info(f"check cfg {cfg}")

    final_rst = True
    all_ranks_path_0, all_ranks_path_1 = get_all_valid_path(report_path_0, report_path_1)
    for path_0, path_1 in zip(all_ranks_path_0, all_ranks_path_1):
        logger.info(f"Checking report in {path_0} and {path_1}")
        final_rst = final_rst and _check_report_impl(path_0, path_1, cfg, diff_phase)
    return final_rst


def _check_report_impl(report_path_0, report_path_1, cfg=None, diff_phase="both"):
    reports = [load_json(report_path_0, "report.json"), load_json(report_path_1, "report.json")]
    trees = [rep["tree"] for rep in reports]

    assert len(trees) == 2
    assert len(trees[0]) == len(trees[1])

    roots = [[clone_dict_tree(root) for root in trees[i]] for i in range(2)]

    check_layer_map(reports)

    for root_0, root_1 in zip(roots[0], roots[1]):
        root_pair = [root_0, root_1]

        if diff_phase in ("forward", "both"):
            # forward check
            res = check_forward(root_pair, reports, cfg)
            if res == False:
                logger.error("The forward stage comparing failed !!!")
                return False
            else:
                struct_info_log(reports, root_pair, "report")

        if diff_phase in ("backward", "both"):
            # backward check
            res = check_backward(root_pair, reports, cfg)
            if res == False:
                logger.error("The backward stage comparing failed !!!")
                return False

    return True


def check_forward(nodes, reports, cfg):
    logger.debug(f"Checking forward of {nodes[0]['name']}")
    action_name = cfg.get("action_name", None)
    act = get_action(reports[0], nodes[0], reports[1], nodes[1], name=action_name)
    try:
        act(nodes[0]["fwd_outputs"], nodes[1]["fwd_outputs"], cfg)
        return True
    except Exception as e:
        compare_info = e
        if len(nodes[0]["children"]) == 0 or len(nodes[1]["children"]) == 0:
            print_report_info(nodes, reports, e, "Forward")
            return False

    # reorder current level
    try:
        if not nodes[1]["reordered"]:
            reorder_and_match_sublayers(nodes, reports)
    except Exception as e:
        msg = f"While checking forward, diff found at {nodes[0]['name']}(base) vs {nodes[1]['name']}(raw)\n"
        msg += "Call `reorder_and_match_sublayers` for more detailed infos, but error occurs again:\n"
        msg += f"{type(e).__name__}:  {str(e)}"
        logger.error(msg)
        # print_report_info(nodes, reports, compare_info, "Forward", msg)
        # return False

    for child_0, child_1 in zip(nodes[0]["children"], nodes[1]["children"]):
        res = check_forward((child_0, child_1), reports, cfg)
        if res == False:
            return False

    # sublayers is compared ok, but diff found at father layer
    msg = (
        f"\n   ⚠️ Sublayers of {nodes[0]['name']} and {nodes[1]['name']} are corresponded, but diff found at their output! "
        "\n   💡 This might be reasonable since errors accumulate if single_step mode is enabled."
    )
    print_report_info(nodes, reports, compare_info, "Forward", msg)
    return False


def check_backward(nodes, reports, cfg):
    logger.debug(f"Checking backward of {nodes[0]['name']}")
    action_name = cfg.get("action_name", None)
    act = get_action(reports[0], nodes[0], reports[1], nodes[1], name=action_name)
    try:
        act(nodes[0]["bwd_grads"], nodes[1]["bwd_grads"], cfg)
        return True
    except Exception as e:
        compare_info = e
        if len(nodes[0]["children"]) == 0 or len(nodes[1]["children"]) == 0:
            print_report_info(nodes, reports, e, "Backward")
            return False

    # reorder current level
    try:
        if not nodes[1]["reordered"]:
            reorder_and_match_sublayers(nodes, reports)
    except Exception as e:
        msg = f"While checking backward, diff found at {nodes[0]['name']}(base) vs {nodes[1]['name']}(raw)\n"
        msg += "Call `reorder_and_match_sublayers` for more detailed infos, but error occurs again:\n"
        msg += f"{type(e).__name__}:  {str(e)}"
        logger.error(msg)
        # print_report_info(nodes, reports, compare_info, "Backward", msg)
        # return False

    for child_0, child_1 in zip(reversed(nodes[0]["children"]), reversed(nodes[1]["children"])):
        res = check_backward((child_0, child_1), reports, cfg)
        if res == False:
            return False

    # sublayers is compared ok, but diff found at father layer
    msg = f"Grad of sublayer {nodes[0]['name']} and {nodes[1]['name']} are corresponded, but current grad found diff!"
    print_report_info(nodes, reports, compare_info, "Backward", msg)
    return False


def check_layer_map(reports):
    if len(reports[0]["layer_map"]["route"]) == 0 and len(reports[1]["layer_map"]["route"]) == 0:
        return True
    logger.info("Start check layer_map:")
    layer_maps = [zip(rep["layer_map"]["route"], rep["layer_map"]["fullname"]) for rep in reports]
    for base_info, raw_info in zip_longest(layer_maps[0], layer_maps[1], fillvalue=None):
        if raw_info is None or base_info is None:
            print(
                "\nError: The number of submodels which need special init is not the same! Check your layer_map first!"
            )
            return False

        base_route, base_fullname = base_info
        raw_route, raw_fullname = raw_info
        func_key = base_fullname + "###" + raw_fullname

        if func_key in init_pool.funcs.keys():
            print(
                f"++++    base_model `{base_fullname}` at `{base_route}` <==>  raw_model `{raw_fullname}` at `{raw_route}`   ++++"
            )
        else:
            print("\nError: When check layer_map in order, find that raw_model can not matchs base_model.")
            print(f"    base_model:  `{base_fullname}` at `{base_route}`")
            print(f"    raw_model: `{raw_fullname}` at `{raw_route}`")
            logger.error("Check layer_map FAILED!!!\n")
            return False

    logger.info("Check layer_map SUCCESS!!!\n")
    return True
