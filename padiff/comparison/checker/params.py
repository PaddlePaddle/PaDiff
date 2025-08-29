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


from ...utils import logger, build_file_name, get_all_valid_path, load_json, traversal_node
from ...configs import parse_cfg
from .base import assert_weight, assert_grad, process_each_param


def check_params(report_path_0, report_path_1, cfg=None):
    cfg = parse_cfg(cfg)
    logger.info(f"Check params cfg: {cfg}")

    weight_rst = True
    grad_rst = True
    all_ranks_path_0, all_ranks_path_1 = get_all_valid_path(report_path_0, report_path_1)
    for path_0, path_1 in zip(all_ranks_path_0, all_ranks_path_1):
        reports = [load_json(path_0, "params.json"), load_json(path_1, "params.json")]
        node_lists = [traversal_node(rep["tree"], []) for rep in reports]

        logger.info(f"Checking params in {path_0} and {path_1}")
        weight_rst = weight_rst and check_target(assert_weight, node_lists, reports, "weights", cfg)
        grad_rst = grad_rst and check_target(assert_grad, node_lists, reports, "grads", cfg)
    return weight_rst and grad_rst


def check_weights(report_path_0, report_path_1, cfg=None):
    cfg = parse_cfg(cfg)
    logger.info(f"Check weights cfg: {cfg}")

    weight_rst = True
    all_ranks_path_0, all_ranks_path_1 = get_all_valid_path(report_path_0, report_path_1)
    for path_0, path_1 in zip(all_ranks_path_0, all_ranks_path_1):
        reports = [load_json(path_0, "weights.json"), load_json(path_1, "weights.json")]
        node_lists = [traversal_node(rep["tree"], []) for rep in reports]

        logger.info(f"Checking weights in {path_0} and {path_1}")
        weight_rst = weight_rst and check_target(assert_weight, node_lists, reports, "weights", cfg)
    return weight_rst


def check_grads(report_path_0, report_path_1, cfg=None):
    cfg = parse_cfg(cfg)
    logger.info(f"Check grads cfg: {cfg}")

    grad_rst = True
    all_ranks_path_0, all_ranks_path_1 = get_all_valid_path(report_path_0, report_path_1)
    for path_0, path_1 in zip(all_ranks_path_0, all_ranks_path_1):
        reports = [load_json(path_0, "grads.json"), load_json(path_1, "grads.json")]
        node_lists = [traversal_node(rep["tree"], []) for rep in reports]

        logger.info(f"Checking grads in {path_0} and {path_1}")
        grad_rst = grad_rst and check_target(assert_grad, node_lists, reports, "grads", cfg)
    return grad_rst


def check_target(fn, node_lists, reports, compare_target, cfg):
    flag = True
    log_name = build_file_name(reports[0], compare_target + "_diff")

    def checker(nodes, param_names, params, settings):
        try:
            fn(params, settings)
        except Exception as e:
            nonlocal flag
            flag = False
            info = (
                "=" * 25 + "\n" + "{} value is different.\n"
                "between base_model: {}\n"
                "        raw_model:  {}\n\n"
                "base_model param path:\n    {}\n"
                "raw_model param path:\n    {}\n\n"
                "{}\n\n".format(
                    compare_target,
                    nodes[0]["repr"],
                    nodes[1]["repr"],
                    nodes[0]["route"] + "." + param_names[0],
                    nodes[1]["route"] + "." + param_names[1],
                    type(e).__name__ + ":  " + str(e),
                )
            )
            logger.log_file(log_name, "a", info)

    try:
        process_each_param(checker, node_lists, reports, compare_target, cfg)
    except Exception as e:
        logger.error("=" * 10 + f"Err occurs when compare {compare_target}!!!" + "=" * 10 + "\n" + str(e))
        return False

    if flag == False:
        logger.info(
            f"Diff found when compare {compare_target}, please check report \n        {logger.log_path}/{log_name}"
        )
    else:
        logger.info(f"{compare_target} compared.")

    return flag
