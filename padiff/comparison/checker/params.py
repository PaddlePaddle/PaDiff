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

import os.path as osp
from itertools import zip_longest

from ...utils import logger, build_file_name, get_all_valid_path, load_json, traversal_node
from ...configs import parse_cfg, global_yaml_loader
from ..actions import get_action


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
        weight_rst = weight_rst and _check_params_impl(node_lists, reports, "weights", cfg)
        grad_rst = grad_rst and _check_params_impl(node_lists, reports, "grads", cfg)
    return weight_rst and grad_rst


def check_weights(report_path_0, report_path_1, cfg=None):
    return _check_param_impl(report_path_0, report_path_1, "weights", cfg)


def check_grads(report_path_0, report_path_1, cfg=None):
    return _check_param_impl(report_path_0, report_path_1, "grads", cfg)


def _check_param_impl(report_path_0, report_path_1, compare_target, cfg=None):
    cfg = parse_cfg(cfg)
    logger.info(f"Check {compare_target} cfg: {cfg}")

    param_rst = True
    all_ranks_path_0, all_ranks_path_1 = get_all_valid_path(report_path_0, report_path_1)
    for path_0, path_1 in zip(all_ranks_path_0, all_ranks_path_1):
        reports = [load_json(path_0, f"{compare_target}.json"), load_json(path_1, f"{compare_target}.json")]
        node_lists = [traversal_node(rep["tree"], []) for rep in reports]

        logger.info(f"Checking {compare_target} in {path_0} and {path_1}")
        param_rst = param_rst and _check_params_impl(node_lists, reports, compare_target, cfg)
    return param_rst


def _check_params_impl(node_lists, reports, compare_target, cfg):
    diff_found = False
    log_name = build_file_name(reports[0], compare_target + "_diff")
    if osp.exists(osp.join(logger.log_path, log_name)):
        with open(osp.join(logger.log_path, log_name), "w") as f:
            pass

    action_name = cfg.get("action_name", None)
    act = get_action(name=action_name)

    for node_0, node_1 in zip_longest(node_lists[0], node_lists[1], fillvalue=None):
        if node_0 is None or node_1 is None:
            raise RuntimeError("Found model with difference number of sublayers. Check your model.")

        for (param_name_0, param_path_0), (param_name_1, param_path_1) in zip(
            node_0[compare_target].items(),
            node_1[compare_target].items(),
        ):
            try:
                assert (
                    param_path_0 is not None and param_path_1 is not None
                ), f"{compare_target.capitalize()} for at least one of base or raw model is not found."

                settings = global_yaml_loader.get_weight_settings(
                    (node_0["name"], node_1["name"]),
                    (reports[0]["framework"], reports[1]["framework"]),
                    (param_name_0, param_name_1),
                )
                settings.update(cfg)

                file_list_0 = [{"path": param_path_0}]
                file_list_1 = [{"path": param_path_1}]

                act(file_list_0, file_list_1, settings)

            except Exception as e:
                diff_found = True
                info = (
                    f"=========================\n"
                    f"FAILED!!! {compare_target.capitalize()} Mismatch:\n"
                    f"   Layer: {node_0['name']}(base) vs {node_1['name']}(raw)\n"
                    f"   Route: {node_0['route']}.{param_name_0}(base) vs {node_1['route']}.{param_name_1}(raw)\n"
                    f"{e}\n\n"
                )
                logger.log_file(log_name, "a", info)

    if diff_found:
        logger.error(f"The {compare_target} comparing failed !!! Please check report '{logger.log_path}/{log_name}'.")
        return False

    logger.info(f"The {compare_target} comparing compared.")
    return True
