# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from .checker import check_report, check_params, check_weights, check_grads
from ..utils import logger
import os


def compare_dumps(dump_path1, dump_path2, cfg=None, diff_phase="both"):
    # check report
    logger.info("🔍 Start comparison report (check_report)...")
    try:
        report_success = check_report(dump_path1, dump_path2, cfg=cfg, diff_phase=diff_phase)
        if report_success:
            logger.info("✅ check_report: SUCCESS !!!\n")
        else:
            logger.error("❌ check_report: FAILED !!!\n")
    except Exception as e:
        logger.error(f"❌ check_report: FAILED with error: {e}\n")
        report_success = False

    # check grads
    grads_success = None
    if os.path.exists(f"{dump_path1}/grads.json") and os.path.exists(f"{dump_path2}/grads.json"):
        logger.info("🔍 Start comparison grads (check_grads)...")
        try:
            grads_success = check_grads(dump_path1, dump_path2, cfg=cfg)
            if grads_success:
                logger.info("✅ check_grads: SUCCESS !!!\n")
            else:
                logger.error("❌ check_grads: FAILED !!!\n")
        except Exception as e:
            logger.error(f"❌ check_grads: FAILED with error: {e}\n")
            grads_success = False

    # check weights
    weights_success = None
    if os.path.exists(f"{dump_path1}/weights.json") and os.path.exists(f"{dump_path2}/weights.json"):
        logger.info("🔍 Start comparison weights (check_weights)...")
        try:
            weights_success = check_weights(dump_path1, dump_path2, cfg=cfg)
            if weights_success:
                logger.info("✅ check_weights: SUCCESS !!!\n")
            else:
                logger.error("❌ check_weights: FAILED !!!\n")
        except Exception as e:
            logger.error(f"❌ check_weights: FAILED with error: {e}\n")
            weights_success = False

    # check params
    params_success = None
    if os.path.exists(f"{dump_path1}/params.json") and os.path.exists(f"{dump_path2}/params.json"):
        logger.info("🔍 Start comparison all parameters (check_params)...")
        try:
            params_success = check_params(dump_path1, dump_path2, cfg=cfg)
            if params_success:
                logger.info("✅ check_params: SUCCESS !!!\n")
            else:
                logger.error("❌ check_params: FAILED !!!\n")
        except Exception as e:
            logger.error(f"❌ check_params: FAILED with error: {e}\n")
            params_success = False

    # final result
    success = report_success
    for res in [grads_success, weights_success, params_success]:
        if res is not None:
            success = success and res
    if success:
        logger.info(f"🎉 final comparison result: SUCCESS !!!")
    else:
        logger.error(f"❌ final comparison result: FAILED !!!")
    return success


if __name__ == "__main__":
    pt_dump_path = "transformer4sr/padiff_dump/model_PT"
    pd_dump_path = "paddle_project/padiff_dump/model_PD"
    compare_dumps(pt_dump_path, pd_dump_path)
