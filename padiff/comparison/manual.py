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
from ..utils import log
import os


def compare_dumps(dump_path1, dump_path2, cfg):
    # check report
    print("🔍 Start comparison report (check_report)...")
    try:
        report_success = check_report(dump_path1, dump_path2, cfg=cfg, diff_phase="both")
        if report_success:
            log("✅ check_report: SUCCESS !!!")
        else:
            log("❌ check_report: FAILED !!!")
    except Exception as e:
        log(f"❌ check_report: FAILED with error: {e}")
        report_success = False

    # check grads
    grads_success = None
    if os.path.exists(f"{dump_path1}/grads.json") and os.path.exists(f"{dump_path2}/grads.json"):
        print("\n🔍 Start comparison grads (check_grads)...")
        try:
            grads_success = check_grads(dump_path1, dump_path2, cfg=cfg)
            if grads_success:
                log("✅ check_grads: SUCCESS !!!")
            else:
                log("❌ check_grads: FAILED !!!")
        except Exception as e:
            log(f"❌ check_grads: FAILED with error: {e}")
            grads_success = False

    # check weights
    weights_success = None
    if os.path.exists(f"{dump_path1}/grads.json") and os.path.exists(f"{dump_path2}/weights.json"):
        print("\n🔍 Start comparison weights (check_weights)...")
        try:
            weights_success = check_weights(dump_path1, dump_path2, cfg=cfg)
            if weights_success:
                log("✅ check_weights: SUCCESS !!!")
            else:
                log("❌ check_weights: FAILED !!!")
        except Exception as e:
            log(f"❌ check_weights: FAILED with error: {e}")
            weights_success = False

    # check params
    params_success = None
    if os.path.exists(f"{dump_path1}/grads.json") and os.path.exists(f"{dump_path2}/params.json"):
        print("\n🔍 Start comparison all parameters (check_params)...")
        try:
            params_success = check_params(dump_path1, dump_path2, cfg=cfg)
            if params_success:
                log("✅ check_params: SUCCESS !!!")
            else:
                log("❌ check_params: FAILED !!!")
        except Exception as e:
            log(f"❌ check_params: FAILED with error: {e}")
            params_success = False

    # final result
    success = report_success
    for res in [grads_success, weights_success, params_success]:
        if res is not None:
            success = success and res
    log(f"🏁 final comparison result: {'SUCCESS !!!' if success else 'FAILED !!!'}")


if __name__ == "__main__":
    pt_dump_path = "transformer4sr/padiff_dump/model_PT"
    pd_dump_path = "paddle_project/padiff_dump/model_PD"
    main(pt_dump_path, pd_dump_path)
