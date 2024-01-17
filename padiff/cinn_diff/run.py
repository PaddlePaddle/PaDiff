# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from padiff import cinn_diff
import json


def run(run_script, base_env, cinn_env):
    run_env = cinn_diff.Env(run_script, base_env, cinn_env)
    run_env.run_base_model()  # 可以提供选项选择不运行base model
    run_env.run_cinn_model()
    ret = cinn_diff.auto_diff(run_env.base_path, run_env.cinn_path, rtol=0, atol=0)
    with open("./cmp_ret.json", "w") as jsonf:
        json.dump(ret, jsonf, indent=4)


if __name__ == "__main__":
    _base_env = {
        "CUDA_VISIBLE_DEVICES": "7",
        "NVIDIA_TF32_OVERRIDE": "1",
        "CUDA_LAUNCH_BLOCKING": "1",
        "FLAGS_cudnn_deterministc": "1",
        "FLAGS_cinn_cudnn_deterministc": "1",
        "FLAGS_prim_all": "true",
    }
    _cinn_env = {
        "FLAGS_use_cinn": "1",
        "FLAGS_deny_cinn_ops": "reduce_sum",
        "FLAGS_use_reduce_split_pass": "1",
        "FLAGS_nvrtc_compile_to_cubin": "0",
        "FLAGS_cinn_use_op_fusion": "1",
        "FLAGS_cinn_parallel_compile_size": "8",
        "FLAGS_cinn_pass_visualize_dir": "",
    }
    run_script = "/root/dev/PaddleNLP/model_zoo/bert/run_bert.sh"
    run(run_script, _base_env, _cinn_env)
