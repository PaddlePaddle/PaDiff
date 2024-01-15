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

from analyze import auto_diff
from env import Env


def run(run_script, base_env, cinn_env):
    run_env = Env(run_script, base_env, cinn_env)
    run_env.run_base_model()  # 可以提供选项选择不运行base model
    run_env.run_cinn_model()
    auto_diff(run_env.base_path, run_env.cinn_path, rtol=0, atol=0)


if __name__ == "__main__":
    run_script = "/root/dev/PaddleNLP/model_zoo/bert/run_bert.sh"
    run(run_script, None, None)
