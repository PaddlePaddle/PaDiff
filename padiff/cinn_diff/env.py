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

import os
import subprocess
from .logs import logger


class Env:

    base_dir_name = "base"
    cinn_dir_name = "cinn"
    cinn_pass_dir = "cinn_pass"
    cinn_graph_dir = "cinn_graph"

    def __init__(
        self,
        script=None,
        base_env=None,
        cinn_env=None,
    ):
        self._base_env = {
            "CUDA_VISIBLE_DEVICES": "7",
            "NVIDIA_TF32_OVERRIDE": "1",
            "CUDA_LAUNCH_BLOCKING": "1",
            "FLAGS_save_static_runtime_data": "1",
            "FLAGS_static_runtime_data_save_path": "./",
            "FLAGS_cudnn_deterministc": "1",
            "FLAGS_cinn_cudnn_deterministc": "1",
            "FLAGS_prim_all": "true",
        }
        self._cinn_env = {
            "FLAGS_use_cinn": "1",
            "FLAGS_deny_cinn_ops": "",
            "FLAGS_use_reduce_split_pass": "1",
            "FLAGS_nvrtc_compile_to_cubin": "0",
            "FLAGS_cinn_use_op_fusion": "1",
            "FLAGS_cinn_parallel_compile_size": "8",
            "FLAGS_cinn_pass_visualize_dir": "",
        }
        self.base_env = base_env if base_env else self._base_env
        self.cinn_env = cinn_env if cinn_env else self._cinn_env
        self.base_path = os.path.join(os.path.dirname(script), self.base_dir_name)
        self.cinn_path = os.path.join(os.path.dirname(script), self.cinn_dir_name)
        self.script = script
        self.script_path = os.path.dirname(script)
        self.script_name = os.path.basename(script)
        self.os_env = dict(os.environ)

    def init_base_env(self):
        if os.path.exists(self.base_path):
            logger.info("base path exists, remove it")
            os.system("rm -rf " + self.base_path)
        self.base_env["FLAGS_static_runtime_data_save_path"] = self.base_path
        self.base_env["FLAGS_save_static_runtime_data"] = "1"

    def set_base_env(self, env):
        self.base_env = env

    def init_cinn_env(self):
        self.base_env["FLAGS_static_runtime_data_save_path"] = self.cinn_path
        if os.path.exists(self.cinn_path):
            logger.info("cinn path exists, remove it")
            os.system("rm -rf " + self.cinn_path)
        self.cinn_env["FLAGS_cinn_pass_visualize_dir"] = os.path.join(self.cinn_path, self.cinn_pass_dir)
        self.cinn_env["FLAGS_cinn_subgraph_graphviz_dir"] = os.path.join(self.cinn_path, self.cinn_graph_dir)

    def set_cinn_env(self, env):
        self.cinn_env = env

    def set_script(self, name):
        self.script = name

    def run_model(self, run_env, log):
        logger.info(self.script)
        ret = subprocess.run(["sh", self.script_name], env=run_env, stdout=log, stderr=log)
        logger.info(ret)

    def run_base_model(self):
        self.init_base_env()
        root_path = os.getcwd()
        os.chdir(self.script_path)
        run_env = self.base_env.copy()
        logger.info(run_env)
        run_env.update(self.os_env)
        base_log = open("base.log", "w")
        self.run_model(run_env, base_log)
        base_log.close()
        os.chdir(root_path)

    def run_cinn_model(self):
        self.init_cinn_env()
        root_path = os.getcwd()
        os.chdir(self.script_path)
        run_env = self.cinn_env.copy()
        base_env = self.base_env.copy()
        for key in base_env:
            if key not in run_env:
                run_env[key] = base_env[key]
        logger.info(run_env)
        run_env.update(self.os_env)
        cinn_log = open("cinn.log", "w")
        self.run_model(run_env, cinn_log)
        cinn_log.close()
        os.chdir(root_path)
