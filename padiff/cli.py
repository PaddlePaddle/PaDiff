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

import argparse
import subprocess
import os
import sys
from .ast_injector import create_injected_script
from .utils import logger
from .comparison import compare_dumps


def run_with_padiff(cmd: str, framework: str, log_dir: str = "padiff_log"):
    # parse command
    parts = cmd.split()
    if not parts or not parts[0].endswith("python"):
        logger.error(f"Command must start with 'python': {cmd}")
        sys.exit(1)

    # open script
    script_path = parts[1]
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        sys.exit(1)

    # run injected script
    injected_script = create_injected_script(script_path, framework, log_dir)
    new_cmd = ["python", injected_script] + parts[2:]
    logger.info(f"Running: {' '.join(new_cmd)}")
    result = subprocess.run(new_cmd, text=True)

    # error
    if result.returncode != 0:
        logger.error(f"{framework} command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="PaDiff: Paddle & PyTorch Model Diff Tool")
    parser.add_argument("--pd_cmd", type=str, required=True, help='Paddle command, e.g., "python ./paddle_model.py"')
    parser.add_argument("--pt_cmd", type=str, required=True, help='PyTorch command, e.g., "python ./torch_model.py"')
    parser.add_argument(
        "--pd_model_name",
        type=str,
        default="model",
        help="The model name that appears in the paddle script's code (default: 'model')",
    )
    parser.add_argument(
        "--pt_model_name",
        type=str,
        default="model",
        help="The model name that appears in the pytorch script's code (default: 'model')",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./padiff_log",
        help="Directory to save logs and reports (default: './padiff_log')",
    )

    args = parser.parse_args()

    run_with_padiff(args.pt_cmd, "torch", args.log_dir)
    run_with_padiff(args.pd_cmd, "paddle", args.log_dir)
    pt_dump = f"{args.log_dir}/padiff_dump/model_torch"
    pd_dump = f"{args.log_dir}/padiff_dump/model_paddle"

    logger.info("Running comparison...")
    try:
        compare_dumps(pt_dump, pd_dump, cfg={"atol": 1e-4})
    except Exception as e:
        logger.error(f"Failed to run compare_dumps: {e}. Please manually run function padiff.compare_dumps(...)")


if __name__ == "__main__":
    main()
