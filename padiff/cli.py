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
import yaml
from .ast_injector import create_injected_script
from .utils import logger
from .comparison import compare_dumps


def load_yaml_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")

    _, ext = os.path.splitext(config_path)
    ext = ext.lower()

    with open(config_path, "r") as f:
        if ext in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")

    if "CLI" not in config:
        raise ValueError("Config file must contain a 'CLI' section.")

    cli_cfg = {}
    for k, v in config["CLI"].items():
        cli_cfg[k] = v

    guard_cfg = dict(config.get("PaDiffGuard") or {})
    compare_cfg = dict(config.get("COMPARE") or {})

    return cli_cfg, guard_cfg, compare_cfg


def run_with_padiff(
    cmd: str,
    framework: str,
    model_name: str = "model",
    optim_name=None,
    mode="base",
    alignment_dir=None,
    guard_cfg=None,
):
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

    script_kwargs = guard_cfg.copy()
    if mode == "base":
        script_kwargs.pop("single_step_mode", None)
        script_kwargs.pop("load_init_weights", None)
        script_kwargs.pop("load_first_inputs", None)
        script_kwargs.pop("base_dump_path", None)
    elif mode == "align":
        script_kwargs["base_dump_path"] = alignment_dir
        logger.warning(
            "The current injection only support 'keys_mapping' parameter of type dict 'dict' for loading "
            "init weights. If you want to pass in Callable type which also supported by function "
            "'load_init_weights_from_dump(...)', please manually modify the injected script "
            f"'debug_inject_{framework}.py' and pass it to 'PaDiffGuard(...)'."
        )
    else:
        logger.error(f"Invalid mode: {mode}. Must be 'base' or 'align'.")
        sys.exit(1)

    if optim_name is not None:
        script_kwargs["optimizer"] = optim_name

    # run injected script
    try:
        injected_script = create_injected_script(script_path, framework, model_name, mode, **script_kwargs)
    except Exception as e:
        logger.error(f"Failed to inject script: {e}")
        sys.exit(1)

    injected_filename = os.path.basename(injected_script)
    new_cmd = ["python", injected_filename] + parts[2:]
    logger.info(f"Running: {' '.join(new_cmd)}")
    script_dir = os.path.dirname(os.path.abspath(script_path))
    result = subprocess.run(new_cmd, text=True, cwd=script_dir)
    dump_path = os.path.join(script_dir, "padiff_dump", f"model_{framework.lower()}")

    # error
    if result.returncode != 0:
        logger.error(f"{framework} command failed with code {result.returncode}")
        sys.exit(result.returncode)
    return dump_path


def main():
    parser = argparse.ArgumentParser(
        description="PaDiff: Paddle & PyTorch Model Diff Tool",
        epilog="""
        === PaDiff 参数使用详解 ===

        本工具通过静态代码注入（AST）自动分析您的模型。
           * 请将参数写入一个文件，然后通过 --config 选项加载
           * 同时提供少量命令行参数，这些参数会覆盖配置文件中的同名参数

        1. 配置文件路径 (--config):
            * 必需 使用配置文件
            * 支持格式: .yaml, .yml

            示例：
              --config "/path/to/your/config.yaml"

        2. 命令参数 (--pt_cmd, --pd_cmd):
           * 必需 被包含在 config 文件中，或 通过命令行传入
           * 这些参数是您运行原始模型的完整命令
           * 通常以 'python' 开头
           * 必须指向包含您模型代码的 Python 脚本

           示例：
              --pt_cmd "python /path/to/your/torch_script.py"
              --pd_cmd "python /path/to/your/paddle_script.py"

        3. base 框架设置参数 (--base_framework):
           * 非必需 被包含在 config 文件中，或 通过命令行传入
           * 设置对齐方向，默认值 'troch'，即 paddle 代码向 torch 代码对齐
           * 影响逐层对齐时，是读取哪个框架下保存的数据

           示例：
              --base_framework "torch"

        4. 日志目录参数 (--log_dir):
           * 可选参数
           * 指定生成报告和日志的目录
           * 默认值: ./padiff_log

        使用示例:
            python -m padiff.cli --config config.yaml --pt_cmd xxx --pd_cmd xxx --log_dir xxx

        配置文件示例：
            CLI:
                pt_cmd: "python torch_project/run.py"
                pd_cmd: "python paddle_project/run.py"
                pt_model_name: "pt_model"
                pd_model_name: "pd_model"
                pt_optim_name: "pt_optimizer"   # not required
                pd_optim_name: "pd_optimizer"   # not required
                base_framework: "torch"   # not required
                log_dir: "./padiff_log"   # not required

            PaDiffGuard:
                align_depth: 1   # not required
                single_step_mode: "forward"   # not required
                max_calls: 1   # not required
                load_init_weights: false   # not required
                load_first_inputs: false   # not required
                black_list: []   # not required
                keys_mapping:   # not required, only support 'dict' now when using cli command
                    "parm_name_of_paddle_model": "parm_name_of_torch_model"
                    "model_pd.layers.0.input_layernorm.weight": "model_pt.layers.0.input_layernorm.weight"

            COMPARE:   # not required
                atol: 1.0e-06
                rtol: 1.0e-06
                compare_mode: "mean"
                action_name: "equal"
                check_mode: "fast"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--pt_cmd",
        type=str,
        help="Override 'pt_cmd' (pyTorch command) in config, e.g., 'python /torch_dir/torch_model.py'",
    )
    parser.add_argument(
        "--pd_cmd",
        type=str,
        help="Override 'pd_cmd' (paddle command) in config, e.g., 'python /paddle_dir/paddle_model.py'",
    )
    parser.add_argument(
        "--base_framework",
        type=str,
        choices=["torch", "paddle"],
        help="Which framework's output should be used as the base framework (base). The other will align to it. Options: 'torch' or 'paddle'.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./padiff_log",
        help="Override 'log_dir' (directory to save logs and reports) in config. (default: './padiff_log')",
    )

    args = parser.parse_args()

    try:
        cli_cfg, guard_cfg, compare_cfg = load_yaml_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    if args.pt_cmd:
        cli_cfg["pt_cmd"] = args.pt_cmd
    if args.pd_cmd:
        cli_cfg["pd_cmd"] = args.pd_cmd
    if args.log_dir:
        cli_cfg["log_dir"] = args.log_dir
    if args.base_framework:
        cli_cfg["base_framework"] = args.base_framework

    log_dir = cli_cfg.pop("log_dir", "./padiff_log")
    logger.reset_dir(log_dir)

    pt_cmd = cli_cfg.get("pt_cmd")
    pd_cmd = cli_cfg.get("pd_cmd")
    if not pt_cmd or not pd_cmd:
        logger.error("Both 'pt_cmd' and 'pd_cmd' must be provided (via config or command line).")
        parser.print_help()
        sys.exit(1)

    pt_model_name = cli_cfg.get("pt_model_name", "model")
    pd_model_name = cli_cfg.get("pd_model_name", "model")
    pt_optim_name = cli_cfg.get("pt_optim_name")
    pd_optim_name = cli_cfg.get("pd_optim_name")

    tasks = {
        "torch": {"cmd": pt_cmd, "framework": "torch", "model_name": pt_model_name, "optim_name": pt_optim_name},
        "paddle": {"cmd": pd_cmd, "framework": "paddle", "model_name": pd_model_name, "optim_name": pd_optim_name},
    }
    base_fw = cli_cfg.pop("base_framework", "torch")
    align_fw = "paddle" if base_fw == "torch" else "torch"

    logger.info("Code injection and script execution...")
    try:
        logger.info(f"Running {base_fw.upper()} as BASE")
        base_path = run_with_padiff(mode="base", alignment_dir=None, guard_cfg=guard_cfg, **tasks[base_fw])

        logger.info(f"Running {align_fw.upper()} in ALIGN mode")
        align_path = run_with_padiff(mode="align", alignment_dir=base_path, guard_cfg=guard_cfg, **tasks[align_fw])

        pt_dump_path = base_path if base_fw == "torch" else align_path
        pd_dump_path = base_path if base_fw == "paddle" else align_path
    except Exception as e:
        logger.error(f"An error occurred during execution: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    logger.info("Running comparison...")
    try:
        compare_dumps(pt_dump_path, pd_dump_path, cfg=compare_cfg)
    except Exception as e:
        logger.error(f"Failed to run compare_dumps: {e}. Please manually run function padiff.compare_dumps(...)")


if __name__ == "__main__":
    main()
