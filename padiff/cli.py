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
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")


def run_with_padiff(
    cmd: str,
    framework: str,
    model_name: str = "model",
    optim_name=None,
    mode="base",
    alignment_dir=None,
    **kwargs,
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

    if optim_name is not None:
        kwargs["optimizer"] = optim_name

    # run injected script
    injected_script = create_injected_script(script_path, framework, model_name, mode, alignment_dir, **kwargs)
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

        本工具通过静态代码注入（AST）自动分析您的模型。为确保注入成功，请正确设置以下参数。
        注意，除了使用命令行外，您可以将所有参数写入一个文件，然后通过 --config 选项加载。
           * 支持格式: .yaml, .yml
           * .yaml 格式示例:
                pt_cmd: python torch_model.py
                pd_cmd: python paddle_model.py
                align_depth: inf
           * 使用方式: padiff --config config.yaml
           * 命令行参数会覆盖配置文件中的同名参数。

        1. 命令参数 (--pt_cmd, --pd_cmd):
           这些参数是您运行原始模型的完整命令。
           * 通常以 'python' 开头。
           * 必须指向包含您模型代码的 Python 脚本。

           示例：
              --pt_cmd "python /path/to/your/torch_script.py"
              --pd_cmd "python /path/to/your/paddle_script.py"

        2. 模型变量名参数 (--pt_model_name, --pd_model_name):
           这些参数指定您在脚本中创建模型实例的**变量名**。
           * 它们不是类名，也不是文件名。
           * 它们是模型实例化时 `=` 左边的标识符。

           示例：
              如果您的 PyTorch 脚本中有：
                my_torch_model = MyNet()
                output = my_torch_model(input_tensor)
              那么您应该使用：
                --pt_model_name my_torch_model

              如果您的 Paddle 脚本中有：
                net = SimplePaddle()
                out = net.generate(input_tensor)
              那么您应该使用：
                --pd_model_name net

        3. 优化器名参数 (--pt_optim_name, --pd_optim_name):
           这些参数指定您在脚本中创建优化器实例的**变量名**。
           * 它们不是类名，也不是文件名。
           * 它们是优化器实例化时 `=` 左边的标识符。
           * 默认值: None (不传递优化器)

           示例：
              如果您的 PyTorch 脚本中有：
                optim = torch.optim.Adam(
                    transformer.parameters(),
                    lr=1.0,
                    betas=(0.9, 0.98),
                    eps=1e-9,
                )
              那么您应该使用：
                --pt_optim_name optim

              如果您的 Paddle 脚本中有：
                optimizer = paddle.optimizer.Adam(
                    parameters=transformer.parameters(),
                    learning_rate=1.0,
                    epsilon=1e-09,
                    beta1=0.9,
                    beta2=0.98,
                    weight_decay=0.0,
                )
              那么您应该使用：
                --pd_optim_name optimizer

        4. 日志目录参数 (--log_dir):
           指定生成报告和日志的目录。
           * 默认值: ./padiff_log

        5. 对齐深度参数 (--align_depth):
           控制对齐的粒度。通过指定一个深度值，可以忽略该深度以下的所有子模块。
           * 值为整数: 指定一个具体的深度。例如，--align_depth 1 会忽略深度为1及以下的所有子模块。
           * 值为 'inf': (默认) 无限深度，会对齐到最细粒度的层（如 Linear, ReLU）。
           * 值为整数，且数值超过模型最大迭代深度时，相当于 'inf'。
           * 示例：
              --align_depth 0  # 只对齐顶层模块
              --align_depth 1  # 对齐到第一层子模块
              --align_depth inf # 对齐到最细粒度

        6. 单步对齐模式参数 (--single_step_mode):
           启用逐层对齐模式。
           * 可选值: forward, backward, both
           * 默认值: None (禁用)
           * 当启用时，工具会从自动加载基准模型的输出，并用其替换对齐模型的相应层输出。

        7. 结果对比参数:
           控制模型输出结果的对比精度和模式。
           * --atol: 绝对误差容忍度 (default: 1e-6)
           * --rtol: 相对误差容忍度 (default: 1e-6)
           * --compare_mode: 对比模式。可选值: mean, strict, abs_mean (default: mean)
           * --action_name: 激活函数名称，可选值: equal, loose_equal, 用于特定的对比逻辑 (default: equal)
           * 示例:
              --atol 1e-4 --rtol 1e-5 --compare_mode mean --action_name equal

        使用示例:
            padiff \\
              --pt_cmd "python torch_model.py" \\
              --pd_cmd "python paddle_model.py" \\
              --pt_model_name "model" \\
              --pd_model_name "model" \\
              --pt_optim_name "optimizer" \\
              --pd_optim_name "optimizer" \\
              --log_dir "./my_alignment_results" \\
              --align_depth 1 \\
              --single_step_mode "forward" \\
              --atol 1e-4 \\
              --rtol 1e-5 \\
              --compare_mode mean \\
              --action_name equal
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--pt_cmd", type=str, help='PyTorch command, e.g., "python /torch_dir/torch_model.py"')
    parser.add_argument("--pd_cmd", type=str, help='Paddle command, e.g., "python /paddle_dir/paddle_model.py"')
    parser.add_argument(
        "--pt_model_name",
        type=str,
        default="model",
        help="The model name that appears in the pytorch script's code (default: 'model')",
    )
    parser.add_argument(
        "--pd_model_name",
        type=str,
        default="model",
        help="The model name that appears in the paddle script's code (default: 'model')",
    )
    parser.add_argument(
        "--pt_optim_name",
        type=str,
        default=None,
        help="The model name that appears in the pytorch script's code (default: 'model')",
    )
    parser.add_argument(
        "--pd_optim_name",
        type=str,
        default=None,
        help="The model name that appears in the paddle script's code (default: 'model')",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./padiff_log",
        help="Directory to save logs and reports (default: './padiff_log')",
    )
    parser.add_argument("--align_depth", type=str, default="inf", help="Depth of alignment (default: 'inf')")
    parser.add_argument(
        "--single_step_mode",
        choices=["forward", "backward", "both"],
        default=None,
        help="Enable single-step alignment mode. Choices: forward, backward, both. (default: None, disabled)",
    )
    parser.add_argument(
        "--black_list",
        type=str,
        nargs="*",
        help="List of layer names to add to the black list.",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-6, help="Absolute tolerance for result comparison (default: 1e-4)"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-6, help="Relative tolerance for result comparison (default: 1e-6)"
    )
    parser.add_argument(
        "--compare_mode",
        choices=["mean", "strict", "abs_mean"],
        default="mean",
        help="Comparison mode for result checking. Choices: mean, strict, abs_mean (default: mean)",
    )
    parser.add_argument(
        "--action_name",
        choices=["equal", "loose_equal"],
        default="equal",
        help="Activation function name for specific comparison logic. Choices: equal, loose_equal (default: equal)",
    )

    known_args, _ = parser.parse_known_args()
    if known_args.config:
        try:
            config_args = load_yaml_config(known_args.config)
            parser.set_defaults(**config_args)
            args = parser.parse_args()
        except Exception as e:
            print(f"Error loading config file {known_args.config}: {e}")
            sys.exit(1)

    args = parser.parse_args()
    args_dict = vars(args)
    config_path = args_dict.pop("config", None)
    if config_path:
        logger.info(f"Configuration loaded from: {config_path}")

    pt_cmd = args_dict.pop("pt_cmd", None)
    pd_cmd = args_dict.pop("pd_cmd", None)
    if pt_cmd is None or pd_cmd is None:
        logger.error("--pt_cmd and --pd_cmd are required. You must provide it via command line or in the config file.")
        parser.print_help()
        sys.exit(1)

    log_dir = args_dict.pop("log_dir", "./padiff_log")
    logger.reset_dir(log_dir)

    pt_model_name = args_dict.pop("pt_model_name", "model")
    pd_model_name = args_dict.pop("pd_model_name", "model")

    single_step_mode_value = args_dict.pop("single_step_mode", None)
    pd_kwargs = dict(args_dict)
    if single_step_mode_value is not None:
        pd_kwargs["single_step_mode"] = single_step_mode_value

    compare_cfg = {
        "atol": args_dict.pop("atol", 1.0e-4),
        "rtol": args_dict.pop("rtol", 1.0e-6),
        "compare_mode": args_dict.pop("compare_mode", "mean"),
        "action_name": args_dict.pop("action_name", "equal"),
    }

    pt_optim_name = args_dict.pop("pt_optim_name", None)
    pd_optim_name = args_dict.pop("pd_optim_name", None)

    pt_dump_path = run_with_padiff(pt_cmd, "torch", pt_model_name, pt_optim_name, **args_dict)
    pd_dump_path = run_with_padiff(pd_cmd, "paddle", pd_model_name, pd_optim_name, "align", pt_dump_path, **pd_kwargs)

    logger.info("Running comparison...")
    try:
        compare_dumps(pt_dump_path, pd_dump_path, cfg=compare_cfg)
    except Exception as e:
        logger.error(f"Failed to run compare_dumps: {e}. Please manually run function padiff.compare_dumps(...)")


if __name__ == "__main__":
    main()
