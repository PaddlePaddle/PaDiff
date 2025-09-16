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


import unittest
import os
import sys
from unittest.mock import patch
from io import StringIO
import numpy as np
import yaml
from padiff.cli import main as padiff_cli_main
import tempfile
import shutil


PADDLE_SCRIPT_TEMPLATE = """
import paddle
import numpy as np

class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear({input_dim}, {hidden_dim})
        self.linear2 = paddle.nn.Linear({hidden_dim}, {output_dim})
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + residual
        x = self.linear2(x)
        return x

def main():
    inp = np.load("{input_file}")
    inp = paddle.to_tensor(inp)

    model = SimpleLayer()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.1)

    out = model(inp)
    loss = paddle.mean(out)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

if __name__ == "__main__":
    main()
"""

TORCH_SCRIPT_TEMPLATE = """
import torch
import numpy as np

class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear({input_dim}, {hidden_dim})
        self.linear2 = torch.nn.Linear({hidden_dim}, {output_dim})
        self.act = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + residual
        x = self.linear2(x)
        return x

def main():
    inp = np.load("{input_file}")
    inp = torch.as_tensor(inp)

    model = SimpleModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    out = model(inp)
    loss = torch.mean(out)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

if __name__ == "__main__":
    main()
"""


class TestCliEndToEnd(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)

        self.config_path = os.path.join(self.test_dir, "config.yaml")
        self.paddle_script_path = os.path.join(self.test_dir, "paddle_script.py")
        self.torch_script_path = os.path.join(self.test_dir, "torch_script.py")
        self.input_file = os.path.join(self.test_dir, "input.npy")
        self.log_dir = os.path.join(self.test_dir, "padiff_logs")

        os.makedirs(self.log_dir, exist_ok=True)

        inp = np.random.rand(100, 100).astype("float32")
        np.save(self.input_file, inp)
        assert os.path.exists(self.input_file), f"Input file not created: {self.input_file}"

        torch_script = TORCH_SCRIPT_TEMPLATE.format(
            input_dim=100,
            hidden_dim=100,
            output_dim=10,
            input_file="input.npy",
            dump_path="torch",
        )

        paddle_script = PADDLE_SCRIPT_TEMPLATE.format(
            input_dim=100,
            hidden_dim=100,
            output_dim=10,
            input_file="input.npy",
            dump_path="paddle",
        )

        with open(self.torch_script_path, "w") as f:
            f.write(torch_script)
        with open(self.paddle_script_path, "w") as f:
            f.write(paddle_script)

        for path in [self.torch_script_path, self.paddle_script_path]:
            assert os.path.exists(path), f"Script not created: {path}"
            assert os.path.getsize(path) > 0, f"Script is empty: {path}"

    def _create_config_file(self, overrides=None):
        config = {
            "CLI": {
                "pt_cmd": f"python {self.torch_script_path}",
                "pd_cmd": f"python {self.paddle_script_path}",
                "pt_model_name": "model",
                "pd_model_name": "model",
                "log_dir": self.log_dir,
            },
            "PaDiffGuard": {
                "align_depth": "inf",
                "single_step_mode": None,
                "max_calls": 1,
                "load_init_weights": False,
                "load_first_inputs": False,
                "black_list": [],
                "keys_mapping": None,
            },
            "COMPARE": {"atol": 1e-6, "rtol": 1e-6, "compare_mode": "mean", "action_name": "equal"},
        }

        if overrides:
            for section, updates in overrides.items():
                if section in config:
                    config[section].update(updates)

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def _run_cli_test(self, extra_args=None, expect_success=True):
        if extra_args is None:
            extra_args = []

        test_args = ["padiff", "--config", self.config_path] + extra_args

        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                with patch("sys.stderr", new=StringIO()) as fake_err:
                    try:
                        padiff_cli_main()

                        if not expect_success:
                            self.fail("CLI was expected to fail but it succeeded.")

                    except SystemExit as e:
                        if expect_success and e.code != 0:
                            self.fail(f"CLI failed with exit code {e.code}: {fake_err.getvalue()}")
                        elif not expect_success and e.code == 0:
                            self.fail(f"CLI was expected to fail but exited with code 0.")
                        if not expect_success and e.code != 0:
                            raise

                    except Exception as e:
                        if not expect_success:
                            raise
                        else:
                            self.fail(f"CLI raised an unexpected exception: {type(e).__name__}: {str(e)}")

    def test_end_to_end_basic(self):
        self._create_config_file()
        self._run_cli_test([])

    def test_end_to_end_with_optimizer(self):
        overrides = {"CLI": {"pt_optim_name": "optimizer", "pd_optim_name": "optimizer"}}
        self._create_config_file(overrides)
        self._run_cli_test()

    def test_end_to_end_with_align_depth(self):
        overrides = {"PaDiffGuard": {"align_depth": 0}}
        self._create_config_file(overrides)
        self._run_cli_test()

    def test_end_to_end_with_single_step(self):
        for mode in ["forward", "backward", "both"]:
            with self.subTest(mode=mode):
                overrides = {"PaDiffGuard": {"single_step_mode": mode}}
                self._create_config_file(overrides)
                self._run_cli_test()

    def test_end_to_end_with_black_list(self):
        overrides = {"PaDiffGuard": {"black_list": ["Linear"]}}
        self._create_config_file(overrides)
        self._run_cli_test()

    def test_end_to_end_with_different_atol_rtol(self):
        overrides = {"COMPARE": {"atol": 1e-3, "rtol": 1e-4}}
        self._create_config_file(overrides)
        self._run_cli_test()

    def test_end_to_end_with_different_compare_mode(self):
        for compare_mode in ["strict", "abs_mean"]:
            with self.subTest(compare_mode=compare_mode):
                overrides = {"COMPARE": {"compare_mode": compare_mode}}
                self._create_config_file(overrides)
                self._run_cli_test()

    def test_end_to_end_with_different_action(self):
        overrides = {"COMPARE": {"action_name": "loose_equal"}}
        self._create_config_file(overrides)
        self._run_cli_test()

    def test_end_to_end_with_command_line_override(self):
        self._create_config_file()
        extra_args = [
            "--pt_cmd",
            f"python {self.torch_script_path}".replace("run.py", "modified_run.py"),
            "--log_dir",
            os.path.join(self.test_dir, "overridden_log_dir"),
        ]
        self._run_cli_test(extra_args)

    def test_single_step_mode_with_max_calls_gt_1_raises_error(self):
        overrides = {"PaDiffGuard": {"single_step_mode": "forward", "max_calls": 2}}
        self._create_config_file(overrides)
        with self.assertRaises(SystemExit) as cm:
            self._run_cli_test(expect_success=False)

    def test_single_step_mode_with_max_calls_1_only_emits_warning(self):
        overrides = {"PaDiffGuard": {"single_step_mode": "both", "max_calls": 1}}
        self._create_config_file(overrides)
        self._run_cli_test()

    def test_multi_call_without_single_step_issues_warning(self):
        overrides = {"PaDiffGuard": {"max_calls": 3}}
        self._create_config_file(overrides)
        self._run_cli_test()


if __name__ == "__main__":
    unittest.main()
