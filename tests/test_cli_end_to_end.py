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
import shutil
import tempfile
import sys
from unittest.mock import patch
from io import StringIO
import numpy as np
from padiff.cli import main as padiff_cli_main


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
    import os
    os.makedirs("{dump_path}", exist_ok=True)
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
    import os
    os.makedirs("{dump_path}", exist_ok=True)
    main()
"""


class TestCliEndToEnd(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)

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
            input_file=self.input_file,
            model_name="model_torch",
            dump_path=os.path.join(self.log_dir, "torch"),
        )

        paddle_script = PADDLE_SCRIPT_TEMPLATE.format(
            input_dim=100,
            hidden_dim=100,
            output_dim=10,
            input_file=self.input_file,
            model_name="model_paddle",
            dump_path=os.path.join(self.log_dir, "paddle"),
        )

        with open(self.torch_script_path, "w") as f:
            f.write(torch_script)
        with open(self.paddle_script_path, "w") as f:
            f.write(paddle_script)

        assert os.path.exists(self.torch_script_path), f"torch_script.py not created: {self.torch_script_path}"
        assert os.path.exists(self.paddle_script_path), f"paddle_script.py not created: {self.paddle_script_path}"
        assert os.path.getsize(self.torch_script_path) > 0, f"torch_script.py is empty: {self.torch_script_path}"
        assert os.path.getsize(self.paddle_script_path) > 0, f"paddle_script.py is empty: {self.paddle_script_path}"

    def _run_cli_test(self, extra_args):
        test_args = [
            "padiff",
            "--pt_cmd",
            f"python {self.torch_script_path}",
            "--pd_cmd",
            f"python {self.paddle_script_path}",
            "--pt_model_name",
            "model",
            "--pd_model_name",
            "model",
            "--log_dir",
            self.log_dir,
        ]
        test_args.extend(extra_args)

        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                with patch("sys.stderr", new=StringIO()) as fake_err:
                    try:
                        padiff_cli_main()
                    except SystemExit as e:
                        if e.code != 0:
                            self.fail(f"CLI failed with exit code {e.code}: {fake_err.getvalue()}")
                    except Exception as e:
                        self.fail(f"CLI raised an unexpected exception: {type(e).__name__}: {str(e)}")

    def test_end_to_end_basic(self):
        self._run_cli_test([])

    def test_end_to_end_with_optimizer(self):
        self._run_cli_test(["--pt_optim_name", "optimizer", "--pd_optim_name", "optimizer"])

    def test_end_to_end_with_align_depth(self):
        self._run_cli_test(["--align_depth", "0"])

    def test_end_to_end_with_single_step(self):
        self._run_cli_test(["--single_step_mode", "forward"])
        self._run_cli_test(["--single_step_mode", "backward"])
        self._run_cli_test(["--single_step_mode", "both"])

    def test_end_to_end_with_black_list(self):
        self._run_cli_test(["--black_list", "Linear"])

    def test_end_to_end_with_different_atol_rtol(self):
        self._run_cli_test(["--atol", "1e-3", "--rtol", "1e-4"])

    def test_end_to_end_with_different_compare_mode(self):
        self._run_cli_test(["--compare_mode", "strict"])
        self._run_cli_test(["--compare_mode", "abs_mean"])

    def test_end_to_end_with_different_action(self):
        self._run_cli_test(["--action_name", "loose_equal"])


if __name__ == "__main__":
    unittest.main()
