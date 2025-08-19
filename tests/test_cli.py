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
import sys
import os
import tempfile
import shutil
import contextlib
from padiff.cli import main


class TestPadiffCLI(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="padiff_test_")
        self._original_argv = sys.argv.copy()
        self._original_cwd = os.getcwd()

    def tearDown(self):
        sys.argv = self._original_argv
        os.chdir(self._original_cwd)
        try:
            shutil.rmtree(self.test_dir)
        except OSError:
            pass

    def test_invalid_command(self):
        sys.argv = [
            "test_cli.py",
            "--pt_cmd",
            "python -c 'print(\"Hello from Torch!\")'",
            "--pd_cmd",
            "python -c 'print(\"Hello from Paddle!\")'",
        ]

        with open(os.path.join(self.test_dir, "stdout.log"), "w") as f:
            with contextlib.redirect_stdout(f):
                try:
                    exit_code = main()
                    self.assertNotEqual(exit_code, 0)
                except SystemExit as e:
                    self.assertNotEqual(e.code, 0)

    def test_missing_required_args(self):
        sys.argv = ["test_cli.py", "--pt_cmd", "python dummy.py"]

        with open(os.path.join(self.test_dir, "stderr.log"), "w") as f:
            with contextlib.redirect_stderr(f):
                with self.assertRaises(SystemExit) as cm:
                    main()
                self.assertNotEqual(cm.exception.code, 0)

        with open(os.path.join(self.test_dir, "stderr.log"), "r") as f:
            stderr_content = f.read()
            self.assertIn("the following arguments are required: --pd_cmd", stderr_content)

    def test_actual_scripts(self):
        script_dir = os.path.dirname(__file__)
        torch_script = os.path.join(script_dir, "cli", "test_cli_torch.py")
        paddle_script = os.path.join(script_dir, "cli", "test_cli_paddle.py")

        self.assertTrue(os.path.exists(torch_script), f"{torch_script} does not exist.")
        self.assertTrue(os.path.exists(paddle_script), f"{paddle_script} does not exist.")

        custom_log_dir = self.test_dir
        sys.argv = [
            "test_cli.py",
            "--pt_cmd",
            f"python {torch_script}",
            "--pd_cmd",
            f"python {paddle_script}",
            "--log_dir",
            self.test_dir,
            "--pt_model_name",
            custom_log_dir,
            "--pd_model_name",
            custom_log_dir,
        ]

        with open(os.path.join(self.test_dir, "stdout.log"), "w") as stdout_f, open(
            os.path.join(self.test_dir, "stderr.log"), "w"
        ) as stderr_f, contextlib.redirect_stdout(stdout_f), contextlib.redirect_stderr(stderr_f):
            try:
                main()
            except SystemExit as e:
                if e.code != 0:
                    self.fail(f"main() exited with code {e.code}. Check stderr.log for details.")
            except Exception as e:
                self.fail(f"main() raised an unexpected exception: {e}")

        paddle_report_path = os.path.join(custom_log_dir, "padiff_dump/model_paddle", "report.json")
        torch_report_path = os.path.join(custom_log_dir, "padiff_dump/model_torch", "report.json")
        self.assertTrue(os.path.exists(paddle_report_path), f"Paddle report not found at {paddle_report_path}")
        self.assertTrue(os.path.exists(torch_report_path), f"Torch report not found at {torch_report_path}")


if __name__ == "__main__":
    unittest.main()
