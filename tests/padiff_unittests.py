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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXCLUDE_FILES = {"test_api_to_Layer"}


def discover_and_run_filtered_tests():
    loader = unittest.TestLoader()
    all_tests = loader.discover(start_dir=".", pattern="test_*.py")

    filtered_suite = unittest.TestSuite()

    def add_filtered_tests(test):
        if isinstance(test, unittest.TestCase):
            module = sys.modules.get(test.__class__.__module__)
            if module and hasattr(module, "__file__"):
                filename = os.path.basename(module.__file__)
                filename_without_ext = os.path.splitext(filename)[0]
                if filename_without_ext not in EXCLUDE_FILES:
                    filtered_suite.addTest(test)
                else:
                    print(f"Excluding test from file: {filename}")
            else:
                filtered_suite.addTest(test)
        elif isinstance(test, unittest.TestSuite):
            for subtest in test:
                add_filtered_tests(subtest)

    for suite in all_tests:
        add_filtered_tests(suite)

    if filtered_suite.countTestCases() == 0:
        print("No tests to run after filtering.")
        return False

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(filtered_suite)

    os.system("rm -rf ./tests/padiff_dump ./tests/padiff_log")

    return result.wasSuccessful()


def main():
    try:
        success = discover_and_run_filtered_tests()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred during test execution: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
