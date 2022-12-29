# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os.path as osp
import traceback

import paddle
import torch


def _is_system_package(filename):
    exclude = [
        "lib/python",
        "/usr/local",
        osp.dirname(paddle.__file__),
        osp.dirname(torch.__file__),
        osp.dirname(__file__),  # exclude padiff
    ]
    for pattern in exclude:
        if pattern in filename:
            return True
    return False


def print_frame(f, indent=8):
    indent = " " * indent
    print(
        "{} File {}: {}    {}\n{}{}{}".format(
            indent, f.filename, f.lineno, f.name, indent, indent, f.line
        )
    )


def print_frames(fs, indent=8):
    for f in fs:
        print_frame(f, indent)


def extract_frame_summary():
    """
    extract the current call stack by traceback module.
    gather the call information and put them into ReportItem to helper locate the error.

    frame_summary:
        line: line of the code
        lineno: line number of the file
        filename: file name of the stack
        name: the function name.
    """
    frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
    last_user_fs = None
    for fs in frame_summarys:
        if not _is_system_package(fs.filename):
            last_user_fs = fs
            break
    assert last_user_fs is not None, "Error happend, can't return None."
    return last_user_fs, frame_summarys
