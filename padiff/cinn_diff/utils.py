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

from contextlib import contextmanager
from functools import wraps
import os, sys


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_stdin = sys.stdin
        sys.stdout = devnull
        sys.stderr = devnull
        sys.stdin = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.stdin = old_stdin


def retry(max_times=1):
    def retry_decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            retry_times = 0
            while retry_times <= max_times:
                try:
                    ret = func(*args, **kwargs)
                    return ret
                except Exception as e:
                    retry_times += 1

        return inner

    return retry_decorator


# \033 [显示方式;字体色;背景色m ...... [\033[0m]
# 显示方式: 0（默认值）、1（高亮）、22（非粗体）、4（下划线）、24（非下划线）、 5（闪烁）、25（非闪烁）、7（反显）、27（非反显）
# 前景色: 30（黑色）、31（红色）、32（绿色）、 33（黄色）、34（蓝色）、35（洋 红）、36（青色）、37（白色）
# 背景色: 40（黑色）、41（红色）、42（绿色）、 43（黄色）、44（蓝色）、45（洋 红）、46（青色）、47（白色）


class console:
    def __init__(self) -> None:
        pass

    @classmethod
    def red(self, str):
        return

    @classmethod
    def info(self, str):
        return

    @classmethod
    def error(self, str):
        return

    @classmethod
    def warning(self, str):
        return
