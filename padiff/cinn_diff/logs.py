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
import sys
import logging
from logging import handlers


class Logger(object):
    def __init__(
        self, filename, level=logging.INFO, when="D", backCount=3, fmt="%(asctime)s" "-%(levelname)s: %(message)s"
    ):

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(level)
        self.format_str = logging.Formatter(fmt)

        time_file_handler = handlers.TimedRotatingFileHandler(
            filename=filename,
            when=when,
            interval=1,
            backupCount=backCount,
            encoding="utf-8",
        )

        time_file_handler.setFormatter(self.format_str)
        self.logger.addHandler(time_file_handler)

    def loggerImp(self):
        return self.logger


logger = Logger(os.path.join(sys.path[0], "cinn_diff.log")).loggerImp()


def log_init(file_name):
    global logger
    if file_name:
        logger = Logger(os.path.join(sys.path[0], file_name)).loggerImp()
