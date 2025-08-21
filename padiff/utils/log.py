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

import os
import sys
import shutil
import logging


class Logger:
    def __init__(self):
        self._logger = None
        self._is_initialized = False

    def setup(self, log_parent_dir):
        if self._is_initialized:
            return

        self._logger = logging.getLogger("padiff")

        debug_flag = os.getenv("PADIFF_DEBUG")
        log_level_flag = os.getenv("PADIFF_LOG_LEVEL")

        if log_level_flag and log_level_flag.upper() in ("DEBUG", "INFO", "WARNING", "ERROR"):
            log_level = getattr(logging, log_level_flag.upper())
        elif debug_flag and debug_flag.strip().lower() in ("1", "true", "on"):
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        self._logger.setLevel(log_level)
        self._logger.propagate = False

        if self._logger.handlers:
            self._logger.handlers.clear()

        log_file_path = os.path.join(log_parent_dir, "padiff.log")
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_formatter = logging.Formatter("[AutoDiff] [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[AutoDiff] [%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

        self._logger.info(f"Logging initialized. Log file: {log_file_path}")
        self._is_initialized = True

    def info(self, *args):
        if self._logger is not None:
            self._logger.info(" ".join(map(str, args)))
        else:
            print(f"[AutoDiff] [INFO] {' '.join(map(str, args))}")

    def warning(self, *args):
        if self._logger is not None:
            self._logger.warning(" ".join(map(str, args)))
        else:
            print(f"[AutoDiff] [WARNING] {' '.join(map(str, args))}")

    def error(self, *args):
        if self._logger is not None:
            self._logger.error(" ".join(map(str, args)))
        else:
            print(f"[AutoDiff] [ERROR] {' '.join(map(str, args))}")

    def debug(self, *args):
        if self._logger is not None:
            self._logger.debug(" ".join(map(str, args)))
        else:
            print(f"[AutoDiff] [DEBUG] {' '.join(map(str, args))}")


logger = Logger()
log_path = os.path.join(sys.path[0], "padiff_log")


def log(*args):
    message = " ".join(map(str, args))
    local_logger = logger if logger is not None else logging.getLogger("padiff")
    local_logger.info(message)


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    logger.setup(path)


def log_file(filename, mode, info):
    os.makedirs(log_path, exist_ok=True)

    filepath = os.path.join(log_path, filename)
    with open(filepath, mode) as f:
        f.write(info)

    return filepath


"""
    other prints
"""


def print_report_info(nodes, reports, exc, stage, msg=None):

    log("FAILED !!!")

    if msg is not None:
        log("ADDITIONAL MESSAGE:")
        print(msg + "\n")
        log("DIFF DETAILS:")
    log(f"    Diff found in {stage} Stage")
    log(f"    Type of layer is: {nodes[0]['name']} vs {nodes[1]['name']}")
    log(f"    Route: {nodes[0]['route']}")
    log(f"           {nodes[1]['route']}\n")

    print(f"{type(exc).__name__}: {str(exc)} \n")

    log("Check model struct:")
    retstr = struct_info_log(reports, [node["origin_node"] for node in nodes], "report")
    print(retstr)


def tree_print(node, mark=None, prefix=[]):
    cur_str = ""
    for i, s in enumerate(prefix):
        if i == len(prefix) - 1:
            cur_str += s
        else:
            if s == " |--- ":
                cur_str += " |    "
            elif s == " +--- ":
                cur_str += "      "
            else:
                cur_str += s

    cur_str += node["name"]
    if "available" in node and node["available"] == False:
        cur_str += " (skip)"
    if os.getenv("PADIFF_PATH_LOG") == "ON":
        cur_str += "  (" + node["route"] + ")"
    if mark is node:
        cur_str += "    <---  *** HERE ***"

    ret_strs = [cur_str]
    for i, child in enumerate(node["children"]):
        pre = " |--- "
        if i == len(node["children"]) - 1:
            pre = " +--- "
        prefix.append(pre)
        retval = tree_print(child, mark, prefix)
        ret_strs.extend(retval)
        prefix.pop()

    return ret_strs


def struct_info_log(reports, nodes, file_prefix):
    file_names = []
    for idx in range(2):
        node = nodes[idx]
        report = reports[idx]
        file_name = build_file_name(report, file_prefix + "_" + report["model_name"])
        file_names.append(file_name)
        title = f"{report['model_name']}\n" + "=" * 40 + "\n"
        retval = []
        for tree in report["tree"]:
            retval.extend(tree_print(tree, mark=node, prefix=[" " * 4]))
        info = title + "\n".join(retval)
        log_file(file_name, "w", info)

    retval = f"Logs: {log_path}/{file_names[0]}\n"
    retval += f"      {log_path}/{file_names[1]}\n"
    return retval


def build_file_name(report, file_name):
    strs = report["file_path"].split("/")
    for s in reversed(strs):
        if "step_" in s:
            return file_name + "_" + s + ".log"
    return file_name
