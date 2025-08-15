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


log_path = os.path.join(sys.path[0], "padiff_log")
__reset_log_dir__ = False  # reset log_path only once


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def log_file(filename, mode, info):
    global __reset_log_dir__
    if not __reset_log_dir__:
        reset_dir(log_path)
        __reset_log_dir__ = True

    filepath = os.path.join(log_path, filename)
    with open(filepath, mode) as f:
        f.write(info)

    return filepath


def log(*args):
    print("[AutoDiff]", *args)


class Counter:
    def __init__(self):
        self.clear()

    def clear(self):
        self.id = 0

    def get_id(self):
        ret = self.id
        self.id += 1
        return ret


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
