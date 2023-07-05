# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import subprocess


for root, dirs, files in os.walk("./"):
    for fname in files:
        if fname.endswith(".py") and fname.startswith("test_"):
            fpath = root + "/" + fname
            (status, output) = subprocess.getstatusoutput("python " + fpath)
            if status != 0:
                err_info = f"*** ===================== {fpath} ========================= ***\n"
                err_info += f"{output}\n"
                print(f"Failed on unittest {fname} with error message \n {err_info}.", end="\n", flush=True)
            else:
                print(f"Succeed on unittest {fname}.", end="\n", flush=True)
            os.system("rm -rf ./tests/padiff_dump ./tests/padiff_log")
