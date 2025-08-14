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

import json
import numpy as np
import os
import paddle
import torch


def load_first_input_from_dump(report_path, framework="paddle"):
    report_json = json.load(open(os.path.join(report_path, "report.json")))
    if not report_json.get("has_first_input"):
        return None

    input_dir = os.path.join(report_path, "first_input")
    all_files = sorted(
        [f for f in os.listdir(input_dir) if f.startswith("input_")], key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if not all_files:
        raise FileNotFoundError(f"Not found any 'input_*' file in {input_dir}. Please check the path.")

    reconstructed_inputs = []
    for file_name in all_files:
        file_path = os.path.join(input_dir, file_name)

        if file_name.endswith(".npy"):
            numpy_array = np.load(file_path)
            if framework == "paddle":
                tensor = paddle.to_tensor(numpy_array)
                tensor.stop_gradient = False
            elif framework == "torch":
                tensor = torch.tensor(numpy_array)
                tensor.requires_grad_ = True
            reconstructed_inputs.append(tensor)

        elif file_name.endswith(".json"):
            try:
                with open(file_path, "r") as f:
                    meta_data = json.load(f)
                data_type = meta_data["type"]
                data_value = meta_data["data"]

                if data_type == "dict":
                    reconstructed_dict = {}
                    for key, (item_type, item_value) in data_value.items():
                        if item_type == "Tensor":
                            raise RuntimeError(
                                f"Including Tensor types in meta JSON is not supported. Please check the serialize logic."
                            )
                        else:
                            reconstructed_dict[key] = item_value
                    reconstructed_inputs.append(reconstructed_dict)
                elif data_type in ["list", "tuple"]:
                    reconstructed_list = []
                    for item_type, item_value in data_value:
                        if item_type == "Tensor":
                            raise RuntimeError(f"Including Tensor types in meta JSON is not supported.")
                        else:
                            reconstructed_list.append(item_value)
                    if data_type == "tuple":
                        reconstructed_list = tuple(reconstructed_list)
                    reconstructed_inputs.append(reconstructed_list)
                else:
                    reconstructed_inputs.append(data_value)
            except Exception as e:
                print(f"[Error] Error loading metadata file {file_name}: {e}")
                raise

        else:
            print(f"[Warning] Ignore unknown files: {file_name}")
            continue
    return reconstructed_inputs
