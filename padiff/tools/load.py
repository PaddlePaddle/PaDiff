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
import os
from collections.abc import Callable

import numpy as np
import paddle
import torch

from ..configs import global_yaml_loader
from ..utils import log


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
                with open(file_path) as f:
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


def load_init_weights_from_dump(
    report_path: str,
    proxy_model: "ProxyModel",
    keys_mapping: dict | Callable | None = None,
    verbose: bool | None = False,
):
    """Load initial weights from specified report_path and use it based on
    the parameter name (or the name converted by keys_mapping), perform necessary
    transformations (such as transpose), and apply to given proxy_model.

    Args:
        report_path (str): The path to the directory where dump reports are stored,
            usually is 'padiff_dump/proxy_model.name'.
        proxy_model (ProxyModel): A model instance wrapped by ProxyModel.
        keys_mapping (Union[Callable, Dict], optional):
            A map used to convert model parameter names. Can be:
                - A dictionary providing an exact mapping from proxy_model parameter
                    names to '*.npy' file names.
                - A callable function that receives the original argument name and
                    returns the key name used for the lookup.
            If None, the parameter name of proxy_model is used directly as the
            search key for the '*.npy' file. Defaults to None.
        verbose (bool, optional): Whether to print all logs. Defaults to False.
    """
    # check files
    report_json = json.load(open(os.path.join(report_path, "report.json")))
    if not report_json.get("has_init_weights"):
        log(f"No init_weights found in {report_path}/report.json")
        return False

    weights_dir = os.path.join(report_path, "init_weights")
    loaded_weights = {}
    for file_name in os.listdir(weights_dir):
        if file_name.endswith(".npy"):
            param_name = file_name[:-4]
            file_path = os.path.join(weights_dir, file_name)
            loaded_weights[param_name] = np.load(file_path)

    if not loaded_weights:
        log(f"Not found any '*.npy' file in {weights_dir}")
        return False

    # get framwork
    tar_framwork = proxy_model.framework
    src_framwork = "torch" if tar_framwork == "paddle" else "paddle"

    # loading
    success_count = 0
    try:
        for submodel_name, submodel in proxy_model.named_submodels():
            submodel_class_name = submodel.class_name
            for sub_param_name, param in submodel.named_parameters(recursively=False):
                sub_route = submodel.route.replace(f"{proxy_model.route}.", "")
                param_name = f"{sub_route}.{sub_param_name}"

                # mapping keys
                if callable(keys_mapping):
                    param_key = keys_mapping(param_name)
                elif isinstance(keys_mapping, dict):
                    param_key = keys_mapping.get(param_name, param_name)
                else:
                    param_key = param_name

                if param_key not in loaded_weights:
                    log(f"[Info] param {param_key}({param_name}) not found, skip it.")
                    continue
                np_value = loaded_weights[param_key]

                # setting
                settings = global_yaml_loader.get_weight_settings(
                    (submodel_class_name, submodel_class_name),
                    (src_framwork, tar_framwork),
                    (sub_param_name, sub_param_name),
                )

                # check shape
                expected_shape = param.shape()
                if settings["transpose"]:
                    expected_shape = expected_shape[::-1]
                if tuple(np_value.shape) != tuple(expected_shape):
                    if verbose:
                        log(
                            f"Shape mismatch for {param_key}({param_name}): "
                            f"expected {param.shape()} but got {list(np_value.shape)}"
                        )
                    continue

                # transpose
                if settings["transpose"]:
                    if verbose:
                        log(f"Transposing: {param_key} {np_value.shape} -> {np_value.shape[::-1]}")
                    np_value = np.transpose(np_value)

                param.set_data(np_value)
                success_count += 1
        log(
            f"SUCCESS: init_weights({success_count} / {len(list(proxy_model.named_parameters()))}) loaded. "
            "If more detailed log information needed, please set the 'verbose = True'."
        )
        return True
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")

    return False
