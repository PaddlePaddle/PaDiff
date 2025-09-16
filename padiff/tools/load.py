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
from ..utils import logger


def load_first_input_from_dump(report_path, tar_framework):
    report_json = json.load(open(os.path.join(report_path, "report.json")))
    if not report_json.get("has_first_input"):
        logger.error(f"No first_input found in {report_path}/report.json")
        return None

    input_dir = os.path.join(report_path, "first_input")
    meta_file = os.path.join(input_dir, "meta.json")

    if not os.path.exists(meta_file):
        logger.warning(f"'meta.json' not found in {input_dir}. Please check the path.")
        return None

    try:
        with open(meta_file, "r") as f:
            meta_info = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load 'meta.json': {e}")
        return None

    args = []
    kwargs = {}

    NATIVE_TYPES = (int, float, str, bool, type(None))

    for item in meta_info:
        file_path = os.path.join(input_dir, item["path"])
        key = item.get("key")

        try:
            if item["type"] == "Tensor":
                numpy_array = np.load(file_path)
                if tar_framework == "paddle":
                    tensor = paddle.to_tensor(numpy_array)
                    tensor.stop_gradient = False
                elif tar_framework == "torch":
                    tensor = torch.tensor(numpy_array)
                    tensor.requires_grad_(True)
                else:
                    raise ValueError(f"Unsupported framework: {tar_framework}")
                value = tensor

            else:
                with open(file_path, "r") as f:
                    full_item = json.load(f)

                if item["type"] == "dict":
                    value = {k: v for k, v in full_item["data"].items()}
                elif item["type"] == "list":
                    value = [v for v in full_item["data"]]
                elif item["type"] == "tuple":
                    value = tuple(v for v in full_item["data"])
                elif item["type"] == "int":
                    value = int(full_item["data"])
                elif item["type"] == "float":
                    value = float(full_item["data"])
                elif item["type"] == "bool":
                    value = full_item["data"].lower() == "true"
                elif item["type"] == "NoneType":
                    value = None
                elif item["type"] == "str":
                    value = full_item["data"]
                else:
                    logger.warning(f"Skipping unsupported input type '{item['type']}' for input(key={key}).")
                    continue

            if key is None:
                args.append(value)
            else:
                kwargs[key] = value

        except Exception as e:
            logger.error(f"Error loading input(key={key}) in {file_path}: {e}")
            raise

    if not args and not kwargs:
        logger.warning("No valid inputs were loaded from the dump.")
        return None

    return (args, kwargs)


def load_init_weights_from_dump(
    report_path: str,
    proxy_model: "ProxyModel",
    keys_mapping: dict | Callable | None = None,
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
    """
    # check files
    report_json = json.load(open(os.path.join(report_path, "report.json")))
    if not report_json.get("has_init_weights"):
        logger.error(f"No init_weights found in {report_path}/report.json")
        return False

    weights_dir = os.path.join(report_path, "init_weights")
    loaded_weights = {}
    for file_name in os.listdir(weights_dir):
        if file_name.endswith(".npy"):
            param_name = file_name[:-4]
            file_path = os.path.join(weights_dir, file_name)
            loaded_weights[param_name] = np.load(file_path)

    if not loaded_weights:
        logger.error(f"Not found any '*.npy' file in {weights_dir}")
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
                    logger.debug(f"param {param_key}({param_name}) not found, skip it.")
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
                    logger.debug(
                        f"Shape mismatch for {param_key}({param_name}): "
                        f"expected {param.shape()} but got {list(np_value.shape)}"
                    )
                    if np_value.size == param.param.size:
                        settings["transpose"] = not settings["transpose"]
                    else:
                        continue

                # transpose
                if settings["transpose"]:
                    logger.debug(f"Transposing: {param_key} {np_value.shape} -> {np_value.shape[::-1]}")
                    np_value = np.transpose(np_value)

                param.set_data(np_value)
                success_count += 1
        all_count = len(list(proxy_model.named_parameters()))
        if success_count == all_count:
            logger.info(f"Loading success: all {all_count} init_weights loaded. ")
        else:
            logger.warning(f"Loading might FAILED! {all_count} init_weights in total but only {success_count} loaded!")
        return True
    except Exception as e:
        logger.error(f"{type(e).__name__}: {e}")

    return False
