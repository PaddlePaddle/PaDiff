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

import yaml
import os
import json


"""
    yaml_loader
"""


class yaml_loader:
    def __init__(self):
        yaml_path = os.path.join(os.path.dirname(__file__), "configs", "assign_weight.yaml")
        with open(yaml_path, "r") as yaml_file:
            self._assign_yaml = yaml.safe_load(yaml_file)
        self._options = {}

    def get_weight_settings(self, paddle_layer, torch_module, param_name):
        assign_config = self._assign_yaml.get(paddle_layer.__class__.__name__, None)
        settings = {
            "atol": self.options["atol"],
            "rtol": self.options["rtol"],
            "transpose": False,
            "compare_mode": self.options["compare_mode"],
        }

        if assign_config is not None:
            assert (
                torch_module.__class__.__name__ in assign_config["torch"]
            ), "Not correspond, paddle layer {}  vs torch module {}. check your __init__ to make sure every sublayer is corresponded, or view the model struct reports in diff_log.".format(
                paddle_layer.__class__.__name__, torch_module.__class__.__name__
            )

        if (
            assign_config is None
            or assign_config.get("param", None) == None
            or param_name not in assign_config["param"]
        ):
            pass
        else:
            if assign_config["param"][param_name] == "transpose":
                settings["transpose"] = True

        return settings

    @property
    def assign_yaml(self):
        return self._assign_yaml

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, val):
        assert isinstance(val, dict)
        self._options.update(val)


global_yaml_loader = yaml_loader()


"""
    json_loader
"""


class json_loader:
    def __init__(self):
        self.TORCH_MODULE = [
            "torch.nn.functional",
            "torch",
            "torch.linalg",
            "torch.fft",
        ]
        self.PADDLE_MODULE = [
            "paddle.nn.functional",
            "paddle",
            "paddle.linalg",
            "paddle.fft",
            "paddle.incubate.sparse",
            "paddle.signal",
        ]

        json_path = os.path.join(os.path.dirname(__file__), "configs", "api_mapping.json")
        with open(json_path, "r") as file:
            self.api_mapping = json.load(file)

        self.torch_apis = {}
        self.paddle_apis = {}

        for k, v in self.api_mapping.items():
            if "paddle_api" not in v.keys():
                continue

            torch_fullname = k
            torch_module = torch_fullname.rpartition(".")[0]
            torch_api = torch_fullname.rpartition(".")[2]

            paddle_fullname = v["paddle_api"]
            paddle_module = paddle_fullname.rpartition(".")[0]
            paddle_api = paddle_fullname.rpartition(".")[2]

            if torch_module not in self.TORCH_MODULE or paddle_module not in self.PADDLE_MODULE:
                continue

            if torch_module not in self.torch_apis.keys():
                self.torch_apis[torch_module] = [torch_api]
            else:
                self.torch_apis[torch_module].append(torch_api)

            if paddle_module not in self.paddle_apis.keys():
                self.paddle_apis[paddle_module] = [paddle_api]
            else:
                self.paddle_apis[paddle_module].append(paddle_api)

        # Deprecated
        self.torch_apis["torch.nn.functional"].remove("sigmoid")


global_json_loader = json_loader()
