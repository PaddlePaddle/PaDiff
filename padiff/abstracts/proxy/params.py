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

import paddle
import torch
from ...utils import get_numpy_from_tensor


class ProxyParam:
    def __init__(self, param, param_type):
        self.param = param
        self.param_type = param_type

    @staticmethod
    def create_from(param):
        if isinstance(param, ProxyParam):
            return param
        elif isinstance(param, paddle.framework.io.EagerParamBase):
            return PaddleParam(param)
        elif isinstance(param, torch.nn.parameter.Parameter):
            return TorchParam(param)
        elif isinstance(param, (torch.Tensor, paddle.Tensor)):
            return ProxyTensor(param)
        else:
            logger.error(f"Can not create ProxyParam from {type(param)}")

    def numpy(self):
        raise NotImplementedError()

    def set_data(self, np_value):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()

    def grad(self):
        raise NotImplementedError()

    def main_grad(self):
        raise NotImplementedError()


class PaddleParam(ProxyParam):
    def __init__(self, param):
        super().__init__(param, "paddle")

    def numpy(self):
        return get_numpy_from_tensor(self.param)

    def set_data(self, np_value):
        paddle.assign(paddle.to_tensor(np_value, dtype=self.param.dtype), self.param)

    def shape(self):
        return list(self.param.shape)

    def grad(self):
        if self.param.grad is not None:
            return get_numpy_from_tensor(self.param.grad)
        else:
            return None

    def main_grad(self):
        if hasattr(self.param, "main_grad") and self.param.main_grad is not None:
            assert self.param.grad is None
            return get_numpy_from_tensor(self.param.main_grad)

        else:
            return None


class TorchParam(ProxyParam):
    def __init__(self, param):
        super().__init__(param, "torch")

    def numpy(self):
        return get_numpy_from_tensor(self.param.data)

    def set_data(self, np_value):
        self.param.data = torch.as_tensor(np_value).type(self.param.dtype).to(self.param.device)

    def shape(self):
        return list(self.param.shape)

    def grad(self):
        if self.param.grad is not None:
            return get_numpy_from_tensor(self.param.grad.data)
        else:
            return None

    def main_grad(self):
        return None


class ProxyTensor(ProxyParam):
    def __init__(self, param):
        super().__init__(param, "tensor")

    def numpy(self):
        return get_numpy_from_tensor(self.param)

    def grad(self):
        return None

    def main_grad(self):
        return None
