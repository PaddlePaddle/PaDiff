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

import functools

from .log import logger


def wrap_optimizer_step(optimizer):
    if hasattr(optimizer, "_original_step"):
        return

    logger.debug("wrap_optimizer_step: wrap optimizer.step() in first call")
    original_step = optimizer.step

    @functools.wraps(original_step)
    def wrapped_step():
        original_step()

        proxy_model = getattr(optimizer, "_padiff_proxy_model", None)
        if proxy_model is not None:
            logger.debug(f"wrap_optimizer_step: Dump grads after optimizer.step()")
            proxy_model.dump_grads(proxy_model.dump_path)

    optimizer.step = wrapped_step
    optimizer._original_step = original_step
