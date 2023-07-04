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

import sys
import unittest
import paddle
import paddle.distributed as dist
from padiff import *

from ppfleetx.utils import config
from ppfleetx.models import build_module
from ppfleetx.distributed.apis import env


class TestGPT3WhiteBlackList(unittest.TestCase):
    def test_check_success(self):
        sys.argv = ["test_gpt3_white_black_list.py", "-c", "./pretrain_gpt_345M_single_card.yaml"]
        args = config.parse_args()
        cfg = config.get_config(args.config, show=False)
        paddle.set_device(cfg["Global"]["device"])
        if dist.get_world_size() > 1:
            env.init_dist_env(cfg)
        env.set_seed(cfg.Global.seed)
        module = build_module(cfg)

        layer = module.model
        layer.eval()
        white_list_layers = [layer.gpt.embeddings.word_embeddings, layer.gpt.embeddings.position_embeddings]
        black_list_layers = [layer.gpt.embeddings.word_embeddings]
        layer = create_model(layer, "layer")
        layer.update_white_list(white_list_layers, "all")
        layer.update_black_list(black_list_layers, "all")

        model = module.model
        model.eval()
        white_list_layers = [model.gpt.embeddings.word_embeddings, model.gpt.embeddings.position_embeddings]
        model = create_model(model, "model")
        model.update_white_list(white_list_layers, "all")

        assign_weight(model, layer)

        inp = paddle.randint(0, 50304, (1, 512))
        inp = ({"input_ids": inp}, {"input_ids": inp})
        assert auto_diff(model, layer, inp, atol=1e-4) is True, "Failed. expected success."
        assert check_report(layer.dump_path + f"/auto_diff", model.dump_path + f"/auto_diff") == True
        assert check_params(layer.dump_path + f"/auto_diff", model.dump_path + f"/auto_diff") == True


if __name__ == "__main__":
    unittest.main()
