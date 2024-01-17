# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np


class Comparator:
    def __init__(self, rtol=0, atol=0) -> None:
        self.cluster_ret = {}
        self.graph_ret = {}
        self.record = []
        self.rtol = rtol
        self.atol = atol

    @classmethod
    def load_var(self, path):
        return paddle.Tensor(paddle.core.load_dense_tensor(path))

    def allclose(self, base_path, compare_path):
        base_var = self.load_var(base_path)
        compare_var = self.load_var(compare_path)
        ret = np.allclose(base_var, compare_var, rtol=self.rtol, atol=self.atol)
        return ret

    def assert_allclose(self, base_path, compare_path):
        base_var = self.load_var(base_path)
        compare_var = self.load_var(compare_path)
        ret = np.testing.assert_allclose(base_var, compare_var, rtol=self.rtol, atol=self.atol)
        return ret

    def record_diff(self, diff, type):
        diff = {
            "type": type,
            "event": diff,
        }
        self.record.append(diff)

    def record_input_diff(self, cluster_idx, input):
        self.record_diff({"cluster_idx": cluster_idx, "cluster_input_diff_name": input}, "cluster_input_diff")

    def record_output_diff(self, cluster_idx, output, output_cinn_name):
        self.record_diff(
            {
                "cluster_idx": cluster_idx,
                "cluster_output_diff_paddle_name": output,
                "cluster_output_diff_cinn_name": output_cinn_name,
            },
            "cluster_output_diff",
        )

    def record_group_output_diff(self, diff_ret):
        self.record_diff(
            {
                "cluster_idx": diff_ret["cluster"],
                "cluster_output_diff_paddle_name": diff_ret["output"],
                "group_idx": diff_ret["group"].group_id,
                "group_output_diff_cinn_name": diff_ret["output_cinn_var"],
                "group_graphviz_path": diff_ret["group"].dot_path,
                "group_test_py_code_path": diff_ret["group"].txt_path,
                "group_diff_subgraph_id": diff_ret["subgraph_id"],
            },
            "group_output_diff",
        )


if __name__ == "__main__":
    cmp = Comparator()
    base = cmp.load_var("/root/dev/PaddleClas/base/saved_tensors/batch_norm_grad-input-batch_norm_0.tmp_3@GRAD")
    cinn = cmp.load_var("/root/dev/PaddleClas/cinn/saved_tensors/batch_norm_grad-input-batch_norm_0.tmp_3@GRAD")
    np.testing.assert_allclose(base, cinn, rtol=0, atol=0)
