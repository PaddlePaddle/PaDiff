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

from .read_file import read_all
from .compare_utils import Comparator
from .logs import logger

# TODO(GGBond8488): Needs optimization in the future
def back_track_group(base, compare, cluster, cmp, graph, node):
    inputs = graph.inputs()
    all_inputs_equal = True
    paddle_output_name = ""
    cur_cluster_cinn2paddle = {v: k for k, v in cluster.varmaps.items()}
    for input in inputs:
        tmp = input
        paddle_name = cur_cluster_cinn2paddle.get(tmp.name, "")
        if not paddle_name:
            logger.info(f"can't find {node.name}'s paddle name")
            diff_ret = {
                "cluster": cluster.idx,
                "group": cluster.cinn_group,
                "output": paddle_output_name,
                "output_cinn_var": node.name,
                "subgraph_id": node.graph_id if node else None,
            }
            return diff_ret
        paddle_output_name = paddle_name
        base_var_path = base.all_vars_paths[paddle_name]
        compare_var_path = compare.all_vars_paths[paddle_name]
        ret = cmp.allclose(base_var_path, compare_var_path)
        if not ret:
            all_inputs_equal = False
            group = cluster.cinn_group
            for graph in group.subgraphs:
                node = graph.find(tmp.name)
                if node and not graph.is_input(node):
                    return back_track_group(base, compare, cluster, cmp, graph, node)
    if all_inputs_equal:
        diff_ret = {
            "cluster": cluster.idx,
            "group": cluster.cinn_group,
            "output": paddle_output_name,
            "output_cinn_var": node.name,
            "subgraph_id": node.graph_id if node else None,
        }
        return diff_ret


def auto_diff(base_path, compare_path, rtol=1e-6, atol=1e-6):
    base = read_all(base_path)
    compare = read_all(compare_path)
    cmp = Comparator(rtol=rtol, atol=atol)

    # step1: Confirm whether the input and output of the cluster are aligned
    for cluster in compare.all_clusters:
        input_equals_flag = True
        output_equals_flag = True
        for input in cluster.inputs:
            base_var_path = base.all_vars_paths[input]
            compare_var_path = compare.all_vars_paths[input]
            ret = cmp.allclose(base_var_path, compare_var_path)
            if not ret:
                input_equals_flag = False
                cmp.record_input_diff(cluster.idx, input)
                continue

        if input_equals_flag:
            # step2: Find the misaligned output of the cluster
            for output in cluster.outputs:
                base_var_path = base.all_vars_paths[output]
                compare_var_path = compare.all_vars_paths[output]
                ret = cmp.allclose(base_var_path, compare_var_path)
                if not ret:
                    output_equals_flag = False
                    # step3: Find the group corresponding to the misaligned output
                    output_cinn_var = cluster.varmaps.get(output, "")
                    if not output_cinn_var:
                        logger.info("can't find var " + output + " corresponding cinn var name")
                    else:
                        find_diff_group_flag = False
                        # step4 : Starting from the misaligned output, find the group where the output misalignment
                        # occurs for the first time (the input can be aligned, but the output cannot be aligned)
                        group = cluster.cinn_group
                        for graph in group.subgraphs:
                            node = graph.find(output_cinn_var)
                            if node and not graph.is_input(node):
                                # Find the first misaligned output and start backtracking
                                diff_ret = back_track_group(base, compare, cluster, cmp, graph, node)
                                if diff_ret:  # Input can be aligned, but output cannot be aligned
                                    diff_ret["output"] = output
                                    cmp.record_group_output_diff(diff_ret)
                                find_diff_group_flag = True
                                break

                        if not find_diff_group_flag:
                            cmp.record_output_diff(cluster.idx, output, cluster.varmaps.get(output, ""))
                            logger.info("can't find diff group in cluster_" + cluster.idx + " but diff exsits")

            if output_equals_flag:
                logger.info("cluster_" + cluster.idx + " has no diff")

    for diff in cmp.record:
        logger.info(diff)
    return cmp.record


if __name__ == "__main__":
    # test code, you can simple use cinn_diff.auto_diff this way
    base_path = "/root/dev/PaddleClas/base"
    compare_path = "/root/dev/PaddleClas/cinn"
    auto_diff(base_path, compare_path, atol=0, rtol=0)
