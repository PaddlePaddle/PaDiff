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

import os

from .graph import Graph, Node, Cluster, Group, Pass, construct_graph_by_dot

import collections

INPUTS_NAME = "cluster_inputs.txt"
OUTPUTS_NAME = "cluster_outputs.txt"
OPS_NAME = "cluster_ops.txt"
GRAPH_NAME = "subgraph.txt"
PADDLE2CINN_VARMAP = "paddle2cinn_varmap.txt"
GRAPH_COMPLATION_KEY = "graph_compilation_key.txt"
VARMAPS_KEY_NAME = "graph_compilation_key"


def read_varmaps(varmaps_file):
    graph2varmaps = {}
    varmaps_file = os.path.join(varmaps_file, PADDLE2CINN_VARMAP)
    with open(varmaps_file) as f:
        lines = f.readlines()
        cur_graph_key = None
        for line in lines:
            var_map = line.strip("\n").split(":")
            if var_map[0] == VARMAPS_KEY_NAME:
                cur_graph_key = var_map[1]
                graph2varmaps[cur_graph_key] = {}
                continue
            if not cur_graph_key:
                return
            graph2varmaps[cur_graph_key][var_map[0]] = var_map[1]

    return graph2varmaps


def read_tensors(tensors_path):
    tensors_map = {}
    assert os.path.isdir(tensors_path)
    for file in os.listdir(tensors_path):
        # ['matmul_v2_grad', 'input', 'scale_0.tmp_0']
        var_info = file.split("-")
        # bind var_name to var tensor file
        if "share_buffer" in file:
            continue
        tensors_map[var_info[-1]] = os.path.join(tensors_path, file)
    return tensors_map


def read_graph(graph_file, idx):
    assert os.path.isfile(graph_file)
    nodes = {}
    edges = {}

    def record_nodes_and_edges(line, type, nodes, edges):
        if type == "nodes":
            node = line.split(" : ")
            node_id, node_desc = node[0], node[1]
            name, node_type = node_desc[1:-1].split(", ")
            if name in ["feed", "fetch"]:
                return
            nodes[node_id] = Node(name, node_type, node_id)
        elif type == "edges":
            edge = line.split(" -> ")
            cur, next = edge[0], edge[1]
            edges[cur] = next
        else:
            raise ValueError(type + "not support")

    type = "nodes"
    with open(graph_file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            if line.startswith("nodes:"):  # start to record node
                type = "nodes"
                continue
            if line.startswith("edges:"):  # start to record edge
                type = "edges"
                continue
            record_nodes_and_edges(line, type, nodes, edges)

    def construct_graph(nodes, edges):
        for k, v in edges.items():
            if k not in nodes or v not in nodes:
                continue
            nodes[k].add_output(nodes[v])
            nodes[v].add_input(nodes[k])
        graph = Graph(nodes.values(), idx)
        return graph

    graph = construct_graph(nodes, edges)
    return graph


def read_strings(string_file):
    assert os.path.isfile(string_file)
    with open(string_file) as f:
        line = f.readline()
        line = line[1:-1]
        rets = line.split(", ")[:-1]
    return rets


def read_string(string_file):
    assert os.path.isfile(string_file)
    with open(string_file) as f:
        line = f.readline()
        rets = line
    return rets


def read_cluster(path, idx):
    assert os.path.isdir(path), f"{path} must be dir"
    inputs = read_strings(os.path.join(path, INPUTS_NAME))
    outputs = read_strings(os.path.join(path, OUTPUTS_NAME))
    ops = read_strings(os.path.join(path, OPS_NAME))
    graph = read_graph(os.path.join(path, GRAPH_NAME), idx)
    graph_key = read_string(os.path.join(path, GRAPH_COMPLATION_KEY))
    return Cluster(idx, graph, ops, inputs, outputs, graph_key)


def read_cinn_pass(path):
    all_groups = {}

    def read_graphviz_dot(path):
        passes = os.listdir(path)
        idx = path.split("_")[-1]
        # print("group idx: " + str(idx))
        all_passes = {}
        for pass_path in passes:
            # print("pass_path: " + pass_path)
            # print(pass_path.split("_"))
            pass_idx = int(pass_path.split("_")[1])
            # print("pass_idx: " + str(pass_idx))
            if pass_idx not in all_passes:
                all_passes[pass_idx] = Pass(pass_idx)
            pass_name = pass_path.split("_")[2]
            all_passes[pass_idx].set_pass_name(pass_name)
            type = pass_path.split("_")[3]  # after.txt
            record_path = os.path.join(path, pass_path)
            if type == "after.txt":
                all_passes[pass_idx].set_after_txt(record_path)
            elif type == "before.txt":
                all_passes[pass_idx].set_before_txt(record_path)
            elif type == "after.dot":
                all_passes[pass_idx].set_after_dot(record_path)
            elif type == "before.dot":
                all_passes[pass_idx].set_before_dot(record_path)
            else:
                raise ValueError(type + "not support")
        max_pass_id = max(all_passes.keys())
        # print("lass_pass_id: " + str(max_pass_id))
        group_cc = Group(idx, all_passes, max_pass_id)
        all_groups[idx] = group_cc

    file_names = os.listdir(path)
    for file_name in file_names:
        read_graphviz_dot(os.path.join(path, file_name))

    return all_groups


def read_cinn_graph(path):
    all_cinn_graphs = {}

    def read_cinn_graph_dot(path, idx):
        graph_path = os.listdir(path)[0]
        file_path = os.path.join(path, graph_path)
        nodes, _ = construct_graph_by_dot(file_path, sep="\n")
        graph = Graph(nodes=nodes, name=str("cinn_graph_" + idx))
        return graph

    file_names = os.listdir(path)
    for file_name in file_names:
        idx = file_name.split("_")[-1]
        graph = read_cinn_graph_dot(os.path.join(path, file_name), idx)
        all_cinn_graphs[idx] = graph
    return all_cinn_graphs


def set_node_cinn_name(all_clusters):
    for cluster in all_clusters:
        nodes = cluster.graph.nodes
        for node in nodes:
            if node.is_var():
                node.cinn_name = cluster.varmaps.get(node.name, "")


def set_cluster_varmaps(clusters, varmaps):
    for cluster in clusters:
        tmp_varmaps = varmaps.get(cluster.graph_key, "")
        if not tmp_varmaps:
            raise KeyError(f"can't find graph key {cluster.graph_key} in graph2varmaps")
        cluster.set_varmaps(tmp_varmaps)


def set_clusters_group(clusters, groups, cinn_graphs):
    for cluster in clusters:
        inputs = cluster.inputs
        outputs = cluster.outputs
        for idx, graph in cinn_graphs.items():
            graph_inputs = graph.graph_inputs()
            graph_outputs = graph.graph_outputs()
            if not graph_inputs and not graph_outputs:
                raise ValueError(f"{graph} does not have inputs or outputs")
            # print(graph_inputs)
            # print(inputs)
            if not set(inputs).difference(graph_inputs) and not set(outputs).difference(graph_outputs):
                print(f"group_{idx} belongs to Cluster_{cluster.idx}")
                cluster.cinn_group = groups[idx]


def read_all(root_path="", type="cinn"):

    assert root_path, f"{root_path} can't be None"
    all_clusters = []
    # paddle2cinn_varmaps
    graph2varmaps = {}
    all_vars_paths = {}
    all_cinn_groups = {}
    all_cinn_graphs = {}
    # exsist some bug
    cinn2paddle_varmaps = {}

    allinone = collections.namedtuple(
        "allinone", ["all_clusters", "all_varmaps", "all_vars_paths", "all_cinn_groups", "cinn2paddle"]
    )

    all_paths = os.listdir(root_path)
    for path in all_paths:
        file_path = os.path.join(root_path, path)
        assert os.path.isfile(file_path) or os.path.isdir(file_path), f"{file_path} must be path or dir"
        if type == "cinn" and path.startswith("cluster"):
            idx = path.split("_")[-1]
            all_clusters.append(read_cluster(file_path, idx))

        if type == "cinn" and path == "paddle2cinn_varmap":
            graph2varmaps.update(read_varmaps(file_path))

        if path == "saved_tensors":
            all_vars_paths.update(read_tensors(file_path))

        if type == "cinn" and path == "cinn_pass":
            all_cinn_groups = read_cinn_pass(file_path)

        if type == "cinn" and path == "cinn_graph":
            all_cinn_graphs = read_cinn_graph(file_path)

    set_cluster_varmaps(all_clusters, graph2varmaps)

    if type == "cinn":
        set_node_cinn_name(all_clusters)
        set_clusters_group(all_clusters, all_cinn_groups, all_cinn_graphs)

    return allinone(all_clusters, graph2varmaps, all_vars_paths, all_cinn_groups, cinn2paddle_varmaps)


if __name__ == "__main__":
    read_all()
