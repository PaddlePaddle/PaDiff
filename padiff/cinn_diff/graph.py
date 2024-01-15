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

from lib2to3.pytree import Node
import graphviz
import pygraphviz as pgv
from utils import retry


@retry(max_times=1)
def get_graph(dot_path):
    return pgv.AGraph(dot_path)


def construct_graph_by_dot(dot_path, sep="\\n"):
    # print("dot_path:" + dot_path)
    graph_source = get_graph(dot_path)
    # ['color', 'label', 'style']
    all_nodes = []
    idx2nodes = {}
    ret_subgraphs = []
    subgraphs = graph_source.subgraphs()
    if not subgraphs:
        subgraphs = [graph_source]
    for subgraph in subgraphs:
        subgraph_id = subgraph.get_name().split("_")[-1]
        tmp_nodes = []
        for node in subgraph.nodes():
            name = node.attr["label"].split(sep)[0]
            idx = node.get_name()
            cls_node = Node(name=name, idx=idx, type="unknown", graph_id=subgraph_id)
            idx2nodes[idx] = cls_node
            all_nodes.append(cls_node)
            tmp_nodes.append(cls_node)
        ret_subgraphs.append(Graph(nodes=tmp_nodes, name=f"group_{subgraph_id}"))

    for edge in graph_source.edges():
        start, end = edge
        if start not in idx2nodes.keys() or end not in idx2nodes.keys():
            continue
        # 输出边
        idx2nodes[start].add_output(idx2nodes[end])
        # 输入边
        idx2nodes[end].add_input(idx2nodes[start])

    return all_nodes, ret_subgraphs


class Graph:
    def __init__(self, nodes, name) -> None:
        self.nodes = nodes
        self.name = str(name)

    def add_node(self, node):
        if isinstance(node, Node):
            if node in self.nodes:
                return
            else:
                self.nodes.append(node)
        else:
            raise ValueError(" param type must be Node")

    # just for cinn graph
    def graph_inputs(self):
        inputs = set()
        for node in self.nodes:
            if node.name == "feed":
                inputs.add(node.outputs[0].name)

        return inputs

    def inputs(self):
        inputs = []
        for node in self.nodes:
            if node.name == "feed":
                inputs.append(node.outputs[0])
            if not node.inputs:
                inputs.append(node)
            # 输入在另一个子图中，也算作当前子图的输入
            for node in node.inputs:
                if node not in self.nodes:
                    inputs.append(node)
        return inputs

    # just for cinn graph
    def graph_outputs(self):
        outputs = set()
        for node in self.nodes:
            if node.name == "fetch":
                outputs.add(node.inputs[0].name)
            if not node.outputs:
                outputs.add(node.name)
        return outputs

    def outputs(self):
        outputs = []
        for node in self.nodes:
            if node.name == "fetch":
                outputs.append(node.inputs[0])
            if not node.outputs:
                outputs.append(node)
            # 输出在另一个子图中，也算作当前子图的输出
            for node in node.outputs:
                if node not in self.nodes:
                    outputs.append(node)
        return outputs

    def find(self, cinn_var_name):
        for node in self.nodes:
            if node.name == cinn_var_name:
                return node
        return None

    def is_input(self, node):
        return node in self.inputs()

    def is_output(self, node):
        return node in self.outputs()

    def export_dot(self):
        dot = graphviz.Digraph(comment=self.name)
        for item in self.nodes:
            dot.node(item.idx, item.idx + "\n" + item.name + ":" + item.node_type)
            for next in item.outputs:
                dot.edge(item.idx, next.idx)
        return dot

    def __str__(self) -> str:
        return "graph_" + str(self.name)

    def __repr__(self) -> str:
        return "graph_" + str(self.name)


class Pass:
    def __init__(self, id, pass_name=None, before_txt=None, after_txt=None, before_dot=None, after_dot=None) -> None:
        self.pass_id = id
        self.pass_name = pass_name
        self.before_txt = before_txt
        self.after_txt = after_txt
        self.before_dot = before_dot
        self.after_dot = after_dot

    def set_pass_name(self, pass_name):
        self.pass_name = pass_name

    def set_before_txt(self, before_txt):
        self.before_txt = before_txt

    def set_after_txt(self, after_txt):
        self.after_txt = after_txt

    def set_before_dot(self, before_dot):
        self.before_dot = before_dot

    def set_after_dot(self, after_dot):
        self.after_dot = after_dot

    def __str__(self) -> str:
        return "pass_" + str(self.pass_id) + "_" + self.pass_name

    def __repr__(self) -> str:
        return "pass_" + str(self.pass_id) + "_" + self.pass_name


class Group:
    def __init__(self, group_id, all_passes, last_pass_id) -> None:
        self.group_id = group_id
        self.passes = all_passes
        self.dot_path = all_passes[last_pass_id].after_dot
        self.txt_path = all_passes[last_pass_id].after_txt
        self.all_nodes, self.subgraphs = construct_graph_by_dot(self.dot_path)
        self.fetch = None
        self.feed = None

    def export_graph(self):
        self.graph = Graph(self.all_nodes, self.__str__)
        dot = self.graph.export_dot()
        dot.render(self.__str__(), format="png", cleanup=True)

    def export_dot(self):
        dot = graphviz.Source(self.dot_path)
        dot.render(self.__str__(), format="png", cleanup=True)

    def __str__(self) -> str:
        return "fusion_group_" + str(self.group_id)

    def __repr__(self) -> str:
        return "fusion_group_" + str(self.group_id)


class Node:
    def __init__(self, name, type, idx, graph_id=None) -> None:
        self.name = name  # var name, like arg_1
        self.node_type = type if type else "unknown"
        self.idx = idx  # node name like node1
        self.inputs = []
        self.outputs = []
        self.cinn_name = ""
        self.graph_id = graph_id

    def is_op(self):
        return self.node_type == "op"

    def is_var(self):
        return self.node_type == "var"

    def is_leaf(self):
        return self.outputs == [] or self.outputs[0].name == ["fetch"]

    def is_root(self):
        return self.inputs == [] or self.inputs[0].name == ["feed"]

    def set_outputs(self, outputs):
        self.outputs = outputs

    def set_inputs(self, inputs):
        self.inputs = inputs

    def add_input(self, node):
        if isinstance(node, Node):
            if node in self.inputs:
                return
            else:
                self.inputs.append(node)
        else:
            raise ValueError("Node input must be Node")

    def add_output(self, node):
        if isinstance(node, Node):
            if node in self.outputs:
                return
            else:
                self.outputs.append(node)
        else:
            raise ValueError("Node output must be Node")

    def __str__(self) -> str:
        return self.name + "_" + self.idx + " : " + self.node_type

    def __repr__(self) -> str:
        return self.name + "_" + self.idx + " : " + self.node_type


class Cluster:
    def __init__(self, idx, graph, ops, inputs, outputs, graph_key, varmaps=None) -> None:
        self.idx = idx
        self.graph = graph
        self.ops = ops
        self.inputs = inputs
        self.outputs = outputs
        self.graph_key = graph_key
        self.varmaps = varmaps
        self.cinn_group = None

    def set_varmaps(self, varmaps: dict):
        self.varmaps = varmaps

    def set_associate_groups(self, group):
        if isinstance(group, [list, set, tuple]):
            self.associate_groups.extend(list(group))
        elif isinstance(group, str):
            self.associate_groups.append(group)
        else:
            raise ValueError(f"group must be str or sequence type, but got {type(group)}")

    def __str__(self) -> str:
        return "Cluster_" + str(self.idx)

    def __repr__(self) -> str:
        return "Cluster_" + str(self.idx)

    def print_varmaps(self):
        for paddle_name, cinn_name in self.varmaps.items():
            print({"graph_key": self.graph_key, "paddle_name": paddle_name, "cinn_name": cinn_name})
