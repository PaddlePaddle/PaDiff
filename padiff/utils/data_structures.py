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


def clone_dict_tree(root):
    new_root = {}
    new_root.update(root)
    new_root["origin_node"] = root
    new_root["reordered"] = False
    new_root["children"] = []
    for child in root["children"]:
        new_root["children"].append(clone_dict_tree(child))
    return new_root


def traversal_node(node, node_list=[]):
    if node["available"]:
        node_list.append(node)
    for child in node["children"]:
        traversal_node(child, node_list)
    return node_list


# reorder second tree based on the first one
def reorder_and_match_sublayers(nodes, reports):
    if len(nodes[0]["children"]) == 0 and len(nodes[1]["children"]) == 0:
        return

    # split children to 3 parts
    base_apis = list(filter(lambda x: x["type"] == "api", nodes[0]["children"]))
    base_opaque_layers = list(filter(lambda x: x["type"] == "in map", nodes[0]["children"]))
    base_layers = list(filter(lambda x: x["type"] == "net", nodes[0]["children"]))

    raw_apis = list(filter(lambda x: x["type"] == "api", nodes[1]["children"]))
    raw_opaque_layers = list(filter(lambda x: x["type"] == "in map", nodes[1]["children"]))
    raw_layers = list(filter(lambda x: x["type"] == "net", nodes[1]["children"]))

    try:
        assert len(base_apis) == len(raw_apis), "number of api is different"
        assert len(base_opaque_layers) == len(raw_opaque_layers), "number of opaque_layers is different"
        assert len(base_layers) == len(raw_layers), "number of normal layer is different"

        # reset orders
        reorder_api(base_apis, raw_apis)
        layer_map = dict(zip(reports[0]["layer_map"], reports[1]["layer_map"]))
        reorder_opaque_layers(base_opaque_layers, raw_opaque_layers, layer_map)
        reorder_normal_layers(base_layers, raw_layers)

        # for every child in nodes[0], find correspond child in nodes[1]
        new_children = []
        for child in nodes[0]["children"]:
            if child["type"] == "api":
                new_children.append(raw_apis[0])
                raw_apis.pop(0)
            elif child["type"] == "in map":
                new_children.append(raw_opaque_layers[0])
                raw_opaque_layers.pop(0)
            elif child["type"] == "net":
                new_children.append(raw_layers[0])
                raw_layers.pop(0)
            else:
                raise RuntimeError("Invalid node type")

        nodes[1]["children"] = new_children
        nodes[1]["reordered"] = True

    except Exception as e:
        raise e


def reorder_api(apis, base):
    """
    reorder apis based on base
    TODO(wuzhafnei): need better match logic there
    Temporarily, just keep in order
    """
    return


def swap(seq, l, r):
    temp = seq[l]
    seq[l] = seq[r]
    seq[r] = temp
    return


def reorder_opaque_layers(base_nodes, raw_nodes, layer_map):
    for idx, base_node in enumerate(base_nodes):
        # an api layer can not have in_layer_map mark, so node.net is save
        correspond_route = layer_map["route"][base_node["route"]]
        correspond_node = next(node for node in raw_nodes if node["route"] == correspond_route)
        item_idx = raw_nodes.index(correspond_node)
        if item_idx == idx:
            continue
        elif item_idx > idx:
            swap(raw_nodes, item_idx, idx)
        else:
            raise RuntimeError("Duplicate key or values, check your LayerMap")

    return


def reorder_normal_layers(base_nodes, raw_nodes):
    # we suppose that: corresponding layers have same net_id
    bucket = {}
    for node in raw_nodes:
        key = node["metas"]["net_id"]
        if key not in bucket:
            bucket[key] = [node]
        else:
            bucket[key].append(node)

    raw_nodes.clear()
    for node in base_nodes:
        correspond_node = bucket[node["metas"]["net_id"]].pop(0)
        raw_nodes.append(correspond_node)
