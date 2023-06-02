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

from ...utils import log_file, log, diff_log_path
import os


class TableView:
    """
    A search speedup wrapper class.
    """

    def __init__(self, data, key=None):
        self.data = data
        self.view = {}
        assert callable(key), "Key must be callable with a paramter: x -> key."
        for item in self.data:
            if key(item) not in self.view:
                self.view[key(item)] = [item]
            else:
                self.view[key(item)].append(item)

    def __getitem__(self, key):
        assert key in self.view, "{} is not found in index.".format(key)
        ret = self.view[key].pop(0)  # pop for sorting.
        return ret

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        return key in self.view


"""
    class definition
"""


class LayerStack(object):
    """
    this class is used to build module structure
    """

    def __init__(self, type_):
        super(LayerStack, self).__init__()
        self.type = type_
        self.stack = []

        self.root = None

    def _push(self, value):
        self.stack.append(value)

    def _pop(self):
        return self.stack.pop()

    def _top(self):
        if len(self.stack) == 0:
            return None
        return self.stack[-1]

    def _empty(self):
        return len(self.stack) == 0

    def push_layer(self, module):
        net = NetWrap(module, self.type)
        if not self._empty():
            net.father = self._top()
            self._top().children.append(net)
        else:
            if self.root is None:
                self.root = net
            else:
                raise RuntimeError("Found multy root layers! This err might caused by torch.utils.checkpoint.")
        self._push(net)

    def pop_layer(self, module):
        assert id(self._top().net) == id(module)
        self._pop()

    def push_api(self, api, fwd, bwd):
        # an api
        if hasattr(api, "__api__"):
            net = NetWrap(api, self.type)
            net.is_api = True
            net.is_leaf = True
            if not self._empty():
                self._top().children.append(net)
                net.father = self._top()
            net.set_report(fwd, bwd)

        # a layer_map marked layer
        else:
            net = self._top()
            net.set_report(fwd, bwd)


class NetWrap(object):
    def __init__(self, net, type_):
        self.type = type_

        self.net = net
        self.net_str = net.__name__ if hasattr(net, "__api__") else net.__class__.__name__
        self.children = []
        self.father = None

        self.is_api = False
        self.in_layer_map = False

        self.is_leaf = False
        self.fwd_report = None
        self.bwd_report = None

    def set_report(self, fwd, bwd):
        self.fwd_report = fwd
        self.bwd_report = bwd

    def __str__(self):
        if self.is_api:
            return "(api) " + self.net_str
        elif self.in_layer_map:
            return "(net in map) " + self.net_str
        else:
            return "(net) " + self.net_str

    def __repr__(self):
        return self.__str__()

    @property
    def fullname(self):
        return f"{self.type}::{self.net_str}"


"""
    operate tree
"""


def copy_module_struct(root):
    """
    copy and create a new struct tree

    Notice:
        a struct tree will not contain wrap layers
        because they are not in the retval of LayerMap.layers_skip_ignore
        so, del_wrap_layers is not needed, this function is used to copy

        why return a list? because i am too lazy to rewrite (x_x)
    """

    def copy_node_attrs(root):
        retval = NetWrap(root.net, root.type)
        for attr in ("is_api", "in_layer_map", "is_leaf", "fwd_report", "bwd_report"):
            val = getattr(root, attr)
            setattr(retval, attr, val)
        if hasattr(root, "model_name"):
            val = getattr(root, "model_name")
            setattr(retval, "model_name", val)
        setattr(retval, "origin", root)
        return retval

    if root.is_leaf:
        retval = copy_node_attrs(root)
        return [retval]

    retval = []
    for child in root.children:
        ret = copy_module_struct(child)
        retval.extend(ret)

    new_node = copy_node_attrs(root)
    for n in retval:
        n.father = new_node
        new_node.children.append(n)

    return [new_node]


# reorder roots[0] with struct of roots[1]
def reorder_and_match_reports(roots, reports):
    if len(roots[0].children) == 0 and len(roots[1].children) == 0:
        return

    layer_map = reports[0].layer_map

    # skip api layers, get table_view to find layer in init order
    fwd_items = list(filter(lambda x: not hasattr(x.net, "__api__"), reports[1].get_fwd_items()))
    table_view = TableView(fwd_items, lambda x: x.net_id)

    # split children to 3 parts
    base_apis = list(filter(lambda x: x.is_api, roots[0].children))
    base_opaque_layers = list(filter(lambda x: x.in_layer_map, roots[0].children))
    base_layers = list(filter(lambda x: not x.is_leaf, roots[0].children))

    raw_apis = list(filter(lambda x: x.is_api, roots[1].children))
    raw_opaque_layers = list(filter(lambda x: x.in_layer_map, roots[1].children))
    raw_layers = list(filter(lambda x: not x.is_leaf, roots[1].children))

    try:
        assert len(base_apis) == len(raw_apis), "number of api is different"
        assert len(base_opaque_layers) == len(raw_opaque_layers), "number of opaque_layers is different"
        assert len(base_layers) == len(raw_layers), "number of layer is different"

        # reset orders
        reorder_api(base_apis, raw_apis)
        reorder_opaque_layers(base_opaque_layers, raw_opaque_layers, layer_map)

        # for every child in roots[0], find correspond child in roots[1]
        new_children = []
        for child in roots[0].children:
            if child.is_api:
                new_children.append(raw_apis[0])
                raw_apis.pop(0)
            elif child.in_layer_map:
                new_children.append(raw_opaque_layers[0])
                raw_opaque_layers.pop(0)
            else:
                # use table_view to find correspond layer with init order
                report_item = table_view[child.fwd_report.net_id]
                correspond_child = next(x for x in raw_layers if x.fwd_report is report_item)
                if correspond_child is None:
                    raise RuntimeError("no match layer found")
                new_children.append(correspond_child)

        roots[1].children = new_children

        setattr(roots[1], "reordered", True)

    except Exception as e:
        raise e


def reorder_api(apis, base):
    """
    reorder apis based on base
    TODO(wuzhafnei): need better match logic there
    Temporarily, just keep in order
    """
    return


def reorder_opaque_layers(base, items, layer_map):
    def swap(seq, l, r):
        temp = seq[l]
        seq[l] = seq[r]
        seq[r] = temp
        return

    for target_idx, target_node in enumerate(base):
        # an api layer can not have in_layer_map mark, so node.net is save
        mapped_net = layer_map.map[target_node.net]
        correspond_node = next(node for node in items if node.net is mapped_net)
        item_idx = items.index(correspond_node)
        if item_idx == target_idx:
            continue
        elif item_idx > target_idx:
            swap(items, item_idx, target_idx)
        else:
            raise RuntimeError("Duplicate key or values, check your LayerMap")

    return


"""
    print tools
"""

# used to print module as a tree
def tree_print(root, mark=None, prefix=[]):

    cur_str = ""
    for i, s in enumerate(prefix):
        if i == len(prefix) - 1:
            cur_str += s
        else:
            if s == " |--- ":
                cur_str += " |    "
            elif s == " +--- ":
                cur_str += "      "
            else:
                cur_str += s

    cur_str += str(root)

    if os.getenv("PADIFF_PATH_LOG") == "ON" and hasattr(root.net, "path_info"):
        cur_str += "  (" + root.net.path_info + ")"

    if mark is root:
        cur_str += "    <---  *** HERE ***"

    ret_strs = [cur_str]

    for i, child in enumerate(root.children):
        pre = " |--- "
        if i == len(root.children) - 1:
            pre = " +--- "
        prefix.append(pre)
        retval = tree_print(child, mark, prefix)
        ret_strs.extend(retval)
        prefix.pop()

    return ret_strs


def get_path(node):
    msg = str(node)
    while node.father is not None:
        node = node.father
        msg = str(node) + " ---> " + msg
    return msg


def print_struct_info(roots, nodes):
    lines = 0
    infos = []

    for idx in range(2):
        node = nodes[idx]
        root = roots[idx]
        title = f"{root.model_name}\n" + "=" * 40 + "\n"
        retval = tree_print(root, mark=node, prefix=[" " * 4])
        info = title + "\n".join(retval)
        infos.append(info)
        lines += len(retval)

    if lines > 100:
        file_names = [f"diff_{roots[idx].model_name}.log" for idx in range(2)]
        for idx in range(2):
            info = infos[idx]
            log_file(file_names[idx], "w", info)
        log(f"Compare diff log saved to `{diff_log_path}/{file_names[0]}` and `{diff_log_path}/{file_names[1]}`.")
        log("Please view the reports and checkout the layers which is marked with `<---  *** HERE ***` !")

    else:
        for info in infos:
            print(info)


"""
    for debug
"""


def debug_print_struct(root):
    ret = tree_print(root)
    return "\n".join(ret)
