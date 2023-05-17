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
                # warnings.warn("Warning: duplicate key is found, use list + pop strategy.")
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
        self.is_one2one_layer = False

        self.is_leaf = False
        self.fwd_report = None
        self.bwd_report = None

    def set_report(self, fwd, bwd):
        self.fwd_report = fwd
        self.bwd_report = bwd

    def __str__(self):
        if self.is_api:
            return "(api) " + self.net_str
        elif self.is_one2one_layer:
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
        for attr in ("is_api", "is_one2one_layer", "is_leaf", "fwd_report", "bwd_report"):
            val = getattr(root, attr)
            setattr(retval, attr, val)
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

    # skip api layers
    fwd_0 = list(filter(lambda x: not hasattr(x.net, "__api__"), reports[0].get_fwd_items()))
    table_view_0 = TableView(fwd_0, lambda x: x.net_id)

    # split children to 3 parts
    apis_0 = list(filter(lambda x: x.is_api, roots[0].children))
    one2one_0 = list(filter(lambda x: x.is_one2one_layer, roots[0].children))
    layers_0 = list(filter(lambda x: not x.is_leaf, roots[0].children))

    apis_1 = list(filter(lambda x: x.is_api, roots[1].children))
    one2one_1 = list(filter(lambda x: x.is_one2one_layer, roots[1].children))
    layers_1 = list(filter(lambda x: not x.is_leaf, roots[1].children))

    try:
        assert len(apis_0) == len(apis_1), "number of api is different"
        assert len(one2one_0) == len(one2one_1), "number of one2one is different"
        assert len(layers_0) == len(layers_1), "number of layer is different"

        # reset orders
        reorder_api(apis_0, apis_1)
        reorder_one2one(one2one_0, one2one_1, layer_map)

        # for every child in roots[1], find correspond child in roots[0]
        new_children = []
        for child in roots[1].children:
            if child.is_api:
                new_children.append(apis_0[0])
                apis_0.pop(0)
            elif child.is_one2one_layer:
                new_children.append(one2one_0[0])
                one2one_0.pop(0)
            else:
                # use table_view to find correspond layer with init order
                report_item = table_view_0[child.fwd_report.net_id]
                correspond_child = next(x for x in layers_0 if x.fwd_report is report_item)
                if correspond_child is None:
                    raise RuntimeError("no match layer found")
                new_children.append(correspond_child)

        roots[0].children = new_children

        setattr(roots[0], "reordered", True)

    except Exception as e:
        raise e


def reorder_api(apis, targets):
    """
    reorder apis based on targets
    TODO(wuzhafnei): need better match logic there
    Temporarily, just keep in order
    """
    return


def reorder_one2one(items, targets, layer_map):
    def swap(seq, l, r):
        temp = seq[l]
        seq[l] = seq[r]
        seq[r] = temp
        return

    for target_idx, target_node in enumerate(targets):
        # an api layer can not have one2one mark, so node.net is save
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


def reorder_and_match_reports_recursively(roots, reports):
    """
    recoder tree's nodes with init order in place
    based on torch module and reorder paddle module

    Notice:
        for one2one layers, they may not in order too (though they are leaves)
    """
    reorder_and_match_reports(roots, reports)

    for child_0, child_1 in zip(roots[0].children, roots[1].children):
        reorder_and_match_reports_recursively([child_0, child_1], reports)

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

    if os.getenv("PADIFF_PATH_LOG") == "ON" and hasattr(root.net, "padiff_path"):
        cur_str += "  (" + root.net.padiff_path + ")"

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


def print_struct_info(nodes):
    for idx, node in enumerate(nodes):
        root = node
        while root.father is not None:
            root = root.father
        title = f"Model[{idx}] {root.fullname}\n" + "=" * 40 + "\n"
        retval = tree_print(root, mark=node, prefix=[" " * 4])
        info = title + "\n".join(retval)

        if len(retval) > 50:
            file_name = f"model_{idx}_struct_{root.fullname}.log"
            log_file(file_name, "w", info)
            log(f"Model Struct saved to `{diff_log_path}/{file_name}`, which is marked with `<---  *** HERE ***` !")
        else:
            print(info)


"""
    for debug
"""


def debug_print_struct(root):
    ret = tree_print(root)
    return "\n".join(ret)
