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

from .. import utils
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
            # for N in self.stack:
            #     N.leafs.append(net)
            return

        # a layer
        net = self._top()
        net.set_report(fwd, bwd)
        # if net.is_one2one_layer:
        #     for N in self.stack[:-1]:
        #         N.leafs.append(net)


class NetWrap(object):
    def __init__(self, net, type_):
        self.type = type_

        self.net = net
        self.net_str = net if isinstance(net, str) else net.__class__.__name__
        self.children = []
        self.father = None

        # leafs under this net
        # self.leafs = []

        self.is_api = False
        self.is_one2one_layer = False

        # if is_leaf, report should exist
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
        # new_node.leafs.extend(n.leafs)

    return [new_node]


def reorder_and_match_reports(t_root, p_root, t_rep, p_rep):
    if len(t_root.children) == 0 and len(p_root.children) == 0:
        return

    layer_map = p_rep.layer_map

    # skip api layers
    p_fwd = list(filter(lambda x: not isinstance(x.net, str), p_rep.get_fwd_items()))
    p_table_view = TableView(p_fwd, lambda x: x.net_id)

    t_apis = list(filter(lambda x: x.is_api, t_root.children))
    t_one2one = list(filter(lambda x: x.is_one2one_layer, t_root.children))
    t_layers = list(filter(lambda x: not x.is_leaf, t_root.children))

    p_apis = list(filter(lambda x: x.is_api, p_root.children))
    p_one2one = list(filter(lambda x: x.is_one2one_layer, p_root.children))
    p_layers = list(filter(lambda x: not x.is_leaf, p_root.children))

    try:
        assert len(p_apis) == len(t_apis), "number of api is different"
        assert len(p_one2one) == len(t_one2one), "number of one2one is different"
        assert len(p_layers) == len(t_layers), "number of layer is different"

        reorder_api(t_apis, p_apis)
        reorder_one2one(t_one2one, p_one2one, layer_map)

        new_children = []
        for child in t_root.children:
            if child.is_api:
                new_children.append(p_apis[0])
                p_apis.pop(0)
            elif child.is_one2one_layer:
                new_children.append(p_one2one[0])
                p_one2one.pop(0)
            else:
                paddle_item = p_table_view[child.fwd_report.net_id]
                p_child = next(x for x in p_layers if x.fwd_report is paddle_item)
                if p_child is None:
                    raise RuntimeError("no match layer found")
                new_children.append(p_child)

        p_root.children = new_children

        setattr(p_root, "reordered", True)

    except Exception as e:
        raise e


def reorder_api(t_apis, p_apis):
    """
    reorder p_apis based on t_apis
    TODO(wuzhafnei): need better match logic there
    Temporarily, just keep in order
    """
    return


def reorder_one2one(t_oos, p_oos, layer_map):
    def swap(items, l, r):
        temp = items[l]
        items[l] = items[r]
        items[r] = temp
        return

    for idx, t_node in enumerate(t_oos):
        # an api layer can not have one2one mark, so node.net is save
        p_net = layer_map.map[t_node.net]
        p_node = next(node for node in p_oos if node.net is p_net)
        p_idx = p_oos.index(p_node)
        if p_idx == idx:
            continue
        elif p_idx > idx:
            swap(p_oos, p_idx, idx)
        else:
            raise RuntimeError("Duplicate key or values, check your LayerMap")

    return


def reorder_and_match_reports_recursively(t_root, p_root, t_rep, p_rep):
    """
    recoder tree's nodes with init order in place
    based on torch module and reorder paddle module

    Notice:
        for one2one layers, they may not in order too (though they are leaves)
    """
    reorder_and_match_reports(t_root, p_root, t_rep, p_rep)

    for t_child, p_child in zip(t_root.children, p_root.children):
        reorder_and_match_reports_recursively(t_child, p_child, p_rep, t_rep)

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


def print_struct_info(t_node, p_node):
    t_root = t_node
    while t_root.father is not None:
        t_root = t_root.father

    p_root = p_node
    while p_root.father is not None:
        p_root = p_root.father

    t_title = "Torch Model\n" + "=" * 25 + "\n"
    t_retval = tree_print(t_root, mark=t_node, prefix=[" " * 4])
    t_info = t_title + "\n".join(t_retval)

    p_title = "Paddle Model\n" + "=" * 25 + "\n"
    p_retval = tree_print(p_root, mark=p_node, prefix=[" " * 4])
    p_info = p_title + "\n".join(p_retval)

    if len(p_retval) + len(t_retval) > 100:
        utils.log_file("paddle_struct.log", "w", p_info)
        utils.log_file("torch_struct.log", "w", t_info)
        utils.log(
            f"Model Struct saved to `{utils.diff_log_path + '/torch_struct.log'}` and `{utils.diff_log_path + '/paddle_struct.log'}`."
        )
        utils.log("Please view the reports and checkout the layers which is marked with `<---  *** HERE ***` !")

    else:
        print(p_info)
        print(t_info)


"""
    for debug
"""


def debug_print_struct(root):
    ret = tree_print(root)
    return "\n".join(ret)
