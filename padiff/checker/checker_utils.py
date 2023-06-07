from ..utils import log_path, log_file, log
import numpy


def clone_dict_tree(root):
    new_root = {}
    new_root.update(root)
    new_root["origin_node"] = root
    new_root["reordered"] = False
    new_root["children"] = []
    for child in root["children"]:
        new_root["children"].append(clone_tree(child))
    return new_root

def print_runtime_info(nodes, reports, exc, stage):
    step_idx = [node["metas"]["fwd_runtime_step"] if stage == "Forward" else nodes["metas"]["bwd_runtime_step"] for node in nodes]
    net_id = [node["metas"]["net_id"] for node in nodes]

    log("FAILED !!!")
    log(f"    Diff found in {stage} Stage in step: {step_idx[0]} vs {step_idx[1]}, net_id is {net_id[0]} vs {net_id[1]}")
    log(f"    Type of layer is: {nodes[0]["name"]} vs {nodes[1]["name"]}")

    print(str(exc) + "\n\n")

    log("Check model struct:")
    retstr = struct_info_log(reports, [node["origin_node"] for node in nodes], "runtime")
    log(retstr)


def struct_info_log(reports, nodes, file_prefix):
    file_names = []
    for idx in range(2):
        node = nodes[idx]
        report = reports[idx]
        file_name = f"{file_prefix}_{report["model_name"]}.log"
        file_name = build_file_name(report, file_name)
        file_names.append(file_name)
        title = f"{report["model_name"]}\n" + "=" * 40 + "\n"
        retval = tree_print(report["tree"], mark=node, prefix=[" " * 4])
        info = title + "\n".join(retval)
        log_file(file_name, "w", info)

    retval = f"Log saved to `{log_path}/{file_names[0]}` and `{log_path}/{file_names[1]}`."
    retval += "Please view the reports and checkout the layer marked with `<---  *** HERE ***` !"
    return retval


def tree_print(node, mark=None, prefix=[]):
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

    cur_str += node["name"]
    if os.getenv("PADIFF_PATH_LOG") == "ON":
        cur_str += "  (" + node["route"] + ")"
    if mark is node:
        cur_str += "    <---  *** HERE ***"

    ret_strs = [cur_str]
    for i, child in enumerate(node["children"]):
        pre = " |--- "
        if i == len(node["children"]) - 1:
            pre = " +--- "
        prefix.append(pre)
        retval = tree_print(child, mark, prefix)
        ret_strs.extend(retval)
        prefix.pop()

    return ret_strs


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
        layer_map = dict[zip(reports[0]["layer_map"], reports[1]["layer_map"])]
        reorder_opaque_layers(base_opaque_layers, raw_opaque_layers, layer_map)
        reorder_normal_layers(base_layers, raw_layers)

        # for every child in nodes[0], find correspond child in nodes[1]
        new_children = []
        for child in nodes[0].children:
            if child.is_api:
                new_children.append(raw_apis[0])
                raw_apis.pop(0)
            elif child.in_layer_map:
                new_children.append(raw_opaque_layers[0])
                raw_opaque_layers.pop(0)
            else:
                new_children.append(raw_layers[0])
                raw_layers.pop(0)

        nodes[1].children = new_children
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
        correspond_route = layer_map[base_node["route"]]
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
        raw_nodes.append(correspond_child)


def traversal_nodes(nodes):
    pass

def build_file_name(report, file_name):
    pass

def load_numpy(path):
    return numpy.load(open(path, "r"))

def load_json(path):
    f = open(path, "r")
    retval = json.load(f)
    return retval