import json
import os, sys
import numpy
from .utils import Counter, reset_dir


dump_root = os.path.join(sys.path[0], "padiff_dump")
def set_dump_root(path):
    global dump_root
    dump_root = path


def numpy_dumper(path, prefix):
    reset_dir(path)
    counter = Counter()
    def dumper(value):
        id_ = counter.get_id()
        file_name = f"{path}/{prefix}_{id_}.npy"
        numpy.save(file_name, value)
        return file_name
    return dumper


def dump_runtime(model, dump_path):
    report = model.report
    tensor_path = dump_path + "/tensors"
    tensor_dumper = numpy_dumper(tensor_path, "tensor")
    report_info = {
        "model_name": model.name,
        "framework": model.framework,
        "file_path": f"{dump_path}/report.json",
        "layer_map": model.marker.layer_map,
        "tree": dump_tree_node(report.stack.root, tensor_dumper),
    }
    with open(f"{dump_path}/runtime.json", "w") as fp:
        json.dump(report_info, fp, indent=4)


def dump_tree_node(node, tensor_dumper):
    node_info = {
        "name": node.net_str,
        "route": node.net.path_info if hasattr(node.net, "path_info") else "",
        "type": node.layer_type,
        "fwd_outputs": [],
        "bwd_grads": [],
        "metas": {
            "fwd_runtime_step": node.fwd_report.step,
            "bwd_runtime_step": node.bwd_report.step,
            "net_id": node.fwd_report.net_id,
        },
        "children": [],
    }
    for tensor in node.fwd_report.tensors_for_compare():
        file_name = tensor_dumper(tensor.detach().numpy())
        node_info["fwd_outputs"].append(file_name)

    for tensor in node.bwd_report.tensors_for_compare():
        file_name = tensor_dumper(tensor.detach().numpy())
        node_info["bwd_grads"].append(file_name)

    for child in node.children:
        child_info = dump_tree_node(child, tensor_dumper)
        node_info["children"].append(child_info)

    return node_info


# TODO update

def dump_param_with_fn(model, fn):
    param_info = {
        "name": model.class_name,
        "route": model.path_info if hasattr(model, "path_info") else "",
        "repr": model.model_repr_info(),
        "children": [],
    }
    for param_name, param in model.named_parameters(recursively=False):
        fn(param_name, param, param_info)
    for name, child in model.named_children():
        param_info["children"].append(dump_param_with_fn(child, fn))
    return param_info


def dump_weights_grads(model, path):
    weight_dumper = numpy_dumper(path + "/weights", "weight")
    grad_dumper = numpy_dumper(path + "/grads", "grads")

    def _dump_weight_grad(param_name, param, param_info):
        if "weights" not in param_info.keys():
            param_info["weights"] = {}
            param_info["grads"] = {}
        file_name = weight_dumper(param.numpy())
        param_info["weights"][param_name] = file_name
        file_name = grad_dumper(param.grad())
        param_info["grads"][param_name] = file_name

    param_info = dump_param_with_fn(model, _dump_weight_grad)

    model_info = {
        "model_name": model.name,
        "framework": model.framework,
        "file_path": f"{path}/params.json",
        "tree": param_info,
    } 
    with open(f"{path}/params.json", "w") as fp:
        json.dump(model_info, fp, indent=4)
