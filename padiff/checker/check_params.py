import json
import numpy
from itertools import zip_longest

from .checker_utils import clone_dict_tree, struct_info_log, load_numpy, build_file_name, global_compare_configs, load_json, traversal_node
from .actions import get_action
from ..utils import log, log_file, assert_tensor_equal
from ..datas import global_yaml_loader as yamls


def check_params(report_path_0, report_path_1, cfg=None):
    if cfg == None:
        cfg = global_compare_configs
    reports = [load_json(report_path_0, "params.json"), load_json(report_path_1, "params.json")]
    node_lists = [traversal_node(rep["tree"]) for rep in reports]

    weight_equal = check_target(assert_weight, node_lists, reports, "weights", cfg)
    grad_equal = check_target(assert_grad, node_lists, reports, "grads", cfg)
    return weight_equal and grad_equal


def check_weights(report_path_0, report_path_1, cfg=None):
    if cfg == None:
        cfg = global_compare_configs
    reports = [load_json(report_path_0, "weights.json"), load_json(report_path_1, "weights.json")]
    node_lists = [traversal_node(rep["tree"]) for rep in reports]

    weight_equal = check_target(assert_weight, node_lists, reports, "weights", cfg)
    return weight_equal


def check_grads(report_path_0, report_path_1, cfg=None):
    if cfg == None:
        cfg = global_compare_configs
    reports = [load_json(report_path_0, "grads.json"), load_json(report_path_1, "grads.json")]
    node_lists = [traversal_node(rep["tree"]) for rep in reports]

    grad_equal = check_target(assert_grad, node_lists, reports, "grads", cfg)
    return grad_equal


def check_target(fn, node_lists, reports, compare_target, cfg):
    flag = True
    log_path = build_file_name(reports[0], compare_target + "_diff")

    def checker(nodes, param_names, params, settings):
        assert_shape(params, settings)
        try:
            fn(params, settings)
        except Exception as e:
            nonlocal flag
            flag = False
            info = (
                "=" * 25 + "\n" + "{} value is different.\n"
                "between base_model: `{}` under {}, "
                "        raw_model: `{}` under {}\n\n"
                "base_model param path:\n    {}\n"
                "raw_model param path:\n    {}\n"
                "{}\n\n".format(
                    compare_target,
                    nodes[0]["repr"],
                    reports[0]["model_name"],
                    nodes[1]["repr"],
                    reports[1]["model_name"],
                    nodes[0]["route"] + "." + param_names[0],
                    nodes[1]["route"] + "." + param_names[1],
                    type(e).__name__ + ":  " + str(e),
                )
            )
            log_file(log_path, "a", info)

    try:
        process_each_param(checker, node_lists, reports, compare_target, cfg)
    except Exception as e:
        log(f"Err occurs when compare {compare_target}!!!\n")
        print(type(e).__name__ + ":  " + str(e))
        return False

    if flag == False:
        log(f"Diff found when compare {compare_target}, please check report `{log_path}`.")
    else:
        log(f"{compare_target} compared.")

    return flag


def process_each_param(process, node_lists, reports, target, cfg):
    for node_0, node_1 in zip_longest(node_lists[0], node_lists[1], fillvalue=None):
        if node_0 is None or node_1 is None:
            raise RuntimeError("Found model with difference number of sublayers. Check your model.")
        for (param_name_0, param_path_0), (param_name_1, param_path_1) in zip(
            node_0[target].items(),
            node_1[target].items(),
        ):
            try:
                settings = yamls.get_weight_settings((node_0["name"], node_1["name"]), (reports[0]["framework"], reports[1]["framework"]), (param_name_0, param_name_1))
                settings.update(cfg)
                param_0 = load_numpy(param_path_0)
                param_1 = load_numpy(param_path_1)
                process(
                    (node_0, node_1),
                    (param_name_0, param_name_1),
                    (param_0, param_1),
                    settings
                )
            except Exception as e:
                err_str = f"Error occured between:\n"
                err_str += f"    base_model: `{node_0['repr']}`\n"
                err_str += f"                `{node_0['route'] + '.' + param_name_0}`\n"
                err_str += f"    raw_model: `{node_1['repr']}`\n"
                err_str += f"               `{node_1['route'] + '.' + param_name_1}`\n"
                err_str += f"{type(e).__name__ + ':  ' + str(e)}\n"
                err_str += struct_info_log(reports, (node_0, node_1), target)

                err_str += "\nHint:\n"
                err_str += "    1. Check the definition order of params is same in submodels.\n"
                err_str += "    2. Check the corresponding submodel have the same style:\n"
                err_str += "       param <=> param, buffer <=> buffer, embedding <=> embedding ...\n"
                err_str += "       cases like param <=> buffer, param <=> embedding are not allowed.\n"
                err_str += "    3. If can not change model codes, try to use a `LayerMap`\n"
                err_str += "       which can solve most problems.\n"
                err_str += "    4. (skip) means this layer is skipped because it is under black_list, or it has no param."
                err_str += "    0. Visit `https://github.com/PaddlePaddle/PaDiff` to find more infomation.\n"

                raise RuntimeError(err_str)


def assert_shape(params, settings):
    shape_0 = list(params[0].shape)
    shape_1 = list(params[1].shape)
    if settings["transpose"]:
        shape_1.reverse()
    assert (
        shape_0 == shape_1
    ), f"Shape not same. {shape_0} vs {shape_1}\n"


def assert_weight(params, settings):
    if settings["transpose"]:
        params[1] = numpy.transpose(params[1])
    assert_tensor_equal(params[0], params[1], settings)


def assert_grad(params, settings):
    if params[0] is None and params[1] is None:
        return
    elif params[0] is None:
        raise RuntimeError(
            f"Found grad in base_model (1st) is `None`, when another is not!"
        )
    elif params[1] is None:
        raise RuntimeError(
            f"Found grad in raw_model (2nd) is `None`, when another is not!"
        )

    if settings["transpose"]:
        params[1] = numpy.transpose(params[1])

    assert_tensor_equal(params[0], params[1], settings)
