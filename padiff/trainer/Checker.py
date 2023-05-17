# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


from .trainer_utils import (
    get_action,
    copy_module_struct,
    print_struct_info,
    reorder_and_match_reports,
)
from ..utils import log, assert_tensor_equal, LayerMap, log_file, diff_log_path
from ..weights import process_each_weight, shape_check
import numpy


__all__ = [
    "Checker",
]


class Checker:
    @staticmethod
    def check_forward_and_backward(reports, options):
        ret = check_forward_and_backward(reports, options)
        return ret

    @staticmethod
    def check_weight(layer, module, options, layer_map=LayerMap()):
        ret = check_weight(layer, module, options=options, layer_map=layer_map)
        return ret

    @staticmethod
    def check_grad(layer, module, options, layer_map=LayerMap()):
        ret = check_grad(layer, module, options=options, layer_map=layer_map)
        return ret


"""
    check forward and backward
"""


def check_forward_and_backward(reports, options):
    roots = [copy_module_struct(x.stack.root)[0] for x in reports]

    # forward check
    res = check_forward(roots, reports, options)
    if res == False:
        return False
    log("forward stage compared.")

    # loss check
    if options["use_loss"]:
        try:
            assert_tensor_equal(reports[0].loss, reports[1].loss, options)
            log("loss compared.")
        except Exception as e:
            log("*** Diff found in loss, Checkout your loss function! ***")
            log("loss compare:\n")
            print("{}".format(type(e).__name__ + ":  " + str(e)))
            return False

    if options["diff_phase"] == "forward":
        log("Diff_phase is `forward`. Backward compare skipped.")
        return True

    # backward check
    res = check_backward(roots, reports, options)
    if res == False:
        return False
    log("backward stage compared.")

    return True


def check_forward(roots, reports, options):
    act = get_action(roots[0].net, roots[1].net)
    items = [x.fwd_report for x in roots]
    assert items[0].type == items[1].type and items[0].type == "forward"

    try:
        act(items[0], items[1], options)
        return True
    except Exception as e:
        compare_info = e
        if len(roots[0].children) == 0 or len(roots[1].children) == 0:
            print_info(items, e, -1, grad=False, nodes=roots)
            return False

    # reorder current level
    try:
        if not hasattr(roots[0], "reordered"):
            reorder_and_match_reports(roots, reports)
    except Exception as e:
        log(
            f"While checking forward, diff found at first model {roots[0].fullname} vs second model {roots[0].fullname}"
        )
        log("Call `reorder_and_match_reports` for more detailed infos, but error occurs again:")
        print(type(e).__name__ + ":  " + str(e))
        log("Compare detail:")
        print_info(items, compare_info, -1, grad=False, nodes=roots)
        return False

    for child_0, child_1 in zip(roots[0].children, roots[1].children):
        res = check_forward((child_0, child_1), reports, options)
        if res == False:
            return False

    # sublayers is compared ok, but diff found at father layer
    log(
        f"Sublayers of first model {roots[0].fullname} and second model {roots[1].fullname} are corresponded, but diff found at their output!"
    )
    print_info(items, compare_info, -1, grad=False, nodes=roots)
    return False


def check_backward(roots, reports, options):
    act = get_action(roots[0].net, roots[1].net)
    items = [x.bwd_report for x in roots]
    assert items[0].type == items[1].type and items[0].type == "backward"

    try:
        act(items[0], items[1], options)
        return True
    except Exception as e:
        compare_info = e
        if len(roots[0].children) == 0 or len(roots[1].children) == 0:
            print_info(items, e, -1, grad=True, nodes=roots)
            return False

    # reorder current level
    try:
        if not hasattr(roots[0], "reordered"):
            reorder_and_match_reports(roots, reports)
    except Exception as e:
        log(
            f"While checking backward, diff found at first model {roots[0].fullname} vs second model {roots[0].fullname}"
        )
        log("Call `reorder_and_match_reports` for more detailed infos, but error occurs again:")
        print(type(e).__name__ + ":  " + str(e))
        log("Compare detail:")
        print_info(items, compare_info, -1, grad=True, nodes=roots)
        return False

    for child_0, child_1 in zip(reversed(roots[0].children), reversed(roots[1].children)):
        res = check_backward((child_0, child_1), reports, options)
        if res == False:
            return False

    # sublayers is compared ok, but diff found at father layer
    log(
        f"Grad of sublayers of first model {roots[0].fullname} and second model {roots[1].fullname} are corresponded, but diff found at their output!"
    )
    print_info(items, compare_info, -1, grad=True, nodes=roots)
    return False


def print_info(items, exc, step_idx, grad=False, nodes=None):
    if step_idx == -1:
        step_idx = items[1].step
    log("FAILED !!!")
    if grad:
        log(
            "    Diff found in `Backward Stage` in step: {}, net_id is {} vs {}".format(
                step_idx, items[0].net_id, items[1].net_id
            )
        )
    else:
        log(
            "    Diff found in `Forward  Stage` in step: {}, net_id is {} vs {}".format(
                step_idx, items[0].net_id, items[1].net_id
            )
        )
    log("    Type of layer is  : {} vs {}".format(items[0].net_str, items[1].net_str))

    print(str(exc))

    if nodes is not None:
        print("\n")
        log("Check model struct:")
        print_struct_info(nodes)

    print(f"\n\nStacks of Model[0]:")
    print("=========================")
    items[0].print_stacks()
    print(f"Stacks of Model[1]:")
    print("=========================")
    items[1].print_stacks()
    print("")


"""
    check weight and grad
"""


def check_weight(model_0, model_1, options, layer_map=LayerMap()):
    _weight_check = True

    def _check_weight(
        submodels,
        param_names,
        params,
        settings,
    ):
        shape_check(
            submodels,
            param_names,
            params,
            settings,
        )
        np_value_0 = params[0].numpy()
        np_value_1 = params[1].numpy()

        if settings["transpose"]:
            np_value_1 = numpy.transpose(np_value_1)

        # check weight
        try:
            assert_tensor_equal(np_value_0, np_value_1, settings)
        except Exception as e:
            nonlocal _weight_check
            _weight_check = False
            info = (
                "=" * 25 + "\n" + "After training, weight value is different.\n"
                "between Model[0] `{}`, Model[1] `{}` \n"
                "Model[0] param path:\n    {}\n"
                "Model[1] param path:\n    {}\n"
                "{}\n\n".format(
                    model_0[0].model_repr_info(),
                    model_1[1].model_repr_info(),
                    submodels[0].padiff_path + "." + param_names[0],
                    submodels[1].padiff_path + "." + param_names[1],
                    type(e).__name__ + ":  " + str(e),
                )
            )
            log_file("weight_diff.log", "a", info)

    try:
        process_each_weight(_check_weight, model_0, model_1, layer_map)
    except Exception as e:
        log("Err occurs when compare weight!!!\n")
        print(type(e).__name__ + ":  " + str(e))
        return False

    if _weight_check == False:
        log(f"Diff found in model weights after optimizer step, check report `{diff_log_path + '/weight_diff.log'}`.")
    else:
        log("weight compared.")

    return _weight_check


def check_grad(model_0, model_1, options, layer_map=LayerMap()):
    _grad_check = True

    def _check_grad(
        submodels,
        param_names,
        params,
        settings,
    ):
        shape_check(
            submodels,
            param_names,
            params,
            settings,
        )

        # grad() returns numpy value here
        grad_0 = params[0].grad()
        grad_1 = params[1].grad()

        # check grad
        try:
            if grad_0 is None and grad_1 is None:
                return
            elif grad_0 is None and grad_1 is not None:
                raise RuntimeError(
                    f"Found grad in first model is `None`, when grad in second model exists. Please check grad value in first model."
                )
            elif grad_0 is not None and grad_1 is None:
                raise RuntimeError(
                    f"Found grad in second model is `None`, when grad in first model exists. Please check grad value in second model."
                )

            if settings["transpose"]:
                grad_1 = numpy.transpose(grad_1)

            assert_tensor_equal(grad_0, grad_1, settings)
        except Exception as e:
            nonlocal _grad_check
            _grad_check = False
            info = (
                "=" * 25 + "\n" + "After training, grad value is different.\n"
                "between Model[0] `{}`, Model[1] `{}` \n"
                "Model[0] grad path:\n    {}\n"
                "Model[1] grad path:\n    {}\n"
                "{}\n\n".format(
                    model_0[0].model_repr_info(),
                    model_1[1].model_repr_info(),
                    submodels[0].padiff_path + "." + param_names[0],
                    submodels[1].padiff_path + "." + param_names[1],
                    type(e).__name__ + ":  " + str(e),
                )
            )
            log_file("grad_diff.log", "a", info)

    try:
        process_each_weight(_check_grad, model_0, model_1, layer_map)
    except Exception as e:
        log("Err occurs when compare grad!!!\n")
        print(type(e).__name__ + ":  " + str(e))
        return False

    if _grad_check == False:
        log(f"Diff found in model grad after backward, check report `{diff_log_path + '/grad_diff.log'}`.")
    else:
        log("grad compared.")

    return _grad_check
