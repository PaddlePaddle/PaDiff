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
from ..utils import log, assert_tensor_equal, LayerMap, model_repr_info, log_file, diff_log_path
from ..weights import process_each_weight, shape_check
import numpy


__all__ = [
    "Checker",
]


class Checker:
    @staticmethod
    def check_forward_and_backward(torch_report, paddle_report, options):
        ret = check_forward_and_backward(torch_report, paddle_report, options)
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


def check_forward_and_backward(torch_rep, paddle_rep, options):
    t_root = copy_module_struct(torch_rep.stack.root)[0]
    p_root = copy_module_struct(paddle_rep.stack.root)[0]

    # forward check
    res = check_forward(t_root, p_root, torch_rep, paddle_rep, options)
    if res == False:
        return False
    log("forward stage compared.")

    # loss check
    if options["use_loss"]:
        try:
            assert_tensor_equal(paddle_rep.loss, torch_rep.loss, options)
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
    res = check_backward(t_root, p_root, torch_rep, paddle_rep, options)
    if res == False:
        return False
    log("backward stage compared.")

    return True


def check_forward(t_root, p_root, t_rep, p_rep, options):
    act = get_action(t_root.net, p_root.net)
    torch_item = t_root.fwd_report
    paddle_item = p_root.fwd_report
    assert torch_item.type == paddle_item.type and paddle_item.type == "forward"
    try:
        act(torch_item, paddle_item, options)
        return True
    except Exception as e:
        compare_info = e
        if len(t_root.children) == 0 or len(p_root.children) == 0:
            print_info(paddle_item, torch_item, e, -1, grad=False, t_root=t_root.origin, p_root=p_root.origin)
            return False

    # reorder current level
    try:
        if not hasattr(p_root, "reordered"):
            reorder_and_match_reports(t_root, p_root, t_rep, p_rep)
    except Exception as e:
        log(f"While checking forward, diff found at torch: {t_root} vs paddle: {p_root}")
        log("Call `reorder_and_match_reports` for more detailed infos, but error occurs again:")
        print(type(e).__name__ + ":  " + str(e))
        log("Compare detail:")
        print_info(paddle_item, torch_item, compare_info, -1, grad=False, t_root=t_root.origin, p_root=p_root.origin)
        return False

    for t_child, p_child in zip(t_root.children, p_root.children):
        res = check_forward(t_child, p_child, t_rep, p_rep, options)
        if res == False:
            return False

    # sublayers is compared ok, but diff found at father layer
    log(f"Sublayers of torch: {t_root} and paddle: {p_root} are corresponded, but diff found at their output!")
    print_info(paddle_item, torch_item, compare_info, -1, grad=False, t_root=t_root.origin, p_root=p_root.origin)
    return False


def check_backward(t_root, p_root, t_rep, p_rep, options):
    act = get_action(t_root.net, p_root.net)
    torch_item = t_root.bwd_report
    paddle_item = p_root.bwd_report
    assert torch_item.type == paddle_item.type and paddle_item.type == "backward"
    try:
        act(torch_item, paddle_item, options)
        return True
    except Exception as e:
        compare_info = e
        if len(t_root.children) == 0 or len(p_root.children) == 0:
            print_info(paddle_item, torch_item, e, -1, grad=True, t_root=t_root.origin, p_root=p_root.origin)
            return False

    # reorder current level
    try:
        if not hasattr(p_root, "reordered"):
            reorder_and_match_reports(t_root, p_root, t_rep, p_rep)
    except Exception as e:
        log(f"While checking backward, diff found at torch: {t_root} vs paddle: {p_root}")
        log("Call `reorder_and_match_reports` for more detailed infos, but error occurs again:")
        print(type(e).__name__ + ":  " + str(e))
        log("Compare detail:")
        print_info(paddle_item, torch_item, compare_info, -1, grad=True, t_root=t_root.origin, p_root=p_root.origin)
        return False

    for t_child, p_child in zip(reversed(t_root.children), reversed(p_root.children)):
        res = check_backward(t_child, p_child, t_rep, p_rep, options)
        if res == False:
            return False

    # sublayers is compared ok, but diff found at father layer
    log(
        f"Grad of sublayers of torch: {t_root} and paddle: {p_root} are corresponded, but diff found at their output grad!"
    )
    print_info(paddle_item, torch_item, compare_info, -1, grad=True, t_root=t_root.origin, p_root=p_root.origin)
    return False


def print_info(paddle_item, torch_item, exc, step_idx, grad=False, t_root=None, p_root=None):
    if step_idx == -1:
        step_idx = torch_item.step
    log("FAILED !!!")
    if grad:
        log(
            "    Diff found in `Backward Stage` in step: {}, net_id is {} vs {}".format(
                step_idx, paddle_item.net_id, torch_item.net_id
            )
        )
    else:
        log(
            "    Diff found in `Forward  Stage` in step: {}, net_id is {} vs {}".format(
                step_idx, paddle_item.net_id, torch_item.net_id
            )
        )
    log("    Type of layer is  : {} vs {}".format(torch_item.net_str, paddle_item.net_str))

    print(str(exc))

    if t_root is not None and p_root is not None:
        print("\n")
        log("Check model struct:")
        print_struct_info(t_root, p_root)

    print("\n\nPaddle Stacks:")
    print("=========================")
    paddle_item.print_stacks()
    print("Torch  Stacks:")
    print("=========================")
    torch_item.print_stacks()
    print("")


"""
    check weight and grad
"""


def check_weight(layer, module, options, layer_map=LayerMap()):
    _weight_check = True

    def _check_weight(
        paddle_sublayer,
        torch_submodule,
        paddle_pname,
        torch_pname,
        paddle_param,
        torch_param,
        settings,
    ):
        shape_check(
            paddle_sublayer,
            torch_submodule,
            paddle_pname,
            torch_pname,
            paddle_param,
            torch_param,
            settings,
        )
        p_param = paddle_param.numpy()
        t_param = torch_param.detach().cpu().numpy()

        if settings["transpose"]:
            t_param = numpy.transpose(t_param)

        # check weight
        try:
            assert_tensor_equal(p_param, t_param, settings)
        except Exception as e:
            nonlocal _weight_check
            _weight_check = False
            info = (
                "=" * 25 + "\n" + "After training, weight value is different.\n"
                "between paddle: `{}`, torch: `{}` \n"
                "paddle path:\n    {}\n"
                "torch path:\n    {}\n"
                "{}\n\n".format(
                    model_repr_info(paddle_sublayer),
                    model_repr_info(torch_submodule),
                    paddle_sublayer.padiff_path + "." + paddle_pname,
                    torch_submodule.padiff_path + "." + torch_pname,
                    type(e).__name__ + ":  " + str(e),
                )
            )
            log_file("weight_diff.log", "a", info)

    try:
        process_each_weight(_check_weight, layer, module, layer_map)
    except Exception as e:
        log("Err occurs when compare weight!!!\n")
        print(type(e).__name__ + ":  " + str(e))
        return False

    if _weight_check == False:
        log(f"Diff found in model weights after optimizer step, check report `{diff_log_path + '/weight_diff.log'}`.")
    else:
        log("weight compared.")

    return _weight_check


def check_grad(layer, module, options, layer_map=LayerMap()):
    _grad_check = True

    def _check_grad(
        paddle_sublayer,
        torch_submodule,
        paddle_pname,
        torch_pname,
        paddle_param,
        torch_param,
        settings,
    ):
        shape_check(
            paddle_sublayer,
            torch_submodule,
            paddle_pname,
            torch_pname,
            paddle_param,
            torch_param,
            settings,
        )
        p_grad = paddle_param.grad.numpy() if paddle_param.grad is not None else None
        t_grad = torch_param.grad.detach().cpu().numpy() if torch_param.grad is not None else None

        # check grad
        try:
            if p_grad is None and t_grad is None:
                return
            elif p_grad is None and t_grad is not None:
                raise RuntimeError(
                    f"Found paddle grad is `None`, when torch grad exists. Please check the paddle grad."
                )
            elif t_grad is None and p_grad is not None:
                raise RuntimeError(
                    f"Found torch grad is `None`, when paddle grad exists. Please check the torch grad."
                )

            if settings["transpose"]:
                t_grad = numpy.transpose(t_grad)

            assert_tensor_equal(p_grad, t_grad, settings)
        except Exception as e:
            nonlocal _grad_check
            _grad_check = False
            info = (
                "=" * 25 + "\n" + "After training, grad value is different.\n"
                "between paddle: `{}`, torch: `{}` \n"
                "paddle path:\n    {}\n"
                "torch path:\n    {}\n"
                "{}\n\n".format(
                    model_repr_info(paddle_sublayer),
                    model_repr_info(torch_submodule),
                    paddle_sublayer.padiff_path + "." + paddle_pname,
                    torch_submodule.padiff_path + "." + torch_pname,
                    type(e).__name__ + ":  " + str(e),
                )
            )
            log_file("grad_diff.log", "a", info)

    try:
        process_each_weight(_check_grad, layer, module, layer_map)
    except Exception as e:
        log("Err occurs when compare grad!!!\n")
        print(type(e).__name__ + ":  " + str(e))
        return False

    if _grad_check == False:
        log(f"Diff found in model grad after backward, check report `{diff_log_path + '/grad_diff.log'}`.")
    else:
        log("grad compared.")

    return _grad_check
