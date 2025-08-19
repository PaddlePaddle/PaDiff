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


import paddle
import torch

from ..abstracts import ProxyModel, SyncStepGuard
from ..abstracts.proxy import create_model
from ..utils import for_each_tensor, log
from .assign_weight import assign_weight
from .checker import check_grads, check_report, check_weights, global_compare_configs
from ..tools import dump_grads, dump_report, dump_weights


def auto_diff(base_model, raw_model, inputs, loss_fns=None, optimizers=None, **kwargs):
    """
    Given example inputs, automatically find the first layer with precision diff.

    Args:
        base_model: paddle.nn.Layer or torch.nn.Module, provides the baseline of data precision.
        raw_model: paddle.nn.Layer or torch.nn.Module, which need to compare with base_model.
        inputs: input data for models, it should be a list of dict.
        loss_fns (list, optional): list of loss function for models.
        optimizers (list, optional): list of optimizers for models.
        kwargs: other options, view `https://github.com/PaddlePaddle/PaDiff` to learn more infomations
    Returns:
        True for success, False for failed.
    """

    options = kwargs

    if not isinstance(base_model, ProxyModel):
        base_model = create_model(base_model, base_model.__class__.__name__ + "_base_model")
    if not isinstance(raw_model, ProxyModel):
        raw_model = create_model(raw_model, raw_model.__class__.__name__ + "_raw_model")
    assert isinstance(inputs, (tuple, list)), "Invalid Argument."

    for input_ in inputs:
        assert isinstance(input_, dict), "Invalid Argument."

    if loss_fns is not None:
        options["use_loss"] = True
        assert len(loss_fns) == 2
        for loss in loss_fns:
            assert callable(loss), "Invalid loss function"
    else:
        loss_fns = [None, None]

    if optimizers is not None:
        options["use_opt"] = True
        assert len(optimizers) == 2
        for opt in optimizers:
            assert isinstance(opt, (paddle.optimizer.Optimizer, torch.optim.Optimizer)) or callable(
                opt
            ), "Invalid optimizer"
        optimizers = [OptimizerHelper(opt) for opt in optimizers]
    else:
        optimizers = [None, None]

    init_options(options)
    cfg = {}
    for key in ("atol", "rtol", "compare_mode"):
        cfg[key] = options[key]
        del options[key]

    if options["auto_init"] and not assign_weight(base_model, raw_model):
        return False

    result = pipeline((base_model, raw_model), inputs, loss_fns, optimizers, options, cfg)

    if result:
        log("SUCCESS !!!\n")
    else:
        log("FAILED !!!\n")

    return result


def pipeline(models, inputs, loss_fns, optimizers, options, cfg):
    if not options["single_step"]:
        return normal_pipeline(models, inputs, loss_fns, optimizers, options, cfg)
    else:
        return single_step_pipeline(models, inputs, loss_fns, optimizers, options, cfg)


def normal_pipeline(models, inputs, loss_fns, optimizers, options, cfg):
    auto_diff_paths = [model.dump_path + "/auto_diff" for model in models]
    for idx in range(2):
        run_and_dump(models[idx], inputs[idx], loss_fns[idx], optimizers[idx], options, auto_diff_paths[idx])

    retval = check_report(auto_diff_paths[0], auto_diff_paths[1], cfg)
    if options["diff_phase"] != "forward":
        retval = retval and check_grads(auto_diff_paths[0], auto_diff_paths[1], cfg)
        if options["use_opt"]:
            retval = retval and check_weights(auto_diff_paths[0], auto_diff_paths[1], cfg)
    return retval


def single_step_pipeline(models, inputs, loss_fns, optimizers, options, cfg):
    auto_diff_paths = [model.dump_path + "/auto_diff" for model in models]

    models[0](**inputs[0])
    dump_report(models[0], auto_diff_paths[0])
    models[0].clear_report()
    if options["diff_phase"] in ("forward", "both"):
        with SyncStepGuard("forward", auto_diff_paths[0]):
            models[1](**inputs[1])
            dump_report(models[1], auto_diff_paths[1])
            models[1].clear_report()
            retval = check_report(auto_diff_paths[0], auto_diff_paths[1], cfg, "forward")
            if retval == False:
                log("In single step mode, diff found at forward stage!")
                return False

    if options["diff_phase"] in ("backward", "both"):
        run_and_dump(models[0], inputs[0], loss_fns[0], optimizers[0], options, auto_diff_paths[0])
        with SyncStepGuard("backward", auto_diff_paths[0]):
            run_and_dump(models[1], inputs[1], loss_fns[1], optimizers[1], options, auto_diff_paths[1])
            retval = check_report(auto_diff_paths[0], auto_diff_paths[1], cfg, "backward")
            if options["diff_phase"] != "forward":
                retval = retval and check_grads(auto_diff_paths[0], auto_diff_paths[1], cfg)
                if options["use_opt"]:
                    retval = retval and check_weights(auto_diff_paths[0], auto_diff_paths[1], cfg)
            if retval == False:
                log("In single step mode, diff found at backward stage!")
                return False
    return True


def run_and_dump(model, input_, loss_fn, optimizer, options, dump_path):
    output = model(**input_)
    if options["diff_phase"] != "forward":
        if options["use_loss"]:
            loss = loss_fn(output)
        else:
            loss = default_loss(output, model.framework)
        model.backward(loss)
        dump_report(model, dump_path)
        model.clear_report()
        dump_grads(model, dump_path)
        if options["use_opt"]:
            optimizer.step()
            dump_weights(model, dump_path)
    else:
        dump_report(model, dump_path)


def default_loss(inp, mode):
    if isinstance(inp, torch.Tensor) or isinstance(inp, paddle.Tensor):
        return inp.mean()

    if mode == "torch":
        means = []
        for t in for_each_tensor(inp):
            means.append(t[0].to(torch.float32).mean())
        loss = torch.stack(means).mean()
        return loss
    elif mode == "paddle":
        means = []
        for t in for_each_tensor(inp):
            means.append(t[0].astype("float32").mean())
        loss = paddle.stack(means).mean()
        return loss
    else:
        raise RuntimeError(f"unrecognized mode `{mode}`, expected: `torch` or `paddle`")


def init_options(options):
    default_options = {
        "auto_init": True,
        "diff_phase": "both",
        "single_step": False,
        "steps": 1,
        "use_loss": False,
        "use_opt": False,
    }
    default_options.update(global_compare_configs)
    default_options.update(options)
    options.update(default_options)

    if not options["single_step"] and options["diff_phase"] == "backward":
        options["diff_phase"] = "both"
        log("  Not in single_step mode, diff_phase `backward` is not supported, set to `both` instead.")

    if options["diff_phase"] == "forward":
        if options["use_opt"]:
            options["use_opt"] = False
            log("  Diff_phase is `forward`, optimizer will not be used.")
        if options["steps"] > 1:
            options["steps"] = 1
            log("  Diff_phase is `forward`, steps is set to `1`.")

    if options["steps"] > 1 and options["use_opt"] == False:
        options["steps"] = 1
        log("  Steps is set to `1`, because optimizers are not given.")

    log("Your options:")
    print("{")
    for key in options.keys():
        if key in ["atol", "rtol", "compare_mode", "auto_init", "single_step", "use_loss", "use_opt"]:
            print(f"  {key}: `{options[key]}`")
    print("}")


class OptimizerHelper:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        if isinstance(
            self.optimizer,
            paddle.optimizer.Optimizer,
        ):
            self.optimizer.step()
            self.optimizer.clear_grad()
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            self.optimizer()
