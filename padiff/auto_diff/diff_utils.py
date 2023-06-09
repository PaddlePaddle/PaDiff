from ..utils import log, for_each_tensor
from ..checker import global_compare_configs
import paddle
import torch


"""
    init tools
"""

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
            print("  {}: `{}`".format(key, options[key]))
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
        raise RuntimeError("unrecognized mode `{}`, expected: `torch` or `paddle`".format(mode))

