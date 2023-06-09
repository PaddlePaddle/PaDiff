from .diff_utils import default_loss
from ..utils import reset_dir
from ..report import SyncStepGuard
from ..dump_tools import dump_report, dump_weights, dump_grads
from ..checker import check_report, check_weights, check_grads


def pipeline(models, inputs, loss_fns, optimizers, options, cfg):
    if not options["single_step"]:
        normal_pipeline(models, inputs, loss_fns, optimizers, options, cfg)
    else:
        single_step_pipeline(models, inputs, loss_fns, optimizers, options, cfg)


def normal_pipeline(models, inputs, loss_fns, optimizers, options, cfg):
    for idx in range(2):
        run_and_dump(models[idx], inputs[idx], loss_fns[idx], optimizers[idx], options)
    
    auto_diff_paths = [model.dump_path + "/auto_diff" for model in models]

    check_report(auto_diff_paths[0], auto_diff_paths[1], cfg)
    check_grads(auto_diff_paths[0], auto_diff_paths[1], cfg)
    check_weights(auto_diff_paths[0], auto_diff_paths[1], cfg)


def single_step_pipeline(models, inputs, loss_fns, optimizers, options, cfg):
    auto_diff_paths = [model.dump_path + "/auto_diff" for model in models]

    models[0](inputs[0])
    dump_report(models[0], auto_diff_paths[0])
    if options["diff_phase"] in ("forward", "both"):
        with SyncStepGuard("forward", auto_diff_paths[0]):
            models[1](inputs[1])
            dump_report(models[1], auto_diff_paths[1])
            models[1].clear_report()
            check_report(auto_diff_paths[0], auto_diff_paths[1], cfg)

    run_and_dump(models[0], inputs[0], loss_fns[0], optimizers[0], options)
    if options["diff_phase"] in ("backward", "both"):
        with SyncStepGuard("backward", auto_diff_paths[0]):
            run_and_dump(models[1], inputs[1], loss_fns[1], optimizers[1], options)
            check_report(auto_diff_paths[0], auto_diff_paths[1], cfg)
            check_grads(auto_diff_paths[0], auto_diff_paths[1], cfg)
            check_weights(auto_diff_paths[0], auto_diff_paths[1], cfg)


def run_and_dump(model, input_, loss_fn, optimizer, options):
    dump_path =  model.dump_path + "/auto_diff"
    output = model(input_)
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
