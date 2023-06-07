from functools import partial
import paddle
import torch

def init_path_info(model):
    def _set_path_info(model, path):
        for name, child in model.named_children():
            path.append(name)
            if not hasattr(child.model, "path_info"):
                setattr(child.model, "path_info", ".".join(path))
            _set_path_info(child, path)
            path.pop()

    if not hasattr(model, "path_info"):
        setattr(model.model, "path_info", model.name)
        _set_path_info(model, [model.name])

def remove_inplace(model):
    """
    Set `inplace` tag to `False` for torch module
    """
    for submodel in model.submodels():
        if hasattr(submodel, "inplace"):
            submodel.inplace = False


def deco_iter(iterator, fn):
    def new_fn(obj):
        try:
            return fn(obj)
        except:
            return obj

    def new_generator():
        for obj in iterator:
            if isinstance(obj, (tuple, list)):
                yield tuple(map(new_fn, obj))
            else:
                yield new_fn(obj)

    return new_generator()


"""
    tools used in hook
"""

def auto(self, base_model, raw_model):
    """
    This method will try to find components which support special init, and add them to layer_map automatically.
    NOTICE: LayerMap.auto suppose that all sublayers/submodules are defined in same order, if not, this method may not work correctly.
    """

    def _traversal_layers(model, path, registered):
        for name, child in model.named_children():
            path.append(name)
            if child.fullname in registered and child.model not in self._ignored_layers:
                yield (child, ".".join(path))
            if child.fullname not in registered and child.model not in self._sublayer_ignored_layers:
                for sublayer, ret_path in _traversal_layers(child, path, registered):
                    yield (sublayer, ret_path)
            path.pop()

    # ProxyModel.create_from will do assert check for models
    base_model = ProxyModel.create_from(base_model)
    raw_model = ProxyModel.create_from(raw_model)

    base_submodels = list(_traversal_layers(base_model, [base_model.class_name], init_pool.registered_base_models))
    raw_submodels = list(_traversal_layers(raw_model, [raw_model.class_name], init_pool.registered_raw_models))

    _map = {}

    log("auto update LayerMap start searching...\n")

    for base_info, raw_info in zip_longest(base_submodels, raw_submodels, fillvalue=None):
        if raw_info is None or base_info is None:
            print(
                "\nError: The number of submodels which need special init is not the same! Check your model struct first!"
            )
            log("auto update LayerMap FAILED!!!\n")
            return False

        base_model, base_path = base_info
        raw_model, raw_path = raw_info
        name = build_name(base_model.model_type, base_model.class_name, raw_model.model_type, raw_model.class_name)
        if name in init_pool.funcs.keys():
            _map.update({base_model.model: raw_model.model})
            print(
                f"++++    base_model `{base_model.fullname}` at `{base_path}` <==>  raw_model `{raw_model.fullname}` at `{raw_path}`   ++++"
            )
        else:
            print("\nError: When generating LayerMap in order, find that raw_model can not matchs base_model.")
            print(f"    base_model:  `{base_model.fullname}` at `{base_path}`")
            print(f"    raw_model: `{raw_model.fullname}` at `{raw_path}`")
            log("auto update LayerMap FAILED!!!\n")
            return False
    print()
    log("auto update LayerMap SUCCESS!!!\n")

    self.map = _map
    return True
