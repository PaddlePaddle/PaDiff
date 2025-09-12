# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import ast
import astor
import os
from .utils import logger


class PaDiffInjector(ast.NodeTransformer):
    def __init__(
        self,
        framework: str,
        model_name="model",
        mode="base",
        **kwargs,
    ):
        self.base_name = model_name.split(".")[0]  # get trainer if trainer.model
        self.model_name = model_name  # model(inputs)
        self.padiff_model_name = f"model_{framework.lower()}"  # "model_paddle"
        self.proxy_model_name = "proxy_model"  # proxy_model = create_model(model)
        self.mode = mode
        self.kwargs = kwargs
        self.kwargs["framework"] = framework

        if self.mode == "align":
            base_dump_path = kwargs.get("base_dump_path", None)
            assert base_dump_path is not None, "'base_dump_path' should not be None in align mode."

        # exclude_methods: calls to these methods will not be injected into PaDiffGuard
        self.exclude_methods = {
            "to",
            "train",
            "eval",
            "state_dict",
            "load_state_dict",
            "parameters",
            "named_parameters",
            "buffers",
            "named_buffers",
            "zero_grad",
            "apply",
            "cuda",
            "cpu",
            "float",
            "double",
        }

    def visit_Module(self, node):
        node.body = self.add_imports(node)
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        # with PaDiffGuard(proxy_model):
        if self.is_model_call(node.value):
            return self.wrap_with_guard(node)

        return node

    def visit_Expr(self, node):
        if self.is_model_call(node.value):
            return self.wrap_with_guard(node)
        return node

    def visit_Return(self, node):
        if node.value is not None and self.is_model_call(node.value):
            return self.wrap_with_guard(node)
        return node

    def is_model_call(self, node):
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        # model(inp)
        if isinstance(func, ast.Name) and func.id == self.base_name:
            return True
        # model.forward(inp), model.submodule(inp)
        if isinstance(func, ast.Attribute):
            if self.base_name != "trainer" and func.attr in self.exclude_methods:
                return False
            return self.is_model_attribute(func, self.base_name)
        return False

    def is_model_attribute(self, node, root="model"):
        seen = set()
        while isinstance(node, ast.Attribute):
            if id(node) in seen:
                break
            seen.add(id(node))
            value = node.value
            if isinstance(value, ast.Name) and value.id == root:
                return True
            node = value
        return False

    def add_imports(self, node):
        has_padiff_import = False
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "padiff":
                imported_names = {alias.name for alias in stmt.names}
                required = {"PaDiffGuard"}
                if required <= imported_names:
                    has_padiff_import = True
                    break
        if has_padiff_import:
            return node.body

        import_from = ast.ImportFrom(
            module="padiff",
            names=[ast.alias(name="PaDiffGuard", asname=None)],
            level=0,
        )
        return [import_from] + node.body

    def wrap_with_guard(self, node):
        path = self.model_name.split(".")
        model_node = ast.Name(id=path[0], ctx=ast.Load())
        for attr in path[1:]:  # if trainer.model
            model_node = ast.Attribute(value=model_node, attr=attr, ctx=ast.Load())
        guard_args = [model_node]

        guard_keywords = []

        # name
        name_kw = ast.keyword(arg="name", value=ast.Constant(value=self.padiff_model_name))
        guard_keywords.append(name_kw)

        # optimizer
        if "optimizer" in self.kwargs:
            optim_kw = ast.keyword(arg="optimizer", value=ast.Name(id=self.kwargs["optimizer"], ctx=ast.Load()))
            guard_keywords.append(optim_kw)

        for key, value in self.kwargs.items():
            if key in ["optimizer"]:
                continue

            ast_value = self.safe_ast_value(value)
            if ast_value is not None:
                keyword = ast.keyword(arg=key, value=ast_value)
                guard_keywords.append(keyword)

        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Name(id="PaDiffGuard", ctx=ast.Load()),
                        args=guard_args,
                        keywords=guard_keywords,
                    ),
                    optional_vars=None,
                )
            ],
            body=[node],
        )
        ast.copy_location(with_stmt, node)
        ast.fix_missing_locations(with_stmt)
        return with_stmt

    def safe_ast_value(self, py_value):
        if isinstance(py_value, bool):
            return ast.Constant(value=py_value)
        elif isinstance(py_value, (int, float, str)):
            return ast.Constant(value=py_value)
        elif py_value is None:
            return ast.Constant(value=None)
        elif isinstance(py_value, list):
            elts = [self.safe_ast_value(item) for item in py_value]
            return ast.List(elts=elts, ctx=ast.Load())
        elif isinstance(py_value, dict):
            keys = [ast.Constant(k) for k in py_value.keys()]
            values = [self.safe_ast_value(v) for v in py_value.values()]
            return ast.Dict(keys=keys, values=values)
        else:
            logger.warning(f"Cannot inject parameter of type {type(py_value)}. Skipping.")
        return None


def create_injected_script(
    src_script_path: str,
    framework: str,
    model_name: str = "model",
    mode: str = "base",
    **kwargs,
) -> str:
    # read source script
    with open(src_script_path, "r", encoding="utf-8") as f:
        source = f.read()

    # appling PaDiffInjector
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.error(f"Failed to parse {src_script_path}: {e}")
        sys.exit(1)

    injector = PaDiffInjector(
        framework,
        model_name=model_name,
        mode=mode,
        **kwargs,
    )
    new_tree = injector.visit(tree)
    ast.fix_missing_locations(new_tree)
    code = astor.to_source(new_tree)

    # save injected script
    script_dir = os.path.dirname(src_script_path)
    if not script_dir:
        script_dir = "."
    script_filename = f"debug_inject_{framework}.py"
    new_script_path = os.path.join(script_dir, script_filename)
    with open(new_script_path, "w", encoding="utf-8") as f:
        f.write(code)
    logger.debug(f"Saved injected script to: {os.path.abspath(new_script_path)}")
    return new_script_path
