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
        src_model_name="model",
        mode="base",
        alignment_dir=None,
        single_step_mode=None,
        align_depth="inf",
    ):
        self.framework = framework
        self.src_model_name = src_model_name  # model(inputs)
        self.padiff_model_name = f"model_{framework.lower()}"  # "model_paddle"
        self.proxy_model_name = "proxy_model"  # proxy_model = create_model(model)
        self.mode = mode
        if self.mode == "align":
            assert alignment_dir is not None, "'alignment_dir' should not be None in align mode."
        self.alignment_dir = alignment_dir
        self.single_step_mode = single_step_mode
        self.align_depth = align_depth

        # black list: calls to these methods will not be injected into PaDiffGuard
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
        # dump_stmt = self.add_dump_report(node)
        # node.body.append(dump_stmt)
        return node

    def visit_Assign(self, node):
        # # model = SimplePaddle()
        # for target in node.targets:
        #     if isinstance(target, ast.Name) and target.id == self.src_model_name:
        #         return self.add_create_model(node)

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
        if isinstance(func, ast.Name) and func.id == self.src_model_name:
            return True
        # model.forward(inp), model.submodule(inp)
        if isinstance(func, ast.Attribute):
            if func.attr in self.exclude_methods:
                return False
            return self.is_model_attribute(func, self.src_model_name)
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
                # required = {"create_model", "PaDiffGuard", "dump_report"}
                required = {"PaDiffGuard"}
                if required <= imported_names:
                    has_padiff_import = True
                    break
        if has_padiff_import:
            return node.body

        import_from = ast.ImportFrom(
            module="padiff",
            names=[
                # ast.alias(name="create_model", asname=None),
                ast.alias(name="PaDiffGuard", asname=None),
                # ast.alias(name="dump_report", asname=None),
            ],
            level=0,
        )
        return [import_from] + node.body

    def add_create_model(self, node):
        # Temporarily abandoned
        # proxy_model = create_model()
        assign_proxy = ast.Assign(
            targets=[ast.Name(id=self.proxy_model_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="create_model", ctx=ast.Load()),
                args=[ast.Name(id=self.src_model_name, ctx=ast.Load())],
                keywords=[ast.keyword(arg="name", value=ast.Constant(value=self.padiff_model_name))],
            ),
        )

        # model._padiff_wrapped = True
        mark_wrapped = ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=self.src_model_name, ctx=ast.Load()), attr="_padiff_wrapped", ctx=ast.Store()
                )
            ],
            value=ast.Constant(value=True),
        )

        # global _padiff_proxy_model
        global_decl = ast.Global(names=["_padiff_proxy_model"])

        # _padiff_proxy_model = proxy_model
        assign_global = ast.Assign(
            targets=[ast.Name(id="_padiff_proxy_model", ctx=ast.Store())],
            value=ast.Name(id=self.proxy_model_name, ctx=ast.Load()),
        )

        # combine to if not hasattr()
        wrapper = ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Call(
                    func=ast.Name(id="hasattr", ctx=ast.Load()),
                    args=[ast.Name(id=self.src_model_name, ctx=ast.Load()), ast.Constant(value="_padiff_wrapped")],
                    keywords=[],
                ),
            ),
            body=[assign_proxy, mark_wrapped, global_decl, assign_global],
            orelse=[],
        )

        ast.copy_location(wrapper, node)
        ast.fix_missing_locations(wrapper)
        return [node, wrapper]

    def wrap_with_guard(self, node):
        guard_args = [ast.Name(id=self.src_model_name, ctx=ast.Load())]
        guard_keywords = []

        if self.mode == "align":
            # load_init_weights
            load_weights_kw = ast.keyword(arg="load_init_weights", value=ast.Constant(value=True))
            guard_keywords.append(load_weights_kw)

            # load_first_inputs
            load_inputs_kw = ast.keyword(arg="load_first_inputs", value=ast.Constant(value=True))
            guard_keywords.append(load_inputs_kw)

            # framework
            framework_kw = ast.keyword(arg="framework", value=ast.Constant(value=self.framework))
            guard_keywords.append(framework_kw)

        if self.align_depth != "inf":
            single_step_kw = ast.keyword(arg="align_depth", value=ast.Constant(value=self.align_depth))
            guard_keywords.append(single_step_kw)

        # single_step_mode
        if self.single_step_mode is not None:
            single_step_kw = ast.keyword(arg="single_step_mode", value=ast.Constant(value=self.single_step_mode))
            guard_keywords.append(single_step_kw)

        # base_dump_path
        if self.mode == "align" or self.single_step_mode is not None:
            base_dump_path_kw = ast.keyword(arg="base_dump_path", value=ast.Constant(value=self.alignment_dir))
            guard_keywords.append(base_dump_path_kw)

        # name
        name_kw = ast.keyword(arg="name", value=ast.Constant(value=self.padiff_model_name))
        guard_keywords.append(name_kw)

        # max_calls
        max_calls_kw = ast.keyword(arg="max_calls", value=ast.Constant(value=1))
        guard_keywords.append(max_calls_kw)

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

    def add_dump_report(self, node):
        # Temporarily abandoned
        dump_stmt = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="dump_report", ctx=ast.Load()),
                args=[
                    ast.Name(id="_padiff_proxy_model", ctx=ast.Load()),
                    ast.Attribute(
                        value=ast.Name(id="_padiff_proxy_model", ctx=ast.Load()), attr="dump_path", ctx=ast.Load()
                    ),
                ],
                keywords=[],
            )
        )
        ast.copy_location(dump_stmt, node)
        ast.fix_missing_locations(dump_stmt)
        return dump_stmt


def create_injected_script(
    src_script_path: str,
    framework: str,
    src_model_name: str = "model",
    mode: str = "base",
    alignment_dir: str = None,
    single_step_mode: str = None,
    align_depth: str = "inf",
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
        src_model_name=src_model_name,
        mode=mode,
        alignment_dir=alignment_dir,
        single_step_mode=single_step_mode,
        align_depth=align_depth,
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
