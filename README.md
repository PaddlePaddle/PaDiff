# padiff
**P**addle and **P**ytorch model **Auto**matically **Diff** precision tools.

##

## 使用说明

-   autodiff使用接口与参数说明

    接口函数签名：`autodiff(layer, module, example_inp, auto_weights=True, options={})`

    -   layer：传入paddle模型
    -   module：传入torch模型
    -   inp：传入输入数据
    -   auto_weights: 是否使用随机数值统一初始化paddle与torch模型，默认为True
    -   options：一个传递参数的字典，目前支持在字典中传入 'atol' 参数

-   注意事项与用例代码：

    -   在使用autodiff时，需要传入paddle模型与torch模型，在模型定义时，需要将forward中所使用的子模型在`__init__`函数中定义，并保证其中的子模型定义顺序一致，具体可见下方示例代码

```py
from padiff import autodiff
import torch
import paddle

# 使用paddle与torch定义相同结构的模型: SimpleLayer 和 SimpleModule
# 样例模型结构为:
#       x -> linear1 -> x -> relu -> x -> add -> linear2 -> output
#       |                                  |
#       |----------------------------------|

# 注意：两个模型定义顺序都是 linear1 linear2 ReLU，顺序必须对齐，submodule内部的定义也是一样。
class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.linear2 = paddle.nn.Linear(100, 10)
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x

class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)
        self.linear2 = torch.nn.Linear(100, 10)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x


layer = SimpleLayer()
module = SimpleModule()
inp = paddle.rand((100, 100)).numpy().astype("float32")
autodiff(layer, module, inp, auto_weights=True, options={'atol': 1e-4})
```

-   autodiff的输出信息：

    -   当正确对齐时，autodiff将输出paddle与torch模型输出结果之间的最大diff值

        ```
        Max output diff is 6.866455078125e-05
        forward 4 steps compared.
        bacward 4 steps compared.
        SUCCESS !!!
        ```

    -   当模型对齐失败时，将输出：

        -   训练后，模型权重以及梯度的对齐情况，具体信息将记录在当前路径的diff_log文件夹下
            -   注意，每次调用autodiff后，diff_log下的报告会被覆盖
        -   在训练过程中首先出现diff的位置（在forward过程或backward过程）
        -   paddle与torch的调用栈

        ```
        Max output diff is 3.270139455795288

        Differences in weight or grad !!!
        Check reports at `/workspace/diff_log`

        FAILED !!!
            Diff found in `Forward  Stagy` in step: 0, net_id is 1 vs 1
            Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>

        Not equal to tolerance rtol=1e-07, atol=0.0001

        Mismatched elements: 10000 / 10000 (100%)
        Max absolute difference: 2.533712
        Max relative difference: 3093.094
         x: array([[ 0.046956, -0.067461, -0.002674, ...,  0.1111  , -0.086927,
                -0.189089],
               [ 0.038736,  0.078785, -0.224214, ...,  0.343784,  0.014116,...
         y: array([[ 0.106868,  0.03917 , -0.386061, ..., -0.237078, -0.305712,
                 0.551715],
               [ 0.201293,  0.19741 , -0.242584, ..., -0.313976,  0.029708,...


        Paddle Stacks:
        =========================
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/stack_info.py: 37    extract_frame_summary
                        frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/autodiff.py: 74    layer_hook
                        frame_info, frames = extract_frame_summary()
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1005    _dygraph_call_func
                        hook_result = forward_post_hook(self, inputs, outputs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1023    __call__
                        return self._dygraph_call_func(*inputs, **kwargs)
                 File pptest.py: 21    forward
                        x = self.linear1(x)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1002    _dygraph_call_func
                        outputs = self.forward(*inputs, **kwargs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1023    __call__
                        return self._dygraph_call_func(*inputs, **kwargs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/autodiff.py: 54    autodiff
                        paddle_output = layer(paddle_input)
                 File pptest.py: 46    <module>
                        autodiff(layer, module, inp, auto_weights=False, options={'atol': 1e-4})
        Torch  Stacks:
        =========================
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/stack_info.py: 37    extract_frame_summary
                        frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/autodiff.py: 74    layer_hook
                        frame_info, frames = extract_frame_summary()
                 File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1151    _call_impl
                        hook_result = hook(self, input, result)
                 File pptest.py: 36    forward
                        x = self.linear1(x)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1148    _call_impl
                        result = forward_call(*input, **kwargs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/autodiff.py: 43    autodiff
                        torch_output = module(torch_input)
                 File pptest.py: 46    <module>
                        autodiff(layer, module, inp, auto_weights=False, options={'atol': 1e-4})
        ```

        以下是backward过程中出现diff时的报错信息

        ```
        Max output diff is 1.9073486328125e-05

        Differences in weight or grad !!!
        Check reports at `/workspace/diff_log`

        /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/utils.py:110: UserWarning: Warning: duplicate key is found, use list + pop strategy.
          warnings.warn("Warning: duplicate key is found, use list + pop strategy.")
        forward 4 steps compared.
        FAILED !!!
            Diff found in `Backward Stagy` in step: 0, net_id is 2 vs 2
            Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>

        Not equal to tolerance rtol=1e-07, atol=0.0001

        Mismatched elements: 8800 / 10000 (88%)
        Max absolute difference: 0.00309907
        Max relative difference: 5.7074623
         x: array([[ 0.002019, -0.002684,  0.001096, ...,  0.001875,  0.001338,
                -0.001434],
               [ 0.002019, -0.002684,  0.001096, ...,  0.001875,  0.001338,...
         y: array([[ 0.002247, -0.002733,  0.002497, ...,  0.001356,  0.001161,
                -0.002005],
               [ 0.002247, -0.002733,  0.002497, ...,  0.001356,  0.001161,...


        Paddle Stacks:
        =========================
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/stack_info.py: 37    extract_frame_summary
                        frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/autodiff.py: 74    layer_hook
                        frame_info, frames = extract_frame_summary()
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1005    _dygraph_call_func
                        hook_result = forward_post_hook(self, inputs, outputs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1023    __call__
                        return self._dygraph_call_func(*inputs, **kwargs)
                 File pptest.py: 36    forward
                        x3 = self.linear2(x)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1002    _dygraph_call_func
                        outputs = self.forward(*inputs, **kwargs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1023    __call__
                        return self._dygraph_call_func(*inputs, **kwargs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/autodiff.py: 54    autodiff
                        paddle_output = layer(paddle_input)
                 File pptest.py: 57    <module>
                        autodiff(layer, module, inp, auto_weights=True, options={'atol': 1e-4})
        Torch  Stacks:
        =========================
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/stack_info.py: 37    extract_frame_summary
                        frame_summarys = traceback.StackSummary.extract(traceback.walk_stack(None))
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/autodiff.py: 74    layer_hook
                        frame_info, frames = extract_frame_summary()
                 File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1151    _call_impl
                        hook_result = hook(self, input, result)
                 File pptest.py: 49    forward
                        x3 = self.linear2(x)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1148    _call_impl
                        result = forward_call(*input, **kwargs)
                 File /workspace/env/env3.7/lib/python3.7/site-packages/padiff-0.0.1-py3.7.egg/padiff/autodiff.py: 43    autodiff
                        torch_output = module(torch_input)
                 File pptest.py: 57    <module>
                        autodiff(layer, module, inp, auto_weights=True, options={'atol': 1e-4})
        ```
