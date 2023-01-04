# PaDiff
**P**addle **Auto**matically **Diff** precision toolkits.


## 简介
PaDiff是基于PaddlePaddle与PyTorch的模型精度对齐工具。传入Paddle与Torch模型，PaDiff将对训练过程中的所有中间结果以及训练后的模型权重进行对齐检查，并以调用栈的形式提示模型第一次出现精度diff的位置。


## 安装
PaDiff v0.1 版本将于近期发布，届时可通过如下命令安装：
```
pip install padiff
```

尝鲜版或开发者推荐如下命令安装：
```
pip install -e .
```
## 使用说明

### auto_diff 使用接口与参数说明

接口函数签名：`auto_diff(layer, module, example_inp, auto_weights=True, options={})`

-   layer：传入paddle模型

-   module：传入torch模型

-   inp：传入输入数据

-   auto_weights: 是否使用随机数值统一初始化paddle与torch模型，默认为True

-   options：一个传递参数的字典

       -   "atol": 精度对齐的误差上限

       -   "compare_mode": 精度对齐模式，默认值为"mean"，表示使用Tensor间误差的均值作为对齐标准，要求 mean(a-b) < atol；另一个可选项为"strict"，表示对Tensor进行逐数据（Elementwise）的对齐检查

### 注意事项与用例代码：

-   在使用auto_diff时，需要传入paddle模型与torch模型，在模型定义时，需要将forward中所使用的子模型在`__init__`函数中定义，并保证其中的子模型定义顺序一致，具体可见下方示例代码

```py
from padiff import auto_diff
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
auto_diff(layer, module, inp, auto_weights=True, options={'atol': 1e-4, 'compare_mode': 'strict'})
```

## 输出信息示例

-   正确对齐时的输出信息：
    auto_diff将输出paddle与torch模型输出结果之间的最大diff值

       ```
       [AutoDiff] Start auto_diff, may need a while to generate reports...
       [AutoDiff] Max output diff is 6.103515625e-05

       [AutoDiff] weight and weight.grad is compared.
       [AutoDiff] forward 4 steps compared.
       [AutoDiff] bacward 4 steps compared.
       [AutoDiff] SUCCESS !!!
       ```

-   模型对齐失败时的输出信息：

       -   训练后，模型权重以及梯度的对齐情况，具体信息将记录在当前路径的diff_log文件夹下
       -   注意，每次调用auto_diff后，diff_log下的报告会被覆盖
       -   在训练过程中首先出现diff的位置（在forward过程或backward过程）
       -   paddle与torch的调用栈，可以追溯到第一次出现不对齐的代码位置

       ```
       [AutoDiff] Start auto_diff, may need a while to generate reports...
       [AutoDiff] Max output diff is 3.0571913719177246

       [AutoDiff] Differences in weight or grad !!!
       [AutoDiff] Check reports at `/workspace/diff_log`

       [AutoDiff] FAILED !!!
       [AutoDiff]     Diff found in `Forward  Stagy` in step: 0, net_id is 1 vs 1
       [AutoDiff]     Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>


       Paddle Stacks:
       =========================
              ...
              File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1022    __call__
                     return self._dygraph_call_func(*inputs, **kwargs)
              File pptest.py: 37    forward
                     x = self.linear1(x)
              ...
       Torch  Stacks:
       =========================
              ...
              File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1151    _call_impl
                     hook_result = hook(self, input, result)
              File pptest.py: 58    forward
                     x = self.linear1(x)
              ...
       ```

-   模型对齐失败且失败位置在反向过程时：
    结合输出文本 " [AutoDiff]     Diff found in `Backward Stagy` in step: 0, net_id is 2 vs 2 "，可知模型前向能够对齐，但是反向过程出现diff。结合调用栈信息可以发现，diff出现在linear2对应的反向环节出现diff


       ```
       [AutoDiff] Start auto_diff, may need a while to generate reports...
       [AutoDiff] Max output diff is 1.71661376953125e-05

       [AutoDiff] Differences in weight or grad !!!
       [AutoDiff] Check reports at `/workspace/diff_log`

       [AutoDiff] forward 4 steps compared.
       [AutoDiff] FAILED !!!
       [AutoDiff]     Diff found in `Backward Stagy` in step: 0, net_id is 2 vs 2
       [AutoDiff]     Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>


       Paddle Stacks:
       =========================
              ...
              File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1022    __call__
                     return self._dygraph_call_func(*inputs, **kwargs)
              File pptest.py: 52    forward
                     x3 = self.linear2(x)
              ...
       Torch  Stacks:
       =========================
              ...
              File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1151    _call_impl
                     hook_result = hook(self, input, result)
              File pptest.py: 66    forward
                     x3 = self.linear2(x)
              ...
       ```
## 调试建议

如果遇到了 auto_diff 函数提示某个 layer 没有对齐，可以考虑如下几个 debug 建议：

- 如果报告不是上述的Success或者是Failed，那么说明模型没有满足预定的假设。可以结合 报错信息 进行分析。常见问题是：Torch 模型和 Paddle 模型没有满足Layer定义的一一对应假设。可以通过 print 两个模型来进行假设验证，一个满足一一对应的例子应该如下图（Layer的名字可以不用相同）![e11cd8bfbcdaf5e19a3894cecd22d212](https://user-images.githubusercontent.com/16025309/209917443-e5c21829-f4a6-4bdf-a621-b123c11e83d6.jpg)


- 如果显示精度有diff，先分析Paddle和Torch的调用栈，找到对应的源码并分析他们在逻辑上是否是对应的Layer，如果不是对应的Layer，那么说明 Torch 模型和 Paddle 模型没有满足Layer定义的一一对应假设。如图 <img width="875" alt="3d569899c42f69198f398540dec89012" src="https://user-images.githubusercontent.com/16025309/209917231-717c8e88-b3d8-41bc-b6a9-0330d0d9ed50.png">

- 如果不是上述的问题，那么可以考虑进行debug，比如构造最小复现样例或者是pdb调试等等。

- 如果上述无法解决您的问题，或者认为找不到问题，可以考虑给本仓库提一个Issue。
