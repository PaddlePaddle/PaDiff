- [auto\_diff 接口 Tutorial](#auto_diff-接口-tutorial)
  - [一、 使用方法](#一-使用方法)
  - [二、阅读输出信息](#二阅读输出信息)
    - [2.1 正确对齐时的输出信息](#21-正确对齐时的输出信息)
    - [2.2 模型权重拷贝失败时的报错信息](#22-模型权重拷贝失败时的报错信息)
    - [2.3 模型前反向对齐失败时的输出信息](#23-模型前反向对齐失败时的输出信息)
    - [2.4 模型weight/grad对齐失败的报错信息](#24-模型weightgrad对齐失败的报错信息)
  - [三、使用loss \& optimizer](#三使用loss--optimizer)
    - [3.1 使用loss](#31-使用loss)
    - [3.2 使用optimizer](#32-使用optimizer)
  - [四、使用assign\_weight](#四使用assign_weight)
- [全局变量开关](#全局变量开关)
    - [API 级别的对齐检查](#api-级别的对齐检查)
    - [略过 wrap\_layer](#略过-wrap_layer)
    - [在前反向对齐时，打印子模型python路径](#在前反向对齐时打印子模型python路径)

# auto_diff 接口 Tutorial

## 一、 使用方法

> `auto_diff` 详细参数解释详见：[接口信息](Interfaces.md)

使用 `padiff` 进行模型对齐检查有几个基本的步骤：

1.   分别构造两个待对齐的 paddle 或 torch 模型
2.   分别构造两个模型的输入数据
3.   调用 `auto_diff` API 接口

以下是一段使用 padiff 工具进行对齐的完整代码 (以对齐 paddle 模型和 torch 模型为例)

> 注意：在模型定义时，需要将forward中所使用的子模型在  `__init__`  函数中定义，并保证其中的子模型定义顺序一致**，具体可见下方示例代码

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


module = SimpleModule()
layer = SimpleLayer()

inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({"x": torch.as_tensor(inp)},
     {"x": paddle.to_tensor(inp)})

auto_diff(module, layer, inp, atol=1e-4, compare_mode="strict", single_step=False)
```



## 二、阅读输出信息

padiff 的工作可以分为几个阶段，在发生错误时，需要首先判断在哪个阶段发生了错误

1.   权重拷贝阶段（当设置参数 `auto_weights` 为 `True` 时）
2.   模型前反向对齐阶段
3.   模型权重&梯度对齐阶段

当 padiff 进行多个 step 的对齐检查时，以上2、3阶段循环执行

下面介绍正确对齐，以及在不同阶段产生错误时的输出信息。



### 2.1 正确对齐时的输出信息

```bash
[AutoDiff] Your options:
{
  atol: `0.0001`
  auto_init: `True`
  single_step: `False`
  use_loss: `False`
  use_opt: `False`
  rtol: `1e-07`
  compare_mode: `mean`
}
[AutoDiff] Assign weight success !!!
[AutoDiff] check cfg {'atol': 0.0001, 'rtol': 1e-07, 'compare_mode': 'mean'}
[AutoDiff] Checking report in /workspace/PaDiff/padiff_dump/SimpleModule(base_model)/auto_diff and /workspace/PaDiff/padiff_dump/SimpleLayer(raw_model)/auto_diff
[AutoDiff] Check grads cfg: {'atol': 0.0001, 'rtol': 1e-07, 'compare_mode': 'mean'}
[AutoDiff] Checking grads in /workspace/PaDiff/padiff_dump/SimpleModule(base_model)/auto_diff and /workspace/PaDiff/padiff_dump/SimpleLayer(raw_model)/auto_diff
[AutoDiff] grads compared.
[AutoDiff] SUCCESS !!!
```



### 2.2 模型权重拷贝失败时的报错信息

当看到 `Assign weight Failed` ，说明权重拷贝出现了问题，并在下文中附上具体的错误信息
-  在拷贝权重过程中，没有 parameter，或被 LayerMap 指定的 layer/module， 会被标注上 (skip)
-  可以通过设置环境变量 `export PADIFF_PATH_LOG=ON` 在 log 信息中添加 layer/module 的具体路径

```bash
[AutoDiff] Your options:
{
  atol: `0.0001`
  auto_init: `True`
  single_step: `False`
  use_loss: `False`
  use_opt: `False`
  rtol: `1e-07`
  compare_mode: `mean`
}
[AutoDiff] Assign weight Failed !!!

RuntimeError:  Error occured when trying init weights, between:
    base_model: `Linear(in_features=100, out_features=4, bias=True)`
                `SimpleModule(base_model).linear1.weight`
    raw_model: `Linear(in_features=100, out_features=10, dtype=float32)`
               `SimpleLayer(raw_model).linear1.weight`
AssertionError:  Shape of param `weight` in torch::Linear and param `weight` in paddle::Linear is not the same. [4, 100] vs [10, 100]

Weight init log saved to
    /workspace/PaDiff/padiff_log/weight_init_SimpleModule(base_model).log
    /workspace/PaDiff/padiff_log/weight_init_SimpleLayer(raw_model).log

Please view the reports and checkout the layer marked with `<---  *** HERE ***` !
Hint:
    1. Check the definition order of params is same in submodels.
    2. Check the corresponding submodel have the same style:
       param <=> param, buffer <=> buffer, embedding <=> embedding ...
       cases like param <=> buffer, param <=> embedding are not allowed.
    3. If can not change model codes, try to use a `LayerMap`
       which can solve most problems.
    4. (skip) means this layer is skipped because it is under black_list, or it has no param.
    0. Visit `https://github.com/PaddlePaddle/PaDiff` to find more infomation.
```

其中打印的 log 信息为：
```
# /workspace/PaDiff/padiff_log/weight_init_SimpleModule(base_model).log

SimpleModule(base_model)
========================================
    SimpleModule
     +--- Linear    <---  *** HERE ***
```
```
# /workspace/PaDiff/padiff_log/weight_init_SimpleLayer(raw_model).log

SimpleLayer(raw_model)
========================================
    SimpleLayer
     +--- Linear    <---  *** HERE ***
```

可能的问题有：

1.   子模型/权重定义顺序不对齐 => 修改代码对齐，或使用 `LayerMap` 指定，
2.   子模型的 paddle 与 torch 实现方式不一致（权重等对不齐）=> 使用 `LayerMap` 指定

> 注：LayerMap 的使用方式详见：[LayerMap使用说明](SpecialInit.md)

若不使用 padiff 的权重初始化功能，可以避免此类错误，但在权重与梯度检查时会遇见同样的问题


### 2.3 模型前反向对齐失败时的输出信息

1.   指明 diff 出现的阶段：`Forward Stage` or `Backward Stage`，该信息出现在日志的开头
2.   打印出现精度 diff 时的比较信息，包括绝对误差和相对误差数值
3.   打印模型结构，并用括号标注结点类型，用`<---  *** HERE ***`指示出现diff的位置（log将输出到文件中）

定位精度误差位置后，可进行验证排查：

```bash
[AutoDiff] Your options:
{
  atol: `0.0001`
  auto_init: `False`
  single_step: `False`
  use_loss: `False`
  use_opt: `False`
  rtol: `1e-07`
  compare_mode: `mean`
}
[AutoDiff] check cfg {'atol': 0.0001, 'rtol': 1e-07, 'compare_mode': 'mean'}
[AutoDiff] Checking report in /workspace/PaDiff/padiff_dump/SimpleModule(base_model)/auto_diff and /workspace/PaDiff/padiff_dump/SimpleLayer(raw_model)/auto_diff
[AutoDiff] FAILED !!!
[AutoDiff]     Diff found in Forward Stage
[AutoDiff]     Type of layer is: Linear vs Linear
[AutoDiff]     Route: SimpleModule(base_model).linear1
[AutoDiff]            SimpleLayer(raw_model).linear1

AssertionError:
Not equal to tolerance rtol=1e-07, atol=0.0001

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 0.2241705
Max relative difference: 0.8343468
 x: array(0.044507, dtype=float32)
 y: array(0.268678, dtype=float32)

[AutoDiff] Check model struct:
Logs: /workspace/PaDiff/padiff_log/report_SimpleModule(base_model)
      /workspace/PaDiff/padiff_log/report_SimpleLayer(raw_model)

[AutoDiff] The forward stage comparing failed !!!
[AutoDiff] FAILED !!!
```

```
# /workspace/PaDiff/padiff_log/report_SimpleModule(base_model)

SimpleModule(base_model)
========================================
    SimpleModule
     +--- Linear    <---  *** HERE ***
```

```
# /workspace/PaDiff/padiff_log/report_SimpleLayer(raw_model)

SimpleLayer(raw_model)
========================================
    SimpleLayer
     +--- Linear    <---  *** HERE ***
```

### 2.4 模型weight/grad对齐失败的报错信息

由于 `weight/grad` 对齐信息一般比较多，所以会将信息输入到日志文件。日志文件的路径会打印到终端（位于当前目录的 `diff_log` 文件夹下），如下面的例子所示：

```
[AutoDiff] Your options:
{
  atol: `0.0001`
  auto_init: `False`
  use_opt: `True`
  single_step: `False`
  use_loss: `False`
  rtol: `1e-07`
  compare_mode: `mean`
}
[AutoDiff] Check grads cfg: {'atol': 0.0001, 'rtol': 1e-07, 'compare_mode': 'mean'}
[AutoDiff] Checking grads in /workspace/PaDiff/padiff_dump/SimpleModule(base_model)/auto_diff and /workspace/PaDiff/padiff_dump/SimpleLayer(raw_model)/auto_diff
[AutoDiff] grads compared.
[AutoDiff] Check weights cfg: {'atol': 0.0001, 'rtol': 1e-07, 'compare_mode': 'mean'}
[AutoDiff] Checking weights in /workspace/PaDiff/padiff_dump/SimpleModule(base_model)/auto_diff and /workspace/PaDiff/padiff_dump/SimpleLayer(raw_model)/auto_diff
[AutoDiff] Diff found when compare weights, please check report
        /workspace/PaDiff/padiff_log/weights_diff
[AutoDiff] FAILED !!!
```

在日志文件中，将记录出现diff的权重路径以及比较信息（对每一处diff都会记录一组信息），例如：

-   当检查到weight或grad存在diff，可能是反向计算出现问题，也可能是Loss function 或 optimizer出现问题（若传入了loss以及optimizer）

```
=========================
weights value is different.
between base_model: Linear(in_features=100, out_features=10, bias=True)
        raw_model:  Linear(in_features=100, out_features=10, dtype=float32)

base_model param path:
    SimpleModule(base_model).linear1.weight
raw_model param path:
    SimpleLayer(raw_model).linear1.weight

AssertionError:
Not equal to tolerance rtol=1e-07, atol=0.0001

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 0.00137274
Max relative difference: 0.72396755
 x: array(-0.000523, dtype=float32)
 y: array(-0.001896, dtype=float32)
```



## 三、使用loss & optimizer

### 3.1 使用loss

能够向 padiff 工具传入自定义的 `loss_fn`，并参与对齐。但传入的 loss 函数有一定限制

须知：

1.   传入的 `loss_fn` 是一个可选项，不指定 `loss_fn` 时，将使用 `auto_diff` 内置的一个 `fake loss function` 进行计算，该函数将 output 整体求平均值并返回。
2.   **`loss_fn` 只接受一个输入（即model的output），并输出一个scale tensor**。无法显式传入label，但可以通过 lambda 或者闭包等方法间接实现。
3.   `loss_fn` 也可以是一个 model ，但是 `loss_fn` 内部的逻辑将不会参与对齐检查， padiff 只会检查 `loss_fn` 的输出是否对齐

> **注：** 利用 `partial` 绑定 label 是一种简单的构造 `loss_fn` 的方法，使用时需注意，必须将参数名与参数值进行绑定，否则可能在传参时错位

```py
class SimpleLayer(paddle.nn.Layer):
    # ...
class SimpleModule(torch.nn.Module):
    # ...

layer = SimpleLayer()
module = SimpleModule()

inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})

label = paddle.rand([10]).numpy().astype("float32")

# 自定义loss函数，若输入不止一个，可以使用partial或者闭包等方法得到单输入的函数，再传入
def paddle_loss(inp, label):
    label = paddle.to_tensor(label)
    return inp.mean() - label.mean()

def torch_loss(inp, label):
    label = torch.tensor(label)
    return inp.mean() - label.mean()

auto_diff(module, layer, inp, auto_init=True, atol=1e-4, loss_fn=[
    partial(torch_loss, label=label)
    partial(paddle_loss, label=label),
])

# 使用 paddle 和 torch 提供的损失函数时，使用方法一致
paddle_mse = paddle.nn.MSELoss()
torch_mse = torch.nn.MSELoss()

auto_diff(module, layer, inp, auto_init=True, atol=1e-4, loss_fn=[
    partial(torch_mse, target=torch.tensor(label))
    partial(paddle_mse, label=paddle.to_tensor(label)),
])
```



### 3.2 使用optimizer

能够向 padiff 工具传入 `optimizers`，在多 step 对齐下，将使用 `optimizers` 更新模型

须知：

1.   `optimizers` 是可选的，若不传入，padiff 并不提供默认的 `optimzers` ，将跳过权重更新的步骤
2.   padiff 不会检查 `optimizers` 内部是否对齐，但是会检查 step 后的 grad 是否对齐
3.   `optimizer` 有两种使用方式：
     - 依次传入一组 `paddle.optimizer.Optimizer` 或 `torch.optim.Optimizer` 类型的 optimizers
     - 依次传入两个**无输入的 lambda**，分别负责 paddle 模型与 torch 模型的权重更新，可在其中实现自定义操作

```py
class SimpleLayer(paddle.nn.Layer):
    # ...
class SimpleModule(torch.nn.Module):
    # ...

layer = SimpleLayer()
module = SimpleModule()

inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})

paddle_opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=layer.parameters())
torch_opt = torch.optim.Adam(lr=0.001, params=module.parameters())

auto_diff(
    module,
    layer,
    inp,
    auto_init=True,
    atol=1e-4,
    optimizers=[torch_opt, paddle_opt],
)
```



## 四、使用assign_weight

`assign_weight` 用于复制 torch 模型的权重到 paddle 模型，具体接口参数信息见：[接口信息](Interfaces.md)，关于权重初始化的高级设置见 [特殊初始化](SpecialInit.md)

`assign_weight` 的逻辑以及报错信息与 `auto_diff` 开启 `auto_weight` 选项是一致的，因此可以参考上文

须知：

-    如果 `assign_weight` 失败，则函数的返回值为 `False`（不会抛出异常）

```py
import os
os.environ["PADIFF_API_CHECK"] = "OFF"

from padiff import assign_weight, LayerMap
import torch
import paddle

layer = SimpleLayer()
module = SimpleModule()

assign_weight(layer, module)
```




# 全局变量开关

### API 级别的对齐检查

目前 PaDiff 工具默认关闭 API 级别的对齐检查

设置环境变量可以打开该功能： `export PADIFF_API_CHECK=ON`

### 添加自定义算子

当`PADIFF_API_CHECK`开启时，可以添加PD_BUILD_OP方式注册的自定义算子，该功能通过`PADDLE_THIRDPARTY_API`环境变量开启

设置时，需将自定义算子的module、api名写全，多个算子之间用逗号分隔，如：

`export PADDLE_THIRDPARTY_API=paddle3d.ops.iou3d_nms,paddle3d.ops.hard_voxelize,paddle_xpu_nn.xpu_rms_norm`

### 略过 wrap_layer

`export PADIFF_SIKP_WRAP_LAYER=TRUE`

可以略过没有 parameter 的 sublayer，默认是不会略过的

### 在前反向对齐时，打印子模型python路径

`export PADIFF_PATH_LOG=ON`

可以在打印的日志信息中，额外打印 python 路径，例如 Model.submodel.linear1
为了日志信息的简洁，这个开关默认关闭
