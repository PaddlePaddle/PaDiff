- [Tutorial](#tutorial)
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
  - [五、 使用API级别的对齐检查](#五-使用api级别的对齐检查)

# Tutorial

## 一、 使用方法

> `auto_diff` 详细参数解释详见：[接口信息](Interfaces.md)

使用 `padiff` 进行模型对齐检查有几个基本的步骤：- [Tutorial](#tutorial)

1.   分别构造 paddle 和 torch 模型
2.   分别构造两个模型的输入数据
3.   调用 `auto_diff` API 接口

以下是一段使用 padiff 工具进行对齐的完整代码，

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
inp = ({'x': paddle.to_tensor(inp)},  ## <-- 注意顺序，paddle_input, torch_input 的形式。
     {'y': torch.as_tensor(inp) })

auto_diff(layer, module, inp, auto_weights=True, options={'atol': 1e-4, 'rtol':0, 'compare_mode': 'strict', 'single_step':False})
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
  rtol: `1e-07`
  diff_phase: `both`
  compare_mode: `mean`
  single_step: `False`
}
[AutoDiff] Assign weight success !!!
[AutoDiff] =================Train Step 0=================
[AutoDiff] Max elementwise output diff is 4.172325134277344e-07
[AutoDiff] forward stage compared.
[AutoDiff] backward stage compared.
[AutoDiff] weight and grad compared.
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
  rtol: `1e-07`
  diff_phase: `both`
  compare_mode: `mean`
  single_step: `False`
}
[AutoDiff] Assign weight Failed !!!

Error occured between:
    paddle: `Linear(in_features=100, out_features=100, dtype=float32)`
            `SimpleLayerDiff.linear2.weight`
    torch: `Linear(in_features=100, out_features=10, bias=True)`
           `SimpleModule.linear2.weight`

Shape of paddle param `weight` and torch param `weight` is not the same. [100, 100] vs [100, 10]

Torch Model
=========================
    SimpleModule  (skip)
     |--- Linear
     +--- Linear    <---  *** HERE ***
Paddle Model
=========================
    SimpleLayer  (skip)
     |--- Linear
     +--- Linear    <---  *** HERE ***

NOTICE: layer/module will be marked with `(skip)` for:
    1. This layer/module is contained by layer_map.
    2. This layer/module has no parameter, so padiff think it is a wrap layer.

Hint:
    1. Check the definition order of params in layer/module is the same.
    2. Check the corresponding layer/module have the same style:
       param <=> param, buffer <=> buffer, embedding <=> embedding ...
       cases like param <=> buffer, param <=> embedding are not allowed,
       because padiff can not know how to init the parameters.
    3. If you can not change model codes, try to use a `LayerMap`
       which can solve almost any problem.
    0. Visit `https://github.com/PaddlePaddle/PaDiff` to find more infomation !!!
```

可能的问题有：

1.   子模型/权重定义顺序不对齐 => 修改代码对齐，或使用 `LayerMap` 指定
2.   子模型的 paddle 与 torch 实现方式不一致（权重等对不齐）=> 使用 `LayerMap` 指定

> 注：LayerMap 的使用方式详见：[LayerMap使用说明](LayerMap.md)

若不使用 padiff 的权重初始化功能，可以避免此类错误，但在权重与梯度检查时，会遇见同样的问题


### 2.3 模型前反向对齐失败时的输出信息

1.   指明 diff 出现的阶段：`Forward Stage` or `Backward Stage`，该信息出现在日志的开头
2.   打印出现精度 diff 时的比较信息，包括绝对误差和相对误差数值
3.   打印模型结构，并用括号标注结点类型，用`<---  *** HERE ***`指示出现diff的位置（log过长时将输出到文件中）
4.   打印调用栈信息，帮助定位到具体的代码位置

定位精度误差位置后，可进行验证排查：

```bash
[AutoDiff] Your options:
{
  atol: `0.0001`
  steps: `1`
  rtol: `1e-07`
  compare_mode: `mean`
  single_step: `False`
  use_loss: `False`
  use_opt: `False`
}
[AutoDiff] =================Train Step 0=================
[AutoDiff] Max elementwise output diff is 4.575464248657227
[AutoDiff] FAILED !!!
[AutoDiff]     Diff found in `Forward  Stage` in step: 0, net_id is -1 vs -1
[AutoDiff]     Type of layer is  : <class 'padiff.wrap_func.<locals>.wrapped.<locals>.TorchApi'> vs <class 'padiff.wrap_func.<locals>.wrapped.<locals>.PaddleApi'>

Not equal to tolerance rtol=1e-07, atol=0.0001

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 0.0622915
Max relative difference: 1.6068412
 x: array(0.023525, dtype=float32)
 y: array(-0.038766, dtype=float32)


[AutoDiff] Check model struct:
Paddle Model
=========================
    (net) SimpleLayerDiff
     |--- (net) Linear
     |     +--- (api) paddle.nn.functional.linear    <---  *** HERE ***
     |--- (net) Linear
     |     +--- (api) paddle.nn.functional.linear
     +--- (net) Linear
           +--- (api) paddle.nn.functional.linear
Torch Model
=========================
    (net) SimpleModule
     |--- (net) Linear
     |     +--- (api) torch.nn.functional.linear    <---  *** HERE ***
     |--- (net) Linear
     |     +--- (api) torch.nn.functional.linear
     +--- (net) Linear
           +--- (api) torch.nn.functional.linear


Paddle Stacks:
=========================
         ...
         File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/nn/layer/common.py: 175    forward
                x=input, weight=self.weight, bias=self.bias, name=self.name
         File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 997    _dygraph_call_func
                outputs = self.forward(*inputs, **kwargs)
         ...
Torch  Stacks:
=========================
         ...
         File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/linear.py: 94    forward
                return F.linear(input, self.weight, self.bias)
         File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 889    _call_impl
                result = self.forward(*input, **kwargs)
         ...

[AutoDiff] FAILED !!!
```



### 2.4 模型weight/grad对齐失败的报错信息

由于 `weight/grad` 对齐信息一般比较多，所以会将信息输入到日志文件。日志文件的路径会打印到终端（位于当前目录的 `diff_log` 文件夹下），如下面的例子所示：

```
[AutoDiff] Your options:
{
  atol: `0.0001`
  rtol: `1e-07`
  diff_phase: `both`
  compare_mode: `mean`
  single_step: `False`
}
[AutoDiff] =================Train Step 0=================
[AutoDiff] Max elementwise output diff is 2.9132912158966064
[AutoDiff] Diff found in model weights, check report `/workspace/PaDiff/tests/diff_log/weight_diff.log`.
[AutoDiff] Diff found in model grad, check report `/workspace/PaDiff/tests/diff_log/grad_diff.log`.
[AutoDiff] FAILED !!
```

在日志文件中，将记录出现diff的权重路径以及比较信息（对每一处diff都会记录一组信息），例如：

-   当检查到weight或grad存在diff，可能是反向计算出现问题，也可能是Loss function 或 optimizer出现问题（若传入了loss以及optimizer）

```
=========================
After training, grad value is different.
between paddle: `Linear(in_features=100, out_features=100, dtype=float32)`, torch: `Linear(in_features=100, out_features=100, bias=True)`
paddle path:
    SimpleLayerDiff.linear2.bias
torch path:
    SimpleModule.linear2.bias

Not equal to tolerance rtol=1e-07, atol=0.0001

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 0.00999998
Max relative difference: 0.9999987
 x: array(0.02, dtype=float32)
 y: array(0.01, dtype=float32)
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

auto_diff(layer, module, inp, auto_weights=True, options={"atol": 1e-4}, loss_fn=[
     partial(paddle_loss, label=label),
     partial(torch_loss, label=label)
])

# 使用 paddle 和 torch 提供的损失函数时，使用方法一致
paddle_mse = paddle.nn.MSELoss()
torch_mse = torch.nn.MSELoss()

auto_diff(layer, module, inp, auto_weights=True, options={"atol": 1e-4}, loss_fn=[
     partial(paddle_mse, label=paddle.to_tensor(label)),
     partial(torch_mse, target=torch.tensor(label))
])
```



### 3.2 使用optimizer

能够向 padiff 工具传入 `optimizer`，在多 step 对齐下，将使用 `optimizer` 更新模型

须知：

1.   `optimizer` 是可选的，若不传入，padiff 并不提供默认的 `optimzer` ，将跳过权重更新的步骤
2.   若需要进行多 step 对齐，必须传入 `optimizer`（若不传入，step 会被自动重置为1）
3.   padiff 不会检查 `optimizer` 内部是否对齐，但在多 step 下会检查模型权重（受 `optimizer` 影响）
4.   `optimizer` 有两种使用方式：
     - 依次传入一组 `paddle.optimizer.Optimizer` 和 `torch.optim.Optimizer`
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
    layer,
    module,
    inp,
    auto_weights=True,
    steps=10,
    options={"atol": 1e-4},
    optimizer=[paddle_opt, torch_opt],
)
```



## 四、使用assign_weight

`assign_weight` 用于复制 torch 模型的权重到 paddle 模型，具体接口参数信息见：[接口信息](Interfaces.md)

`assign_weight` 的逻辑以及报错信息与 `auto_diff` 开启 `auto_weight` 选项是一致的，因此可以参考上文

须知：

1.   如果 `assign_weight` 失败，则函数的返回值为 `False`（不会抛出异常）
2.   如果只使用 `assign weight` 接口，不使用 `auto_diff` 接口，请设置环境变量 `export PADIFF_API_CHECK=OFF`

```py
import os
os.environ["PADIFF_API_CHECK"] = "OFF"

from padiff import assign_weight, LayerMap
import torch
import paddle

layer = SimpleLayer()
module = SimpleModule()
layer_map = LayerMap()

assign_weight(layer, module, layer_map)
```




## 五、 使用API级别的对齐检查

目前 PaDiff 工具已默认开启 API 级别的对齐检查

设置环境变量可以关闭该功能： `export PADIFF_API_CHECK=OFF`
