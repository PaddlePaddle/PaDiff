# Tutorial



## 使用方法

auto_diff的具体接口参数信息见：[接口信息]()

使用padiff进行模型对齐检查有几个基本的步骤

1.   分别构造 paddle 和 torch 模型
2.   分别构造两个模型的输入数据
3.   调用auto_diff

以下是一段使用padiff工具进行对齐的完整代码，值得注意的是：<font color="red">在模型定义时，需要将forward中所使用的子模型在  `__init__`  函数中定义，并保证其中的子模型定义顺序一致</font>，具体可见下方示例代码

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



## 阅读输出信息

padiff的工作可以分为几个阶段，在发生错误时，需要首先判断在哪个阶段发生了错误

1.   权重拷贝阶段（当设置参数 auto_weights 为 True时）
2.   模型前反向对齐阶段
3.   模型权重&梯度对齐阶段

当padiff进行多个step的对齐检查时，以上2、3阶段循环执行

下面介绍正确对齐，以及在不同阶段产生错误时的输出信息



### 正确对齐时的输出信息

```
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



### 模型权重拷贝失败时的报错信息

当看到`Assign weight Failed`，说明权重拷贝出现了问题，并在下文中附上具体的错误信息

可能的问题有：

1.   子模型/权重定义顺序不对齐 => 修改代码对齐，或使用LayerMap指定
2.   子模型的paddle与torch实现方式不一致（权重等对不齐）=> 使用LayerMap指定

LayerMap的使用方式详见[]()

若不使用padiff的权重初始化功能，可以避免此类错误，但在权重与梯度检查时，会遇见同样的问题

```
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
    paddle: `Linear(in_features=100, out_features=100, dtype=float32)` parameter `weight`
    torch: `Linear(in_features=100, out_features=10, bias=True)` parameter `weight`

Shape of paddle param `weight` and torch param `weight` is not the same. [100, 100] vs [100, 10]

Torch Model
=========================
    SimpleModule
     |--- Linear
     +--- Linear    <---  *** HERE ***
Paddle Model
=========================
    SimpleLayer
     |--- Linear
     +--- Linear    <---  *** HERE ***

Hint:
      1. check the init order of param or layer in definition is the same.
      2. try to use `LayerMap` to skip the diff in models, you can find the instructions at `https://github.com/PaddlePaddle/PaDiff`.
```



### 模型前反向对齐失败时的输出信息

-   指明diff出现的阶段：`Forward Stage` or `Backward Stage`，该信息出现在日志的开头
-   打印出现精度diff时的比较信息，包括绝对误差和相对误差数值
-   打印模型结构，并用`<---  *** HERE ***`标注出现diff的位置（log过长时将输出到文件中）
-   打印调用栈信息，帮助定位到具体的代码位置

定位精度误差位置后，可进行验证排查

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
[AutoDiff] Max elementwise output diff is 3.9604315757751465
[AutoDiff] FAILED !!!
[AutoDiff]     Diff found in `Forward  Stage` in step: 0, net_id is 1 vs 1
[AutoDiff]     Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>

Not equal to tolerance rtol=1e-07, atol=0.0001

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 0.04014074
Max relative difference: 0.69478023
 x: array(0.017634, dtype=float32)
 y: array(0.057775, dtype=float32)


[AutoDiff] Check model struct:
Paddle Model
=========================
    (net) SimpleLayer
     |--- (net) Linear    <---  *** HERE ***
     +--- (net) Linear
Torch Model
=========================
    (net) SimpleModule
     |--- (net) Linear    <---  *** HERE ***
     +--- (net) Linear


Paddle Stacks:
=========================
         ...
         File tests/test_simplenet1.py: 37    forward
                x = self.linear1(x)
         File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 993    _dygraph_call_func
                outputs = self.forward(*inputs, **kwargs)
         ...

Torch  Stacks:
=========================
         ...
         File tests/test_simplenet1.py: 58    forward
                x = self.linear1(x)
         File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1208    _call_impl
                result = forward_call(*input, **kwargs)
         ...

[AutoDiff] FAILED !!!
```



### 模型weight/grad对齐失败的报错信息

由于weight/grad对齐信息一般比较多，所以会将信息输入到log文件，并输出log文件路径，请打开打印的文件路径以查看具体信息

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



## 使用loss & optimizer

### 使用loss

能够向padiff工具传入自定义的loss_fn，并参与对齐。但传入的loss函数有一定限制

须知：

1.   传入的loss_fn是一个可选项，不指定loss_fn时，将使用auto_diff内置的一个fake loss function进行计算
2.   <font color="red">loss_fn 只接受一个输入（即model的output），并输出一个scale tensor</font>。无法显式传入label，但可以通过lambda或者闭包等方法间接实现。
3.   loss_fn 也可以是一个model，但是loss_fn内部的逻辑将不会参与对齐检查，padiff只会检查loss_fn的输出是否对齐

注意事项：

1.   利用partial绑定label是一种简单的构造loss_fn的方法，使用时需注意，必须将参数名与参数值进行绑定，否则可能在传参时错位

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



### 使用optimizer

能够向padiff工具传入optimizer，在多step对齐下，将使用optimizer更新模型

须知：

1.   optimizer是可选的，若不传入，padiff并不提供默认的optimzer，将跳过权重更新的步骤
2.   若需要进行多step对齐，必须传入optimizer
3.   padiff不会检查optimizer内部是否对齐，但在多step下会检查模型权重（受optimizer影响）
4.   optimizer有两种使用方式：传入一个正常的optimizer（支持clear grad以及step），或者传入一个<font color="red">无输入的lambda</font>，并在其中实现自定义操作

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



## 使用assign_weight

assign_weight 用于复制 torch 模型的权重到 paddle 模型，具体接口参数信息见：[接口信息]()

assign_weight的逻辑以及报错信息与 auto_diff 开启 auto_weight 选项是一致的，因此可以参考上文

须知：

-   注意，这个函数不会raise，只会return False并在终端打印信息
-   由于 padiff 的 api 级别对齐机制目前默认开启，若仅仅需要将assign_weight作为一个辅助其他任务的工具使用，请设置环境变量以关闭api级别对齐机制 `export PADIFF_API_CHECK=OFF`

```py
from padiff import assign_weight, LayerMap
import torch
import paddle

layer = SimpleLayer()
module = SimpleModule()
layer_map = LayerMap()

assign_weight(layer, module, layer_map)
```




## 使用API级别的对齐检查

目前PaDiff工具已默认开启API级别的对齐检查

设置环境变量可以关闭该功能： `export PADIFF_API_CHECK=OFF`
