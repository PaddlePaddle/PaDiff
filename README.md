# PaDiff

**P**addle  **A**utomatically  **Diff**  precision toolkits.



## 最近更新

-   添加了api级别对齐检查，可以通过设置环境变量来关闭：`export PADIFF_API_CHECK=OFF`
-   更新对齐策略：自顶向下对齐
-   更新模型遍历策略：现在会尽可能滤过wrap layer，大部分情况无需手动调用LayerMap
-   提供了新接口：assign_weight接口，将torch模型的权重拷贝到paddle模型
-   优化权重初始化过程以及对齐报错信息： 现在会打印树形结构，并标注出错的位置
-   更新了optimizer的使用方法：可以传入一个lambda(需要在lambda内自行clear grad)
-   提供了自定义初始化接口：无法直接对齐的模型，现在能够通过提供一个自定义初始化函数进行初始化




## 简介

PaDiff是基于PaddlePaddle与PyTorch的模型精度对齐工具。传入Paddle与Torch模型，PaDiff将对训练过程中的所有中间结果以及训练后的模型权重进行对齐检查，并以调用栈的形式提示模型第一次出现精度diff的位置。




## 安装

  PaDiff v0.1 版本已发布，可通过如下命令安装：

  ```
pip install padiff
  ```

  尝鲜版或开发者推荐clone源码并使用如下命令安装：

  ```
python setup.py install
  ```



## 使用说明

### auto_diff 接口参数与使用说明

  接口功能：进行模型对齐检查

  接口函数签名：`auto_diff(layer, module, example_inp, auto_weights=False, steps=1, options={}, layer_map={}, loss_fn=None, optimizer=None)`

  -   layer：传入paddle模型

  -   module：传入torch模型

  -   example_inp：传入输入的样例数据，样例数据包含 ( paddle_input, torch_input ) 的结构，其中paddle_input和torch_input是一个dict，包含了需要传入给对应的layer和module的name到value的映射，即最后调用时使用 layer(**paddle_input) 的形式。注意顺序，paddle在前torch在后。

  -   auto_weights: 是否使用随机数值统一初始化paddle与torch模型，默认为True

  -   layer_map: 指定paddle与torch的layer映射关系，当模型结构无法完全对齐时需要通过此参数指定layer的映射关系。[LayerMap使用说明](#layermap)

      -   layer_map的具体使用方法详见LayerMap使用说明，以及实例代码的 case2
  -   options：一个传递参数的字典

      -   “atol”: 绝对精度误差上限，默认值为  `0`

      -   “rtol”: 相对精度误差上限，默认值为  `1e-7`

      -   “diff_phase”:  `"both"|"forward"`  默认为  `"both"`。设置为  `"both"`  时，工具将比较前反向的diff；当设置为  `"forward"`  时，仅比较前向diff，且会跳过模型的backward计算过程。

      -   “compare_mode”:  `"mean"|"strict"`  默认为  `"mean"`。  `"mean"`  表示使用Tensor间误差的均值作为对齐标准；  `"strict"`  表示对Tensor进行逐数据（Elementwise）的对齐检查。

      -   “single_step”:  `True|False`  默认为  `False`。设置为  `True`  时开启单步对齐模式，forward过程中每一个step都会同步模型的输入，可以避免层间误差累积。注意：开启single_step后将不会触发backward过程，"diff_phase"参数将被强制设置为  `"forward"`。

  -   loss_fn：由paddle和torch使用的损失函数按顺序组成的list。在使用时，要求传入的loss function只接受一个参数。

  -   optimizer：由paddle和torch使用的优化器或lambda函数按顺序组成的list，当传入lambda函数，它需要同时完成step和clear grad的操作。

  -   steps: 支持多step的对齐检查，默认值为1。当输入steps >1 时要求  `option["diff_phase"]`  为  `"both"`，且传入了optimizer

  关于 auto_diff 接口的使用方法和样例，详情可以见 [样例代码](#examples)

  ### assign_weight 接口参数与使用说明

  接口功能：将torch模型权重复制到paddle模型中，可以结合layer_map进行自定义初始化

  函数接口签名：`assign_weight(layer, module, layer_map=LayerMap())`

  -   layer：传入paddle模型

  -   module：传入torch模型

  -   layer_map: 指定paddle与torch的layer映射关系，当模型结构无法完全对齐时需要通过此参数指定layer的映射关系。

      -   layer_map的具体使用方法详见LayerMap使用说明，以及实例代码的 case2

  使用样例见：[assign_weight样例代码](#assign_weight)


​

## LayerMap <span id="layermap"></span>

  #### 概述

    1. layer_map可以指定两个sublayer之间的对应关系，这样做可以略过sublayer内部的数据对齐，但仍保留指定sublayer的输出数据对齐检查。
       -   指定对应关系后，auto_diff将尝试初始化这些sublayer。<font color="red">若目前auto_diff未支持此类sublayer的初始化，将在输出信息中进行提示，用户必须通过 [special init 机制](#special_init) 自行初始化这些sublayer。</font>
    2. layer_map可以指定igoner layers，这些layer的所有相关数据将不会进行对齐检查。

  #### LayerMap的构建


LayerMap是一个类，不需要初始化参数，使用前需要构造一个instance

-   LayerMap.map：用来指定模型中组件的对应关系，如果两个模型无法自然对齐，那么需要指定Layermap来告诉auto_diff他们的对应关系。

    -   ```py
        from padiff import LayerMap
        layer_map = LayerMap()
        layer_map.map = {
            layer.sublayer1 : module.submodule1
        }
        # 表示 paddle中的layer.sublayer1和torch的model.submodule1 是对应的。
        ```

    -   指定对应关系后，将<font color="red">**不进行sublayer的比较，仅进行顶层模块的比较**</font>

-   LayerMap.ignore：用于略过模型中某组件的对齐检查，但是他们的子layer/module 还是会进行对比，只是跳过当前的这一层。如果希望跳过所有子layer，请使用 `LayerMap.ignore_recursivly`

    -   ```py
        layer_map = LayerMap()
        layer_map.ignore(layer.nop_layer)
        ```

-   LayerMap.ignore_recursivly：用于略过模型中某组件以及其sublayer的所有对齐检查，如果不希望跳过子layer，请使用`LayerMap.ignore`

    -   ```py
        layer_map = LayerMap()
        layer_map.ignore_recursivly(layer.nop_layer)
        ```

-   LayerMap.ignore_class：略过`layer`和他子layer中所有class类型为`LayerClass` 的模型。

    -   ```py
        layer_map = LayerMap()
        layer_map.ignore_class(layer, LayerClass)
        ```

#### LayerMap使用样例

见 [样例代码](#layer_map)



## Special Init <span id="special_init"></span>

<font color="red"> 如果 LayerMap 指定 paddle_layer 和 torch_module为对应模型，并且不在如下支持自动初始化模型中，则必须指定 special init 方法。</font>

- MultiHeadAttention

- LSTM

#### Special init 添加方法

`add_special_init`是给某个Layer添加Special init的入口函数：

```py
from padiff import add_special_init
def init_function(paddle_layer, torch_module):
    ## your initialization logic
    ## ...
    ## ...

add_special_init({"LSTM": init_function})
```

上述代码给LSTM指定了一个 init_function，这个函数接受paddle_layer和torch_module，并且确保他们的参数一致。



## 样例代码：

### case1：auto_diff基本使用 <span id="examples"></span>

-   在使用auto_diff时，需要传入paddle模型与torch模型，在模型定义时，需要将forward中所使用的子模型在  `__init__`  函数中定义，并保证其中的子模型定义顺序一致，具体可见下方示例代码

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

### case2：layer_map的使用 <span id="layer_map"></span>

layer_map使用情景之一： 顶层模型对应，但子模型无对应关系。

```py
# 由于paddle与torch的MultiHeadAttention无法直接对齐
# 需要使用 layer_map 功能

class SimpleLayer(paddle.nn.Layer):
  def __init__(self):
      super(SimpleLayer, self).__init__()
      self.attn = paddle.nn.MultiHeadAttention(16, 1)

  def forward(self, q, k, v):
      x = self.attn(q, k, v)
      return x

class SimpleModule(torch.nn.Module):
  def __init__(self):
      super(SimpleModule, self).__init__()
      self.attn = torch.nn.MultiheadAttention(16, 1, batch_first=True)

  def forward(self, q, k, v):
      x, _ = self.attn(q, k, v)
      return x


layer = SimpleLayer()
module = SimpleModule()

# 目前 auto_diff 已支持 MultiHeadAttention 的权重自动初始化，因此此处无需其他操作
layer_map = LayerMap()
layer_map.map = {layer.attn: module.attn}

inp = paddle.rand((2, 4, 16)).numpy()
inp = (
    {"q": paddle.to_tensor(inp), "k": paddle.to_tensor(inp), "v": paddle.to_tensor(inp)},
    {"q": torch.as_tensor(inp), "k": torch.as_tensor(inp), "v": torch.as_tensor(inp)},
)

auto_diff(layer, module, inp, auto_weights=True, layer_map=layer_map, options={"atol": 1e-4})

```

layer_map使用情景之二： 略过无法对齐的sublayer

使用 auto_diff 时，可能出现这样的情况：从计算逻辑上 paddle 与 torch 模型是对齐的，但从模型结构看，它们并不对齐。**若的确找不到合适的顶层模块设置对应**，那么可以使用 ignore layer 功能，略过部分layer的对齐检查。

（更新后，以下大部分情况都已经可以自动避免）

1.  在 paddle / torch 模型定义中，某一方使用了wrap layer（比如 Sequential 或者自定义的类），而另一方并未使用（或者使用了另一种包裹方式）
2.  在 paddle / torch 模型定义中，某一方使用了 API 接口，另一方使用了sublayer，例如 Relu，导致模型结构存在差异，需要使用 ignore layer 功能略过 API 所对应的 sublayer （暂未支持 API 与 sublayer 的对齐）
3.  在 paddle / torch 模型定义中，一系列顺序的sublayer可以对齐，但是单个sublayer无法对应，auto_diff暂时不支持直接在LayerMap中设置多对多的映射关系

```py
class NOPLayer(paddle.nn.Layer):
  def __init__(self):
      super(NOPLayer, self).__init__()

  def forward(self, x):
      return x

class SimpleLayer4(paddle.nn.Layer):
  def __init__(self):
      super(SimpleLayer4, self).__init__()
      self.nop = NOPLayer()
      self.linear = paddle.nn.Linear(100, 10)

  def forward(self, x):
      x = self.nop(x)
      x = self.linear(x)
      return x

class SimpleModule4(torch.nn.Module):
  def __init__(self):
      super(SimpleModule4, self).__init__()
      self.linear = torch.nn.Linear(100, 10)

  def forward(self, x):
      x = self.linear(x)
      return x

layer = SimpleLayer4()
module = SimpleModule4()

inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})

layer_map = LayerMap()
# layer.nop 导致模型无法对齐，故而需要ignore
layer_map.ignore(layer.nop)

auto_diff(
  layer, module, inp, auto_weights=True, layer_map=layer_map, options={"atol": 1e-4}
)


```

### case3：assign_weight的使用<span id="assign_weight"></span>

```py
from padiff import assign_weight, LayerMap
import torch
import paddle

layer = SimpleLayer()
module = SimpleModule()
layer_map = LayerMap()

# 可以使用assign_weight接口，将torch模型的参数拷贝到paddle模型
assign_weight(layer, module, layer_map)
```

### case4：loss_fn的使用

-   不指定loss_fn时，将使用auto_diff内置的一个fake loss function进行计算
-   不支持传入label

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



## 输出信息示例

-   正确对齐时的输出信息：

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

-   模型前反向对齐失败时的输出信息：

    -   指明diff出现的阶段：`Forward  Stage` or `Backward Stage`
    -   diff出现时具体的对比数据
    -   打印模型结构，并标注出现diff的位置（当log过长，将输出到log文件中，并打印log文件路径）
    -   打印栈信息，可以定位到具体的代码位置

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

-   模型weight/grad对齐失败的报错信息
    -   由于weight/grad对齐信息一般比较多，所以会将信息输入到log文件，并输出log文件路径

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



## 注意事项

### 关于device

auto_diff 工具的工作与 device 无关，如果需要进行 cpu/gpu 的对齐，只需要传入device 为 cpu/gpu 的模型以及输入即可

-   在调用paddle模型构造函数以及input data初始化前，使用 paddle.set_device(xxx)
-   在构造torch模型后，使用 torch_module = torch_module.to(xxx)， torch_input = torch_input.to(xxx)



## 调试建议

如果遇到了 auto_diff 函数提示某个 layer 没有对齐，可以考虑如下几个 debug 建议：

-   如果报告不是上述的Success或者是Failed，那么说明模型没有满足预定的假设。可以结合 报错信息 进行分析。常见问题是：Torch 模型和 Paddle 模型没有满足Layer定义的一一对应假设。可以通过 print 两个模型来进行假设验证，一个满足一一对应的例子应该如下图（Layer的名字可以不用相同）![e11cd8bfbcdaf5e19a3894cecd22d212](https://user-images.githubusercontent.com/16025309/209917443-e5c21829-f4a6-4bdf-a621-b123c11e83d6.jpg)

-   如果显示精度有diff，先分析Paddle和Torch的调用栈，找到对应的源码并分析他们在逻辑上是否是对应的Layer，如果不是对应的Layer，那么说明 Torch 模型和 Paddle 模型没有满足Layer定义的一一对应假设。如图  ![3d569899c42f69198f398540dec89012](https://user-images.githubusercontent.com/16025309/209917231-717c8e88-b3d8-41bc-b6a9-0330d0d9ed50.png)

-   如果模型没有满足Layer定义的一一对应假设，可以通过`layer_map`指定Layer的映射关系。例如下图中共有三个SubLayer没有一一对齐，因此需要通过`layer_map`指定三个地方的映射关系。 如图  ![image](https://user-images.githubusercontent.com/40840292/212643420-b30d5d6f-3a26-4a41-8dc2-7b3e6622c1d5.png)

```
     layer = SimpleLayer()
     module = SimpleModule()

     layer_map = {
     layer.transformer.encoder.layers[0].self_attn: module.transformer.encoder.layers[0].self_attn,
     layer.transformer.decoder.layers[0].self_attn: module.transformer.decoder.layers[0].self_attn,
     layer.transformer.decoder.layers[0].cross_attn: module.transformer.decoder.layers[0].multihead_attn,
     } # object pair的形式

     auto_diff(layer, module, inp, auto_weights=False, layer_map=layer_map, options={"atol": 1e-4})

```

-   如果不是上述的问题，那么可以考虑进行debug，比如构造最小复现样例或者是pdb调试等等。

-   如果上述无法解决您的问题，或者认为找不到问题，可以考虑给本仓库提一个Issue。
