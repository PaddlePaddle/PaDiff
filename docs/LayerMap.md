- [概述](#概述)
- [LayerMap的可用接口](#layermap的可用接口)
  - [指定对应关系](#指定对应关系)
    - [LayerMap.map](#layermapmap)
  - [指定忽略子模型](#指定忽略子模型)
    - [LayerMap.ignore](#layermapignore)
    - [LayerMap.ignore\_recursivly](#layermapignore_recursivly)
    - [LayerMap.ignore\_class](#layermapignore_class)
- [LayerMap使用样例](#layermap使用样例)


## 概述
`LayerMap` 是一个辅助模型结构对齐的工具，它主要有两种功能

1. `LayerMap` 可以指定两个子模型之间的对应关系，这样做可以略过子模型内部的数据对齐，但仍保留指定子模型的输出数据对齐检查

   -   适用情况：模型顶层可对齐，但内部无法对齐。例如：MultiHeadAttention

2. `LayerMap` 可以指定 igoner layers ，被指定的 layer 的所有相关数据将跳过对齐检查

    -   适用情况：通过忽略某些子模型后能够达成对齐



## LayerMap的可用接口

`LayerMap`是一个类，构造函数没有参数，使用前需要构造一个 instance

### 指定对应关系

#### LayerMap.map

用来指定模型中组件的对应关系，如果两个模型无法自然对齐，那么需要指定 `Layermap` 来指定他们的对应关系。

典型使用场景有：

1.   模型内部实现方式不对齐，但顶层对齐
2.   模型内部参数/子模块初始化顺序不同，导致无法对齐，但顶层对齐


>   注：指定对应关系后，**将不进行子模型的比较，仅进行顶层模块的比较**

>   注： 指定对应关系后，auto_diff将尝试初始化这些sublayer。**若目前 auto_diff 未支持此类子模型的初始化，将在输出信息中进行提示，用户必须通过 [special init 机制](SpecialInit.md) 自行初始化这些子模型**

```py
from padiff import LayerMap
layer_map = LayerMap()
layer_map.map = {
    layer.sublayer1 : module.submodule1
}
# 表示 paddle中的layer.sublayer1和torch的model.submodule1 是对应的。
```

### 指定忽略子模型

#### LayerMap.ignore

用于略过模型中某组件的对齐检查，但是他们的子模型还是会进行对比，只是跳过当前的这一层。如果希望跳过所有子模型，请使用 `LayerMap.ignore_recursivly`

```py
layer_map = LayerMap()
layer_map.ignore(layer.nop_layer)
```

#### LayerMap.ignore_recursivly

用于略过模型中某组件以及其所有子模型的对齐检查，如果不希望跳过子模型，请使用 `LayerMap.ignore`

```py
layer_map = LayerMap()
layer_map.ignore_recursivly(layer.nop_layer)
```

#### LayerMap.ignore_class

略过输入 layer 和他的子模型中所有 class 类型为 LayerClass 的模型。

```py
layer_map = LayerMap()
layer_map.ignore_class(layer, LayerClass)
```



## LayerMap使用样例

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
# 否则，此处应自定义初始化函数，并使用 special init 机制自行注册
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

使用 auto_diff 时，可能出现这样的情况：从计算逻辑上 paddle 与 torch 模型是对齐的，但从模型结构看，它们并不对齐。**若的确找不到合适的顶层模块设置对应**，那么可以使用 ignore layer 功能，略过部分子模型的对齐检查。

（更新后，以下大部分情况都已经可以自动避免）

1.  在 paddle / torch 模型定义中，某一方使用了 wrap layer（比如 Sequential 或者自定义的类），而另一方并未使用（或者使用了另一种包裹方式）
2.  在 paddle / torch 模型定义中，某一方使用了 API 接口，另一方使用了 layer/module (例如 Relu)。导致模型结构存在差异，需要使用 ignore layer 功能略过 API 所对应的 layer/module （暂未支持 API 与 layer/module 的对齐）
3.  在 paddle / torch 模型定义中，一系列顺序的子模型可以对齐，但是单个子模型无法一一对应，auto_diff暂时不支持直接在LayerMap中设置多对多的映射关系

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
