- [Interfaces](#interfaces)
  - [一、`auto_diff` 接口参数](#一auto_diff-接口参数)
    - [接口函数签名](#接口函数签名)
    - [必要参数](#必要参数)
    - [可选参数](#可选参数)
    - [kwargs 可选项](#kwargs-可选项)
  - [二、`assign_weight` 接口参数](#二assign_weight-接口参数)
    - [函数接口签名](#函数接口签名)
    - [参数说明](#参数说明)
  - [三、`check_dataloader` 接口参数](#三check_dataloader-接口参数)
    - [函数接口签名](#函数接口签名-1)
    - [参数说明](#参数说明-1)

# Interfaces
## 一、`auto_diff` 接口参数

### 接口函数签名
`auto_diff(base_model, raw_model, inputs, loss_fns=None, optimizers=None, layer_map=None, **kwargs)`

用于对齐模型

### 必要参数

  -   `base_model` ：作为对齐基准的 paddle/torch 模型。
      -   在模型初始化时，将 base_model 的权重拷贝至 raw_model。
      -   在 single_step 模式下，将 base_model 的输入同步作为 raw_model 的输入。

  -   `raw_model` ：待对齐的 paddle/torch 模型。

  -   `inputs` ：样例数据。传入结构为 (base_model_inputs, raw_model_inputs) 的 list/tuple，其中 base_model_inputs 和 raw_model_inputs 是 dict 类型，最终以 `model(**model_inputs)` 的形式进行传参。

### 可选参数

  -   `loss_fns` ：损失函数。传入结构为 (base_model_loss, raw_model_loss) 的 list/tuple。 要求传入的 loss function 只接受一个参数。

  -   `optimizers` ：优化器。传入结构为 (base_model_opt, raw_model_opt) 的 list/tuple。由 paddle/torch 的优化器或 lambda 函数组成，当传入 lambda 函数，它需要同时完成 step 和clear grad的 操作。

  -   `layer_map` : 指定 base_model 与 raw_model 间的映射关系，要求 base_model 的部分作为 key，raw_model 的部分作为 value。当模型结构无法完全对齐时需要通过此参数指定 layer 的映射关系，详见[LayerMap使用说明](LayerMap.md)

### kwargs 可选项

  -   `atol` ： 绝对精度误差上限，默认值为  `0`

  -   `rtol` ： 相对精度误差上限，默认值为  `1e-7`

  -   `auto_init` : 是否使用 base_model 的权重初始化 raw_model，默认为 `True`

  -   `compare_mode` ：  `"mean"|"strict"`  默认为  `"mean"`。  `"mean"`  表示使用Tensor间误差的均值作为对齐标准；  `"strict"`  表示对Tensor进行逐数据（Elementwise）的对齐检查。

  -   `diff_phase` ：  `"both"|"forward"|"backward"`  默认为  `"both"`。设置为  `"both"`  时，工具将比较前反向的 diff；当设置为  `"forward"`  时，仅比较前向 diff，且会跳过模型的 backward 计算过程。"backward" 仅在使用 single_step 时有效。

  -   `single_step` ：  `True|False`  默认为  `False`。设置为  `True`  时开启单步对齐模式，forward 过程中每一个 step 都会同步模型的输入，可以避免层间误差累积。

  > 注：
  >
  > single_step 模式下，对齐检查的逻辑会随着 diff_phase 属性的变化而不同。如果需要同时用 single_step 对齐前反向，则 padiff 将会运行模型两次，并分别进行前向和反向的 single_step 对齐检查 （single step 模式下运行模型的 forward 无法正常训练）。

  -   `steps` ： 支持多 step 的对齐检查，默认值为 1。只有当 `option["diff_phase"]`  为  `"both"`，且传入了optimizers 时才允许 steps > 1。

  -   `model_names` :  指定 base_model 与 raw_model 的名字，用于打印 log 信息，默认为类型名加后缀。

使用代码示例：
```py
from padiff import auto_diff
import torch
import paddle

class SimpleModule(torch.nn.Module):
  def __init__(self):
      super(SimpleModule, self).__init__()
      self.linear1 = torch.nn.Linear(100, 10)
  def forward(self, x):
      x = self.linear1(x)
      return x

class SimpleLayer(paddle.nn.Layer):
  def __init__(self):
      super(SimpleLayer, self).__init__()
      self.linear1 = paddle.nn.Linear(100, 10)
  def forward(self, x):
      x = self.linear1(x)
      return x

module = SimpleModule()
layer = SimpleLayer()

inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({'x': torch.as_tensor(inp) },
     {'x': paddle.to_tensor(inp)})

auto_diff(module, layer, inp, atol=1e-4, rtol=0, auto_init=True, compare_mode='strict', single_step=False)
```



## 二、`assign_weight` 接口参数

### 函数接口签名
`assign_weight(base_model, raw_model, layer_map={})`

将 base_model 模型权重复制到 raw_model 模型中，可以结合 `layer_map` 进行自定义初始化

### 参数说明

-   `base_model` ：基准权重值

-   `raw_model` ：待初始化的模型

-   `layer_map` ： 指定 base_model 与 raw_model 的映射关系，当模型结构无法完全对齐时需要通过此参数指定 layer的 映射关系；详见 [LayerMap使用说明](LayerMap.md)


使用代码示例：
```py
from padiff import assign_weight
import torch
import paddle

class SimpleModule(torch.nn.Module):
  def __init__(self):
      super(SimpleModule, self).__init__()
      self.linear1 = torch.nn.Linear(100, 10)
  def forward(self, x):
      x = self.linear1(x)
      return x

class SimpleLayer(paddle.nn.Layer):
  def __init__(self):
      super(SimpleLayer, self).__init__()
      self.linear1 = paddle.nn.Linear(100, 10)
  def forward(self, x):
      x = self.linear1(x)
      return x

module = SimpleModule()
layer = SimpleLayer()

assign_weight(module, layer)
```


## 三、`check_dataloader` 接口参数

### 函数接口签名
`check_dataloader(first_loader, second_loader, options=None)`

传入两个 dataloader，对它们的数据进行比较

### 参数说明

-   `first_loader`，`second_loader` ：两个用于比对的 dataloader ，无顺序要求

-   `kwargs` ： 支持传入额外的对比选项

  -   `atol` ： 绝对精度误差上限，默认值为  `0`

  -   `rtol` ： 相对精度误差上限，默认值为  `1e-7`

  -   `compare_mode` ：  `"mean"|"strict"`  默认为  `"mean"`。  `"mean"`  表示使用Tensor间误差的均值作为对齐标准；  `"strict"`  表示对Tensor进行逐数据（Elementwise）的对齐检查。

使用示例：
```py
from paddle_data import paddle_dataloader
from torch_data import torch_dataloader

check_dataloader(paddle_dataloader, torch_dataloader, atol=1-e7)
```
