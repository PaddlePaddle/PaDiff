- [Interfaces](#interfaces)
  - [一、单模型运行及文件dump](#一单模型运行及文件dump)
    - [关于可dump的信息](#关于可dump的信息)
    - [创建proxy\_model](#创建proxy_model)
    - [运行前反向逻辑](#运行前反向逻辑)
    - [try\_dump接口](#try_dump接口)
    - [自由度更高的dump接口](#自由度更高的dump接口)
    - [设置黑白名单](#设置黑白名单)
    - [设置 layer\_map](#设置-layer_map)
    - [调用原模型的接口](#调用原模型的接口)
  - [二、离线对齐工具](#二离线对齐工具)
  - [三、`auto_diff` 接口参数](#三auto_diff-接口参数)
    - [接口函数签名](#接口函数签名)
    - [必要参数](#必要参数)
    - [黑白名单和layer\_map](#黑白名单和layer_map)
    - [可选参数](#可选参数)
    - [kwargs 可选项](#kwargs-可选项)
  - [四、`assign_weight` 接口参数](#四assign_weight-接口参数)
    - [函数接口签名](#函数接口签名)
    - [参数说明](#参数说明)




# Interfaces

## 一、单模型运行及文件dump

### 关于可dump的信息

1.   Report

     调用create_model接口后，生成的ProxyModel绑定着一份Report，当模型执行forward或backward逻辑时，会将相关的信息记录到ProxyModel绑定的Report中，在运行完毕后可以选择进行dump操作。Report会随着运行不断累积，直到调用 clear_report 或 try_dump 接口来实现清空。

2.   Parameters

     调用create_model接口后，能够dump ProxyModel 的 parameters，包括权重值和梯度值。在dump parameter 相关数据时，会根据ProxyModel所记录的黑白名单信息进行筛选，并按照模型结构进行保存。

3.   文件结构
     每一次dump都会生成一个json文件，json文件中记录各类meta信息以及树状结构，tensor信息被保存为npy文件，其文件路径记录在json文件中。

### 创建proxy_model

```py
from padiff import create_model

class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear = paddle.nn.Linear(100, 100)

    def forward(self, x):
        x = self.linear(x)
        return x

model = create_model(SimpleLayer(), name="Simple")  # name 是可选的
```



### 运行前反向逻辑

损失函数以及优化器的使用与ProxyModel无关

```py
model = create_model(SimpleLayer(), name="Simple")

output = model(inputs)			# 通过 model.__call__ 触发forward逻辑
loss = loss_fn(output)
model.backward(loss)				# 通过 model.backward 来执行反向
optimizer.step()
```



### try_dump接口

-   try_dump有一个内置的计数，当调用try_dump per_step次后，才会真正地触发dump，否则清空当前的report
-   这个接口实际会调用 dump_report 和 dump_params （见下文）

```py
model = create_model(SimpleLayer(), name="Simple")

for data in dataloader():
    output = model(data)
    loss = loss_fn(output)
    model.backward(loss)

		model.try_dump(per_step=10, dir_path)		# dir_path 可选项，try_dump 提供默认值
```



### 自由度更高的dump接口

try_dump 旨在提供一个方便快速测试的接口。事实上也可以使用下面的其他接口来实现灵活度更高的dump方案。

```py
model = create_model(SimpleLayer(), name="Simple")

for idx, data in enumerate(dataloader()):
    output = model(data)
    loss = loss_fn(output)
    model.backward(loss)

    model.dump_report(dir_path)
    model.dump_params(dir_path)				# dump_params 包含了 weight 和 grad 信息

    model.dump_grads(dir_path)
    optimizer.step()
    model.dump_weights(dir_path)			# 记录 optimizer.step 后的权重，可以检查 optimizer 的效果

    model.clear_report() 							# 需要手动删除 report，否则将不断累积
```



### 设置黑白名单

黑白名单将影响模型dump哪些部分的数据

-   白名单的优先级高于黑名单，当设置白名单后，黑名单将失效
-   设置黑白名单的接口需要提供 "mode" 参数:
    -   mode = "self"，仅将目标加入黑名单/白名单
    -   mode = "sublayers"，仅将目标的 sublayer 加入黑名单/白名单
    -   mode = "all"，将目标及其 sublayer 加入黑名单/白名单

```py
model = create_model(SimpleLayer(), name="Simple")

model.update_black_list([component0， component1], "all")
model.update_white_list([component2], "self")
```



### 设置 layer_map

-   该功能用于指定两个模型中的某些组件的对应关系，它的主要作用是：
    -   对齐顶层接口对齐，但内部实现不同的组件（使用黑名单可以达到同样效果）
    -   调整模型对齐的顺序（例如有两个结构上平行的sublayer，但它们实际的调用顺序不一致，这不影响逻辑但影响对齐）
    -   在需要使用 padiff 工具初始化模型权重时，配合自定义特殊初始化逻辑使用（见[LayerMap使用说明](LayerMap.md)）
-   设置layer_map的同时会自动调用 `model.update_black_list(layer_map, "sublayers")`
-   指定layer_map后，在离线对齐时，会根据layer_map的顺序调整对齐顺序

```py
model0 = create_model(SimpleLayer0(), name="Simple0")
model1 = create_model(SimpleLayer1(), name="Simple1")

model0.set_layer_map([model0.model.linear1, model0.model.linear2])
model1.set_layer_map([model1.model.linear1, model1.model.linear2])
```



### 调用原模型的接口

ProxyModel 的 model 成员即为原模型

```py
model = create_model(SimpleLayer(), name="Simple")
model.model.XXX
```



## 二、离线对齐工具

为不同的dump接口提供了不同的离线对齐工具：

-   check_report
-   check_params
-   check_weights
-   check_grads

离线对齐工具的接口都是一致的，以check_report 为例，函数签名为：

`check_report(report_path_0, report_path_1, cfg=None)`

-   report_path_0、report_path_1：待对齐的文件路径，这个路径与调用dump时的路径保持一致即可（即json文件所在文件夹的路径）
-   cfg：一个字典，记录了用于对齐的参数
    -   "atol"：绝对精度误差上限，默认值为  `1e-4`
    -   "rtol"：相对精度误差上限，默认值为  `1e-7`
    -   "compare_mode"：比较模式设定，可选 `"mean"|"strict"`  默认为  `"mean"`。  `"mean"`  表示使用Tensor间误差的均值作为对齐标准；  `"strict"`  表示对Tensor进行逐数据（Elementwise）的对齐检查。

## 三、`auto_diff` 接口参数

### 接口函数签名
`auto_diff(base_model, raw_model, inputs, loss_fns=None, optimizers=None, **kwargs)`

用于对齐模型

### 必要参数

  -   `base_model` ：作为对齐基准的模型，期望为ProxyModel。若传入原生 paddle/torch 模型，则进行自动转换，并且默认不使用黑白名单以及 layer_map 机制
      -   在模型初始化时，将 base_model 的权重拷贝至 raw_model。
      -   在 single_step 模式下，将 base_model 的输入同步作为 raw_model 的输入。

  -   `raw_model` ：待对齐的模型，期望为ProxyModel。若传入原生 paddle/torch 模型，则进行自动转换，并且默认不使用黑白名单以及 layer_map 机制

  -   `inputs` ：样例数据。传入结构为 (base_model_inputs, raw_model_inputs) 的 list/tuple，其中 base_model_inputs 和 raw_model_inputs 是 dict 类型，最终以 `model(**model_inputs)` 的形式进行传参。

### 黑白名单和layer_map

若需要设置黑白名单或 layer_map，方法与单模型运行时所使用的 ProxyModel 的接口一致。

若传入的 base_model，raw_model 是原生的 paddle/torch 模型，那么默认不存在黑白名单以及layer_map

### 可选参数

  -   `loss_fns` ：损失函数。传入结构为 (base_model_loss, raw_model_loss) 的 list/tuple。 要求传入的 loss function 只接受一个参数。

  -   `optimizers` ：优化器。传入结构为 (base_model_opt, raw_model_opt) 的 list/tuple。由 paddle/torch 的优化器或 lambda 函数组成，当传入 lambda 函数，它需要同时完成 step 和clear grad的 操作。


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



## 四、`assign_weight` 接口参数

### 函数接口签名
`assign_weight(base_model, raw_model)`

将 base_model 模型权重复制到 raw_model 模型中

### 参数说明

-   `base_model` ：基准权重值，期望为ProxyModel。若传入原生 paddle/torch 模型，则进行自动转换，并且默认不使用黑白名单以及 layer_map 机制
-   `raw_model` ：待初始化的模型，期望为ProxyModel。若传入原生 paddle/torch 模型，则进行自动转换，并且默认不使用黑白名单以及 layer_map 机制




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
