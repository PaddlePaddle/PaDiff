- [已支持Special Init的组件](#已支持special-init的组件)
- [概述](#概述)
  - [为什么需要 SpecialInit 机制](#为什么需要-specialinit-机制)
  - [什么时候触发 SpecialInit 机制](#什么时候触发-specialinit-机制)
  - [设置自定义初始化逻辑的完整流程示例](#设置自定义初始化逻辑的完整流程示例)
- [设置 layer\_map 的方法](#设置-layer_map-的方法)
  - [使用 set\_layer\_map 手动指定](#使用-set_layer_map-手动指定)
  - [使用 auto\_layer\_map 自动指定](#使用-auto_layer_map-自动指定)
- [自定义模型初始化函数](#自定义模型初始化函数)
- [使用 add\_special\_init 接口注册自定义初始化函数](#使用-add_special_init-接口注册自定义初始化函数)
- [如何向本 repo 贡献模型初始化函数](#如何向本-repo-贡献模型初始化函数)


## 已支持Special Init的组件

-   MultiHeadAttention
-   LSTM
-   BatchNorm2D


## 概述

### 为什么需要 SpecialInit 机制

在模型对齐时总会有自定义的模型存在，当两个子模型的功能一致，但由于实现方式不同导致无法直接初始化权重时，则需要使用 SpecialInit 机制进行初始化

### 什么时候触发 SpecialInit 机制

在layer_map中被指定的子模型会触发 SpecialInit 机制，从 base model 中使用特殊方式拷贝权重到 raw model。

### 设置自定义初始化逻辑的完整流程示例

自定义模型初始化逻辑的过程主要为：
1. 编写模型初始化函数
2. 使用 add_special_init 接口注册函数
3. 通过设置 layer_map 触发并使用模型初始化函数
4. 向本 repo 贡献你的初始化函数 （与前面的步骤无关）



## 设置 layer_map 的方法

### 使用 set_layer_map 手动指定

首先调用 create_model 将待对齐模型转换为 ProxyModel
然后通过 set_layer_map 接口手动指定一一对应的子模型（需传入 object，且对应顺序一致）

```py
model0 = create_model(SimpleLayer0(), name="Simple0")
model1 = create_model(SimpleLayer1(), name="Simple1")

model0.set_layer_map([model0.model.linear1, model0.model.linear2])
model1.set_layer_map([model1.model.linear1, model1.model.linear2])
```


### 使用 auto_layer_map 自动指定

使用 auto_layer_map 能搜索当前已经支持 SpecialInit 的组件
调用时需要传入 `"base"|"raw"` 指明模型在对齐时所处的定位

```py
model0 = create_model(SimpleLayer0(), name="Simple0")
model1 = create_model(SimpleLayer1(), name="Simple1")

model0.auto_layer_map("base")
model1.auto_layer_map("raw")
```

## 自定义模型初始化函数

在 layer_map 中的模型将会触发特殊初始化机制，此时会检查当前存在的初始化函数并尝试调用它，若不存在，则会报出错误。

> 在 PaDiff 工具内已经注册了部分框架组件的初始化逻辑，如 LSTM、MultiHeadAttention 等，无需重复编写

当需要注册使用自定义的初始化函数时，需要按照初始化函数的接口规范编写，其签名为：

`def init_logic(base_model, raw_model)`

以 init_logic 函数为例，它有两个参数，其逻辑应该是将 base_model 的权重拷贝到 raw_model。

下面是 BatchNorm2D 模型初始化的例子

```py
def init_BatchNorm2D(module, layer):
    param_dict = {}
    for name, param in module.state_dict().items():
        name = name.replace("running_var", "_variance").replace("running_mean", "_mean")
        param_dict[name] = paddle.to_tensor(param.cpu().detach().numpy())
    layer.set_state_dict(param_dict)

```

## 使用 add_special_init 接口注册自定义初始化函数

在定义完毕初始化函数后，需要注册才能生效，只需要使用 add_special_init 接口即可， 其签名为：

`def add_special_init(base_framework, base_model_name, raw_framework, raw_model_name, func)`

add_special_init 共有5个输入，前两个标明 base_model 的框架名和模型名（类名），后续两个参数标注 raw_model 的框架名和模型名，最后一个参数传入自定义初始化函数。

> 框架名传入 "paddle" 或 "torch"
> 注意保证 base_model 和 raw_model 的顺序
> 模型名在不同框架下可能有所差异，例如下面代码中，BatchNorm2D 在 paddle 和 torch 中就有不同的名字

例子：
```py
from padiff import add_special_init

def init_BatchNorm2D(module, layer):
    param_dict = {}
    for name, param in module.state_dict().items():
        name = name.replace("running_var", "_variance").replace("running_mean", "_mean")
        param_dict[name] = paddle.to_tensor(param.cpu().detach().numpy())
    layer.set_state_dict(param_dict)

add_special_init("torch", "BatchNorm2d", "paddle", "BatchNorm2D", init_BatchNorm2D)
```

注册完毕后，layer_map 内的子模型就能搜索到自定义初始化函数，并触发相关逻辑了

## 如何向本 repo 贡献模型初始化函数

**对 Paddle 框架提供的Layer组件，如果存在与 Torch 提供的组件不对齐，可以将相应的初始化函数提供给本 Repo**

1.   找到 special_init 文件夹，并新建你的文件

该文件夹位于 `PaDiff/padiff/weight_init/special_init`，请在该文件夹下新建文件 `init_XXXXX.py` （必须以 `init_` 开头）

2.   在新建的文件中编写初始化函数

3.   使用 register 装饰器装饰初始化函数

    register 装饰器的参数与 add_special_init 接口的前四个参数一致

示例代码如下：

```py
# PaDiff/padiff/special_init/init_BatchNorm2D.py

import paddle

from .special_init_pool import global_special_init_pool as init_pool


@init_pool.register("torch", "BatchNorm2d", "paddle", "BatchNorm2D") # base_model 框架名、类名，raw_model 框架名、类名
def init_BatchNorm2D(module, layer):
    param_dict = {}
    for name, param in module.state_dict().items():
        name = name.replace("running_var", "_variance").replace("running_mean", "_mean")
        param_dict[name] = paddle.to_tensor(param.cpu().detach().numpy())
    layer.set_state_dict(param_dict)
```

4.   提交PR

完成上面的文件编写后，可以联系 repo 管理员 review 并合入，提交后 padiff 工具就能够支持对应模型的初始化逻辑，无需重复编写。
