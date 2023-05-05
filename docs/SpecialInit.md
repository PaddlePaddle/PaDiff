- [已支持Special Init的组件](#已支持special-init的组件)
- [设置自定义初始化逻辑的完整流程示例](#设置自定义初始化逻辑的完整流程示例)
  - [错误例子示例](#错误例子示例)
  - [自定义模型初始化函数](#自定义模型初始化函数)
  - [将模型初始化函数提供给本 repo](#将模型初始化函数提供给本-repo)


## 已支持Special Init的组件

-   MultiHeadAttention
-   LSTM
-   BatchNorm2D

## 设置自定义初始化逻辑的完整流程示例

### 错误例子示例

以下方代码块中的模型对齐为例，可见 paddle_net 与 torch_net 的功能完全一致，但在 padiff 工具中无法进行权重初始化，这是因为它们各自的 part1 组件中的 linear 定义顺序不对齐，此类情况需要自定义初始化逻辑。

```py
import paddle
import torch
from padiff import auto_diff

class PaddleNet(paddle.nn.Layer):
    def __init__(self):
        super(PaddleNet, self).__init__()
        self.part1 = PaddleComponent()
        self.part2 = paddle.nn.Linear(10, 1)

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        return x

class TorchNet(torch.nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.part1 = TorchComponent()
        self.part2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        return x


class PaddleComponent(paddle.nn.Layer):
    def __init__(self):
        super(PaddleComponent, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.linear2 = paddle.nn.Linear(100, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class TorchComponent(torch.nn.Module):
    def __init__(self):
        super(TorchComponent, self).__init__()
        self.linear2 = torch.nn.Linear(100, 10)
        self.linear1 = torch.nn.Linear(100, 100)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

paddle_net = PaddleNet()
torch_net = TorchNet()

inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})

auto_diff(paddle_net, torch_net, inp, auto_weights=True, options={"atol": 1e-4})
```

上方代码的报错信息：

```
AssertionError:  Shape of paddle param `weight` and torch param `weight` is not the same. [100, 100] vs [100, 10]

Torch Model
=========================
    TorchNet  (skip)
     |--- TorchComponent  (skip)
     |     |--- Linear    <---  *** HERE ***
     |     +--- Linear
     +--- Linear
Paddle Model
=========================
    PaddleNet  (skip)
     |--- PaddleComponent  (skip)
     |     |--- Linear    <---  *** HERE ***
     |     +--- Linear
     +--- Linear
```



### 自定义模型初始化函数

初始化函数的签名非常简单，两个输入参数分别为 paddle 模型与对应的 torch 模型，在函数中实现从 torch 模型拷贝参数到 paddle 模型即可，无需返回值。

编写自定义模型初始化函数并注册触发的示例代码如下：

```py
import paddle
import torch
from padiff import auto_diff, assign_weight, add_special_init, LayerMap

import numpy 

'''
		模型定义同上，此处略过...
		class PaddleNet ...
		class TorchNet ...
		class PaddleComponent ...
		class TorchComponent ...
'''

# 自定义模型初始化函数
def init_component(layer, module):
		print("init component")  

    # 在自定义的初始化函数中，手动对齐 linear1 <=> linear1, linear2 <=> linear2
    for (name, paddle_param), torch_param in zip(
        layer.linear1.named_parameters(prefix="", include_sublayers=False),
        module.linear1.parameters(recurse=False),
    ):
        np_value = torch_param.data.detach().cpu().numpy()
        p_shape = list(paddle_param.shape)
        t_shape = list(torch_param.shape)

        # torch 的 linear 组件中，weight 参数与 paddle 是转置的关系，需要处理
        if name == "weight":
            t_shape.reverse()
            np_value = numpy.transpose(np_value)

        assert p_shape == t_shape, "shape diff at linear1"
        paddle.assign(paddle.to_tensor(np_value), paddle_param)

    for (name, paddle_param), torch_param in zip(
        layer.linear2.named_parameters(prefix="", include_sublayers=False),
        module.linear2.parameters(recurse=False),
    ):
        np_value = torch_param.data.detach().cpu().numpy()
        p_shape = list(paddle_param.shape)
        t_shape = list(torch_param.shape)
        if name == "weight":
            t_shape.reverse()
            np_value = numpy.transpose(np_value)

        assert p_shape == t_shape, "shape diff at linear2"
        paddle.assign(paddle.to_tensor(np_value), paddle_param)

# 使用 add_special_init 接口，注册针对 PaddleComponent 和 TorchComponent 的初始化函数
add_special_init(paddle_name="PaddleComponent", torch_name="TorchComponent", func=init_component)

paddle_net = PaddleNet()
torch_net = TorchNet()

# 注册后，可以使用 auto 成员方法来自动检查对应的 layer，对 auto_diff 传入这个 layer_map 来触发自定义初始化函数
layer_map = LayerMap()
layer_map.auto(paddle_net, torch_net)

inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})

auto_diff(paddle_net, torch_net, inp, auto_weights=True, options={"atol": 1e-4}, layer_map=layer_map)
```

上述代码的运行结果如下图：
![Kev s) rorkspace prthon zetest py](https://user-images.githubusercontent.com/79986504/236400634-7b4ec90e-326f-4845-9e1d-318e83b9bfd9.png)



### 将模型初始化函数提供给本 repo

**对 Paddle 框架提供的Layer组件，如果存在与 Torch 提供的组件不对齐，可以将相应的初始化函数提供给本 Repo**

1.   找到 special_init 文件夹，并新建你的文件

该文件夹位于 `PaDiff/padiff/special_init`，请在该文件夹下新建文件 `init_XXXXX.py` （必须以 `init_` 开头）

2.   在新建的文件中编写初始化函数

在本例中，即为 init_component 函数，编写完函数后，需要添加一个装饰器以注册该函数。

示例代码如下：

```py
# PaDiff/padiff/special_init/init_componet.py

import numpy
import paddle
from .special_init_pool import global_special_init_pool as init_pool


@init_pool.register(paddle_name="PaddleComponent", torch_name="TorchComponent")  	# 此处填写模型的类名
def init_component(layer, module):
    print(" ***** init component ***** ")  
    # 在自定义的初始化函数中，手动对齐 linear1 vs linear1, linear2 vs linear2
    for (name, paddle_param), torch_param in zip(
        layer.linear1.named_parameters(prefix="", include_sublayers=False),
        module.linear1.parameters(recurse=False),
    ):
        np_value = torch_param.data.detach().cpu().numpy()

        p_shape = list(paddle_param.shape)
        t_shape = list(torch_param.shape)

        # torch 的 linear 组件中，weight 参数与 paddle 是转置的关系，需要处理
        if name == "weight":
            t_shape.reverse()
            np_value = numpy.transpose(np_value)
        assert p_shape == t_shape, "shape diff at linear1"
        paddle.assign(paddle.to_tensor(np_value), paddle_param)

    for (name, paddle_param), torch_param in zip(
        layer.linear2.named_parameters(prefix="", include_sublayers=False),
        module.linear2.parameters(recurse=False),
    ):
        np_value = torch_param.data.detach().cpu().numpy()

        p_shape = list(paddle_param.shape)
        t_shape = list(torch_param.shape)
        if name == "weight":
            t_shape.reverse()
            np_value = numpy.transpose(np_value)

        assert p_shape == t_shape, "shape diff at linear2"
        paddle.assign(paddle.to_tensor(np_value), paddle_param)
```

3.   提交PR

完成上面的文件编写后，可以联系 repo 管理员 review 并合入，提交后 padiff 工具就能够支持对应模型的初始化逻辑，无需重复编写。


