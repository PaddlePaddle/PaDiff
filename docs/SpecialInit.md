- [已支持Special Init的组件](#已支持special-init的组件)
- [使用 Special Init](#使用-special-init)
  - [定义初始化函数](#定义初始化函数)
  - [add\_special\_init接口](#add_special_init接口)
  - [为本repo贡献你的初始化函数](#为本repo贡献你的初始化函数)


## 已支持Special Init的组件

-   MultiHeadAttention
-   LSTM
-   BatchNorm2D


## 使用 Special Init

使用LayerMap指定了子模型对应关系后，若padiff未支持此组件的初始化，则需要手动添加初始化函数



### 定义初始化函数

初始化函数的签名非常简单，两个输入参数分别为 paddle 模型与对应的 torch 模型，在函数中实现从 torch 模型拷贝参数到 paddle 模型即可，无需返回值

```py
def my_init(paddle_layer, torch_module):
    # copy torch param to paddle
```



### add_special_init接口

`add_special_init` 是给某个Layer添加Special init的入口函数，如下代码所示：


```py
from padiff import add_special_init
def my_init(paddle_layer, torch_module):
    # your initialization logic
    # ...

add_special_init(
    paddle_name="LSTM", torch_name="LSTM", my_init
)
```

上述代码给LSTM指定了一个 init_function ，这个函数接受 paddle_layer 和 torch_module ，并且确保他们的参数一致。



### 为本repo贡献你的初始化函数

本repo的special_init函数均存储于 `PaDiff/padiff/special_init` 文件夹下

1.   在该文件夹下新建 `init_XXX.py`（必须以 `init_` 开头）
2.   在文件中编写你的初始化函数，并用装饰器进行注册，见下方示例代码
3.   提交后 padiff 工具就能够支持 XXX 模型的 special init ，无需重复编写

> 注： 将 torch 权重复制到 paddle ，还需要注意保证复制后的 device 不变

```py
# PaDiff/padiff/special_init/init_XXX.py

from .special_init_pool import global_special_init_pool as init_pool


@init_pool.register(paddle_name="ClassXXX", torch_name="ClassYYY")  	# 此处填写模型的类名
def init_XXX(paddle_layer, torch_module):
    # your initialization logic
```
