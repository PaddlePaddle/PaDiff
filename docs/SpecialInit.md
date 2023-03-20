## 已支持Special Init的组件

-   MultiHeadAttention
-   LSTM



## Special Init

使用LayerMap指定了子模型对应关系后，若padiff未支持此组件的初始化，则需要手动添加初始化函数



### 定义初始化函数

初始化函数的签名非常简单，两个输入参数分别为 paddle 模型与对应的 torch 模型，在函数中实现从torch模型拷贝参数到paddle模型即可，无需返回值

```py
def my_init(paddle_layer, torch_module):
    # copy torch param to paddle
```



### add_special_init接口

`add_special_init`是给某个Layer添加Special init的入口函数：

add_special_init接受一个字典作为输入

-   字典的key为paddle模型的类名
-   字典的value为对应的初始化函数

```py
from padiff import add_special_init
def my_init(paddle_layer, torch_module):
    # your initialization logic
    # ...

add_special_init(
    paddle_name="LSTM", torch_name="LSTM", my_init
)
```

上述代码给LSTM指定了一个 init_function，这个函数接受paddle_layer和torch_module，并且确保他们的参数一致。



### 为本repo贡献你的初始化函数

本repo的special_init函数均存储于 PaDiff/padiff/special_init文件夹下

1.   在PaDiff/padiff/special_init文件夹下新建 init_XXX.py（必须以init_开头）
2.   在文件中编写你的初始化函数，并用装饰器进行注册，见下方示例代码
3.   提交后padiff工具就能够支持XXX模型的special init，无需重复编写

```py
# PaDiff/padiff/special_init/init_XXX.py

from .special_init_pool import global_special_init_pool as init_pool


@init_pool.register(paddle_name="ClassXXX", torch_name="ClassYYY")  	# 此处填写模型的类名
def init_XXX(paddle_layer, torch_module):
    # your initialization logic
```
