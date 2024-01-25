# PaDiff ![](https://img.shields.io/badge/version-v0.1-brightgreen) ![](https://img.shields.io/badge/docs-latest-brightgreen) ![](https://img.shields.io/badge/PRs-welcome-orange) ![](https://img.shields.io/badge/pre--commit-Yes-brightgreen)


**P**addle  **A**utomatically  **Diff**  precision toolkits.



## 最近更新

-   支持添加Paddle自定义算子
-   支持单模型运行并dump相关数据
-   提供离线对齐工具



## 简介

PaDiff 是基于 PaddlePaddle 与 PyTorch 的模型精度对齐工具。传入 Paddle 或 Torch 模型，对齐训练中间结果以及训练后的模型权重，并提示精度 diff 第一次出现的位置。

-   文档目录 [Guides](docs/README.md)
-   使用教程 [Tutorial](docs/Tutorial.md)
-   对齐ViTPose流程 [ViTPose](docs/CheckViTPose.md)
-   接口参数说明 [Interface](docs/Interfaces.md)
-   常见问题解答 [FAQs](docs/FAQs.md)




## 安装

  PaDiff v0.2 版本已发布，可通过如下命令安装：

  ```
pip install padiff
  ```

  尝鲜版或开发者推荐clone源码并使用如下命令安装：

  ```
python setup.py install
  ```



## 快速开始

### 使用 auto_diff 接口进行对齐

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

auto_diff(module, layer, inp, atol=1e-4, auto_init=True)
```



### 离线对齐

```py
############################
#      torch_model.py      #
############################

from padiff import *
import torch

class SimpleModule(torch.nn.Module):
  def __init__(self):
      super(SimpleModule, self).__init__()
      self.linear1 = torch.nn.Linear(100, 10)
  def forward(self, x):
      x = self.linear1(x)
      return x

module = SimpleModule()
module = create_model(module)

inp = paddle.ones((100, 100)).numpy().astype("float32")

for i in range(6):
    out = module(torch.as_tensor(inp))
    loss = out.mean()
    module.backward(loss)
    module.try_dump(2, f"./torch/step_{i}")


############################
#      paddle_model.py     #
############################

from padiff import *
import paddle

class SimpleLayer(paddle.nn.Layer):
  def __init__(self):
      super(SimpleLayer, self).__init__()
      self.linear1 = paddle.nn.Linear(100, 10)
  def forward(self, x):
      x = self.linear1(x)
      return x

# 此处需自行保证两个模型的初始权重以及输入数据是对齐的
layer = SimpleLayer()
layer = create_model(layer)

inp = paddle.rand((100, 100)).numpy().astype("float32")

for i in range(6):
    out = layer(paddle.to_tensor(inp))
    loss = out.mean()
    layer.backward(loss)
    layer.try_dump(2, f"./paddle/step_{i}")


############################
#         check.py        #
############################

from padiff import *

for i in range(6):
    if i % 2 == 0:
        assert check_report(f"./torch/step_{i}", f"./paddle/step_{i}") == True
        assert check_params(f"./torch/step_{i}", f"./paddle/step_{i}") == True
```

### 框架与编译器对齐
使用文档 [CINN](padiff/cinn_diff/README.md)

```python
import os
from padiff import cinn_diff


def run(run_script, base_env, cinn_env):
    run_env = cinn_diff.Env(run_script, base_env, cinn_env)
    run_env.run_base_model() #可以注释掉选择不运行base model
    run_env.run_cinn_model() #也可以注释掉选择不运行cinn model
    cinn_diff.auto_diff(run_env.base_path, run_env.cinn_path, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    run_script = "/root/workspace/PaddleNLP/model_zoo/bert/run_bert.sh"
    run(run_script, None, None)
```

## 已支持 `Special Init` 的组件

-   MultiHeadAttention
-   LSTM
-   BatchNorm2D
