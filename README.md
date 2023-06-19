# PaDiff ![](https://img.shields.io/badge/version-v0.1-brightgreen) ![](https://img.shields.io/badge/docs-latest-brightgreen) ![](https://img.shields.io/badge/PRs-welcome-orange) ![](https://img.shields.io/badge/pre--commit-Yes-brightgreen)


**P**addle  **A**utomatically  **Diff**  precision toolkits.



## 最近更新

-   单模型运行并dump相关数据
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



## 已支持 `Special Init` 的组件

-   MultiHeadAttention
-   LSTM
-   BatchNorm2D
