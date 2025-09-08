# PaDiff ![](https://img.shields.io/badge/version-v0.1-brightgreen) ![](https://img.shields.io/badge/docs-latest-brightgreen) ![](https://img.shields.io/badge/PRs-welcome-orange) ![](https://img.shields.io/badge/pre--commit-Yes-brightgreen)


**P**addle  **A**utomatically  **Diff**  precision toolkits.


## 最近更新（latest 9.8）

### 使用单行命令对齐（支持前反向对齐）

运行命令前，请运行 `python -m padiff.cli -h` 获取更详细的参数说明。

直接通过命令行运行

```sh
python -m padiff.cli \
  --pt_cmd "python torch_project/run.py" \
  --pd_cmd "python paddle_project/run.py" \
  --pt_model_name "pt_model" \
  --pd_model_name "pd_model" \
  --pt_optim_name "pt_optimizer" \
  --pd_optim_name "pd_optimizer" \
  --log_dir "./padiff_log" \
  --align_depth 1 \
  --single_step_mode "forward" \
  --atol 1e-4 \
  --rtol 1e-5 \
  --compare_mode mean \
  --action_name equal
```

或将命令写入 .yaml 文件后，运行

```sh
python -m padiff.cli --config padiff_config.yaml
```

yaml 文件样例

```python
# padiff_config.yaml
pt_cmd: "python transformer4sr/train_transformer.py"
pd_cmd: "python paddle_project/train_transformer.py"
pt_model_name: "transformer_pt"
pd_model_name: "transformer_pd"
pt_optim_name: "optimizer_pt"
pd_optim_name: "optimizer_pd"
log_dir: "./padiff_log"
align_depth: 2
single_step_mode: "forward"
atol: 1.0e-04
rtol: 1.0e-05
compare_mode: "mean"
action_name: "equal"
```

### log 设置

#### 开启 debug 模式

为了获取更多 log 信息，可以设置环境变量 `export PADIFF_LOG_LEVEL=DEBUG`，或使用命令运行 `PADIFF_LOG_LEVEL=DEBUG python -m padiff.cli ...`

#### 开启静默模式

或者为了保持控制台信息简洁，可以设置环境变量 `PADIFF_SILENT=1`，以便仅保存 log 文件，不在控制台输出 log 信息


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
