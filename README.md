# PaDiff ![](https://img.shields.io/badge/version-v0.3-brightgreen) ![](https://img.shields.io/badge/docs-latest-brightgreen) ![](https://img.shields.io/badge/PRs-welcome-orange) ![](https://img.shields.io/badge/pre--commit-Yes-brightgreen)

**P**addle **A**utomatically **Diff** precision toolkits.

## 简介

PaDiff 是基于 PaddlePaddle 与 PyTorch 的模型精度对齐工具。传入 Paddle 和 Torch 模型，对齐训练中间结果以及训练后的模型权重，并提示精度 diff 第一次出现的位置。


## 安装

当前推荐通过源码安装，本工具需要安装 `paddlepaddle` 和 `torch`，但这两个包的版本可能存在冲突。建议通过如下命令安装后，根据 [paddlepaddle官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) 和 [pytorch官网](https://pytorch.org/get-started/locally/) 自行安装 `paddlepaddle` 和 `torch`

```sh
python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 快速开始

### 使用单行命令对齐（支持前反向对齐）

将命令写入配置文件后，通过如下命令运行

```sh
python -m padiff.cli --config padiff_config.yaml
```

完整文件示例请参考 [配置文件说明文档](docs/CLIConfig.md)，同时，运行命令前，请运行 `python -m padiff.cli -h` 获取更详细的参数说明。

### log 设置

#### 开启 debug 模式

为了获取更多 log 信息，可以设置环境变量 `export PADIFF_LOG_LEVEL=DEBUG`，或使用命令运行 `PADIFF_LOG_LEVEL=DEBUG python -m padiff.cli ...`

#### 开启静默模式

为了保持控制台信息简洁，可以设置环境变量 `PADIFF_SILENT=1`，此模式下仅保存 log 文件，不在控制台输出 log 信息

## 旧版本特性（v0.2版本）

### 使用 auto_diff 接口和其它方法

请查阅此文档[auto_diff 接口和其它方法](docs/README.md)
