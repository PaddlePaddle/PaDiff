# PaDiff 工具设计方案

PaDiff 工具的实现方案基于 paddle/torch 框架所提供的 hook 机制。为一个 paddle/torch 模型注册 hook ，能够在模型执行前后触发相关逻辑，从而收集运行过程中的内部信息。PaDiff 工具将它们保存下来用于对比分析。

<img width="787" alt="image-20230620111245259" src="https://github.com/feifei-111/PaDiff/assets/79986504/ea6d668c-f624-4513-8be1-f3e658fca319">
