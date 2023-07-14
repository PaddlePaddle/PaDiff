- [SingleStep 的工作原理](#singlestep-的工作原理)


# SingleStep 的工作原理

在 auto_diff 接口中传入 `single_step=True` 即可开启单步对齐功能，其原理如下图所示。
在运行 base model 时，会存储模型运行的中间结果，在运行 raw model 时，利用 hook 机制将每一个 sublayer 的输出替换为 base model 对应的输出
