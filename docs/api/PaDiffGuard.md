# PaDiffGuard API 文档

`PaDiffGuard` 是一个上下文管理器（Context Manager），用于在深度学习模型的前向和反向传播过程中自动进行模型对齐检查。它通过拦截计算过程，比较不同框架（如 PyTorch 和 PaddlePaddle）或同一框架下不同实现的中间结果、权重和梯度，以确保它们的行为一致。

## 参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `model` | `paddle.nn.Layer` or `torch.nn.Module` | 是 | - | 需要进行对齐检查的主模型实例。 |
| `optimizer` | `paddle.optimizer` or `torch.optim` | 否 | `null` | 与 `model` 配对的优化器实例。<br><br>**当反向传播不能被 `PaDiffGuard` 包裹时**：如果传递此参数，`PaDiffGuard` 会包装其 `step()` 方法以确保梯度更新的一致性。<br><br>**当反向传播可以被 `PaDiffGuard` 包裹时**：梯度一旦更新就会被自动获取，此时不需要传递此参数。 |
| `name` | `string` | 否 | `"model"` | 为当前 `model` 指定一个名称，该名称将用于日志记录和结果文件的命名。 |
| `align_depth` | `int` or `"inf"` | 否 | `"inf"` | 对齐的深度。<br><br>- 可以是一个整数，表示只检查前 N 层。<br>- 也可以是 `"inf"`，表示检查所有层。 |
| `single_step_mode` | `string` | 否 | `null` | 单步对齐模式，可选值为 `"forward"`, `"backward"`, `"both"`。<br><br>在此模式下，`PaDiffGuard` 会从 `base_dump_path` 加载参考数据进行比对。 |
| `load_init_weights` | `bool` | 否 | `false` | 是否从 `base_dump_path` 加载初始化权重。<br><br>若已手动对齐初始化权重，请设置为 `false`。 |
| `load_first_inputs` | `bool` | 否 | `false` | 是否从 `base_dump_path` 加载第一轮输入。<br><br>若已手动对齐输入，请设置为 `false`。 |
| `base_dump_path` | `string` | 否 | `null` | 基准数据的根目录路径。<br><br>当 `load_init_weights`、`load_first_inputs` 或 `single_step_mode` 被启用时，此参数必须提供。 |
| `framework` | `string` | 否 | `null` | 指定模型所属的框架，可选值为 `"paddle"` 和 `"torch"`。<br><br>用于指导如何处理特定框架的数据，当 `load_first_inputs` 被启用时，此参数必须提供。 |
| `seed` | `int` | 否 | `42` | 设置随机种子，以确保对齐过程中的随机行为是可复现的。 |
| `max_calls` | `int` | 否 | `1` | 最大前/反向调用次数。<br><br>达到上限后，会强制退出整个程序。例如，前向对齐时应该设置为 `1`。 |
| `black_list` | `list[str]` | 否 | `[]` | 不参与对齐的层名列表，列表内元素为 `str` 类型。 |
| `keys_mapping` | `dict` or `Callable` | 否 | `null` | 模型参数名映射，可以为：<br><br>- **字典形式**：键和值分别为 Paddle/PyTorch 的参数名。<br>- **可调用函数**：如 `lambda name: name.replace('model_pt', 'model_pa')`。 |

## 用法

```python
from padiff import PaDiffGuard

with PaDiffGuard(
    model=model,
    optimizer=optimizer,
    name="my_model",
):
    out = model(input_data)
loss = loss_fn(out, target)
loss.backward()
optimizer.step()

```
``` python
trainer = SFTTrainer(...)
with PaDiffGuard(trainer.model, name='model_name'):
    trainer.train()
```
