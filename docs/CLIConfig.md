# PaDiff 配置文件参考（命令行运行）

本工具支持代码自动注入，通过命令行运行，运行时通过 YAML 配置文件接收所有参数。
配置文件分为三个主要部分：`CLI`, `PaDiffGuard`, 和 `COMPARE`。

完整文件示例请参考 [config_example](./config_example.yaml)

## CLI 部分

定义与脚本执行相关的参数。

| 参数            | 类型   | 必需 | 默认值         | 说明                             |
| --------------- | ------ | ---- | -------------- | -------------------------------- |
| `pt_cmd`        | string | 是   | -              | 运行 PyTorch 脚本的完整命令      |
| `pd_cmd`        | string | 是   | -              | 运行 PaddlePaddle 脚本的完整命令 |
| `pt_model_name` | string | 否   | "model"        | PyTorch 模型实例的变量名         |
| `pd_model_name` | string | 否   | "model"        | PaddlePaddle 模型实例的变量名    |
| `pt_optim_name` | string | 否   | null           | PyTorch 优化器实例的变量名       |
| `pd_optim_name` | string | 否   | null           | PaddlePaddle 优化器实例的变量名  |
| `log_dir`       | string | 否   | "./padiff_log" | 日志和报告的输出目录             |

## PaDiffGuard 部分

定义模型对齐的核心行为。

| 参数                | 类型         | 必需 | 默认值 | 说明                                                 |
| ------------------- | ------------ | ---- | ------ | ---------------------------------------------------- |
| `align_depth`       | int or "inf" | 否   | "inf"  | 对齐的深度。"inf" 表示最细粒度                       |
| `single_step_mode`  | string       | 否   | null   | 单步对齐模式 ("forward", "backward", "both")         |
| `load_init_weights` | bool         | 否   | false  | 是否自动对齐初始化权重，若已手动对齐，请设置为 false |
| `load_first_inputs` | bool         | 否   | false  | 是否自动第一次的输入，若已手动对齐，请设置为 false   |
| `max_calls`         | int          | 否   | 1      | 最大前反向调用次数                                   |
| `black_list`        | list         | 否   | []     | 不参与对齐的层名列表，列表内元素为 str 类型          |
| `keys_mapping`      | dict         | 否   | null   | 模型参数名映射字典                                   |

## COMPARE 部分

定义结果对比的精度和逻辑。

| 参数           | 类型   | 必需 | 默认值  | 说明                                    |
| -------------- | ------ | ---- | ------- | --------------------------------------- |
| `atol`         | float  | 否   | 1e-6    | 绝对误差容忍度                          |
| `rtol`         | float  | 否   | 1e-6    | 相对误差容忍度                          |
| `compare_mode` | string | 否   | "mean"  | 对比模式 ("mean", "strict", "abs_mean") |
| `action_name`  | string | 否   | "equal" | 对比动作 ("equal", "loose_equal")       |

## 示例和详细说明

#### 命令参数 (pt_cmd, --pd_cmd)

- 这些参数是您运行原始模型的完整命令
- 通常以 'python' 开头
- 必须指向包含您模型代码的 Python 脚本
- 必需被包含在 config 文件中，或通过命令行传入

```
pt_cmd: "python torch_project/run.py"
pd_cmd: "python paddle_project/run.py"
```

#### 模型变量名参数 (pt_model_name, pd_model_name)

- 这些参数指定您在脚本中创建模型实例的**变量名**
- 它们不是类名，也不是文件名
- 它们是模型实例化时 `=` 左边的标识符

```
# 如果您的 PyTorch 脚本中有：
my_torch_model = MyNet()
output = my_torch_model(input_tensor)
# 那么应该使用：
pt_model_name: "my_torch_model"

# 如果您的 Paddle 脚本中有：
net = SimplePaddle()
out = net.generate(input_tensor)
# 那么应该使用：
pd_model_name: "net"

# 如果您的 Paddle 脚本中有：
trainer = SFTTrainer(
    args=training_args,
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
)
trainer.train()
那么应该使用：
pd_model_name: "trainer.model"
```

#### 优化器名参数 (pt_optim_name, pd_optim_name)

- 这些参数指定您在脚本中创建优化器实例的**变量名**。
- 它们不是类名，也不是文件名。
- 它们是优化器实例化时 `=` 左边的标识符。
- 该参数为非必须参数，默认值: None (不传递优化器)

```
# 如果您的 PyTorch 脚本中有：
optim = torch.optim.Adam(
    transformer.parameters(),
    lr=1.0,
    betas=(0.9, 0.98),
    eps=1e-9,
)
# 那么应该使用：
pt_optim_name: "optim"

# 如果您的 Paddle 脚本中有：
trainer = SFTTrainer(
    args=training_args,
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
)
trainer.train()
# 由于 trainer.train() 中通常已经包含了完整的前反向过程，因此不需要传递此参数
```

#### 日志目录参数 (log_dir)

- 指定生成报告和日志的目录
- 默认值: ./padiff_log

```
log_dir: "./padiff_log"
```

#### 对齐深度参数 (align_depth)

- 控制对齐的粒度。通过指定一个深度值，可以忽略该深度以下的所有子模块
- 值为整数: 指定一个具体的深度。例如，--align_depth 1 会忽略深度为1及以下的所有子模块
- 默认值: 'inf' ，即无限深度，会对齐到最细粒度的层（如 Linear, ReLU）
- 值为整数，当数值超过模型最大迭代深度时，相当于 'inf'

```
align_depth: 0   # 只对齐顶层模块
align_depth: 1   # 对齐到第一层子模块
align_depth: "inf" # 对齐到最细粒度
```

#### 单步对齐模式参数 (single_step_mode)

- 启用逐层对齐模式
- 可选值: forward, backward, both
- 默认值: None (不启用)
- 当启用时，工具会从自动加载基准模型的输出，并用其替换对齐模型的相应层输出

#### 结果对比参数（COMPARE）

- 控制模型输出结果的对比精度和模式
- atol: 绝对误差容忍度 (default: 1e-6)
- rtol: 相对误差容忍度 (default: 1e-6)
- compare_mode: 对比模式，具体内容请看对应文档。可选值: mean, strict, abs_mean, 默认值: "mean"
- action_name: 对比行为，具体内容请看对应文档。可选值: equal, loose_equal, 默认值: "equal"

```
COMPARE:
    atol: 1e-4
    rtol: 1e-5
    compare_mode: "mean"
    action_name: "loose_equal"
```
