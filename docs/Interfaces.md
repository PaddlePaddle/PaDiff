- [Interfaces](#interfaces)
  - [一、`auto_diff` 接口参数](#一auto_diff-接口参数)
  - [二、`assign_weight` 接口参数](#二assign_weight-接口参数)

# Interfaces
## 一、`auto_diff` 接口参数

  **接口功能**：进行模型对齐检查

  **接口函数签名**：`auto_diff(layer, module, example_inp, auto_weights=False, steps=1, options={}, layer_map={}, loss_fn=None, optimizer=None)`

  -   `layer` ：传入待对齐的 paddle 模型

  -   `module` ：传入待对齐的 torch 模型

  -   `example_inp` ：传入输入的样例数据，样例数据包含 ( paddle_input, torch_input ) 的结构，其中 `paddle_input` 和 `torch_input` 是一个 dict ，包含了需要传入给对应的 layer 和 module 的 name 到 value 的映射，即最后调用时使用 `layer(**paddle_input)` 的形式。注意顺序，paddle 在前 torch 在后。

  -   `auto_weights` : 是否使用随机数值统一初始化 paddle 与 torch 模型，默认为 `True`

  -   `layer_map` : 指定 paddle 与 torch 的 layer 映射关系，当模型结构无法完全对齐时需要通过此参数指定 layer 的映射关系，详见[LayerMap使用说明](LayerMap.md)

  -   `options` ：一个传递参数的字典

      -   `atol` ： 绝对精度误差上限，默认值为  `0`

      -   `rtol` ： 相对精度误差上限，默认值为  `1e-7`

      -   `diff_phase` ：  `"both"|"forward"|"backward"`  默认为  `"both"`。设置为  `"both"`  时，工具将比较前反向的 diff；当设置为  `"forward"`  时，仅比较前向 diff，且会跳过模型的 backward 计算过程。"backward" 仅在使用 single_step 时有效。

      -   `compare_mode` ：  `"mean"|"strict"`  默认为  `"mean"`。  `"mean"`  表示使用Tensor间误差的均值作为对齐标准；  `"strict"`  表示对Tensor进行逐数据（Elementwise）的对齐检查。

      -   `single_step` ：  `True|False`  默认为  `False`。设置为  `True`  时开启单步对齐模式，forward 过程中每一个 step 都会同步模型的输入，可以避免层间误差累积。

      > 注：
      >
      > single_step 模式下，对齐检查的逻辑会随着 diff_phase 属性的变化而不同。如果需要同时用 single_step 对齐前反向，则 padiff 将会运行模型两次，并分别进行前向和反向的 single_step 对齐检查 （single step 模式下运行模型的 forward 无法正常训练）。

  -   `loss_fn` ：由 paddle 和 torch 使用的损失函数按顺序组成的 list。在使用时，要求传入的 loss function 只接受一个参数。

  -   `optimizer` ：由 paddle 和 torch 使用的优化器或 lambda 函数按顺序组成的 list，当传入 lambda 函数，它需要同时完成 step 和clear grad的 操作。

  -   `steps` ： 支持多 step 的对齐检查，默认值为 `1`。当输入steps >1 时要求  `option["diff_phase"]`  为  `"both"`，且传入了optimizer

```py
from padiff import auto_diff
import torch
import paddle

class SimpleLayer(paddle.nn.Layer):
    # your definition

class SimpleModule(torch.nn.Module):
    # your definition


layer = SimpleLayer()
module = SimpleModule()

inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({'x': paddle.to_tensor(inp)},
     {'y': torch.as_tensor(inp) })

auto_diff(layer, module, inp, auto_weights=True, options={'atol': 1e-4, 'rtol':0, 'compare_mode': 'strict', 'single_step':False})
```



  ## 二、`assign_weight` 接口参数

  **接口功能**：将 torch 模型权重复制到 paddle 模型中，可以结合 `layer_map` 进行自定义初始化

  **函数接口签名*：`assign_weight(layer, module, layer_map=LayerMap())`

  -   `layer` ：传入待对齐的 paddle 模型

  -   `module` ：传入待对齐的 torch 模型

  -   `layer_map` ： 指定 paddle 与 torch 的 layer 映射关系，当模型结构无法完全对齐时需要通过此参数指定 layer的 映射关系；详见 [LayerMap使用说明](LayerMap.md)

```py
from padiff import assign_weight, LayerMap
import torch
import paddle

layer = SimpleLayer()
module = SimpleModule()
layer_map = LayerMap()

assign_weight(layer, module, layer_map)
```
