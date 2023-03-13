## auto_diff 接口参数

  接口功能：进行模型对齐检查

  接口函数签名：`auto_diff(layer, module, example_inp, auto_weights=False, steps=1, options={}, layer_map={}, loss_fn=None, optimizer=None)`

  -   layer：传入paddle模型

  -   module：传入torch模型

  -   example_inp：传入输入的样例数据，样例数据包含 ( paddle_input, torch_input ) 的结构，其中paddle_input和torch_input是一个dict，包含了需要传入给对应的layer和module的name到value的映射，即最后调用时使用 layer(**paddle_input) 的形式。注意顺序，paddle在前torch在后。

  -   auto_weights: 是否使用随机数值统一初始化paddle与torch模型，默认为True

  -   layer_map: 指定paddle与torch的layer映射关系，当模型结构无法完全对齐时需要通过此参数指定layer的映射关系。

      -   layer_map的具体使用方法详见[LayerMap使用说明]()
  -   options：一个传递参数的字典

      -   “atol”: 绝对精度误差上限，默认值为  `0`

      -   “rtol”: 相对精度误差上限，默认值为  `1e-7`

      -   “diff_phase”:  `"both"|"forward"`  默认为  `"both"`。设置为  `"both"`  时，工具将比较前反向的diff；当设置为  `"forward"`  时，仅比较前向diff，且会跳过模型的backward计算过程。

      -   “compare_mode”:  `"mean"|"strict"`  默认为  `"mean"`。  `"mean"`  表示使用Tensor间误差的均值作为对齐标准；  `"strict"`  表示对Tensor进行逐数据（Elementwise）的对齐检查。

      -   “single_step”:  `True|False`  默认为  `False`。设置为  `True`  时开启单步对齐模式，forward过程中每一个step都会同步模型的输入，可以避免层间误差累积。注意：开启single_step后将不会触发backward过程，`"diff_phase"` 参数将被强制设置为  `"forward"`。

  -   loss_fn：由paddle和torch使用的损失函数按顺序组成的list。在使用时，要求传入的loss function只接受一个参数。

  -   optimizer：由paddle和torch使用的优化器或lambda函数按顺序组成的list，当传入lambda函数，它需要同时完成step和clear grad的操作。

  -   steps: 支持多step的对齐检查，默认值为1。当输入steps >1 时要求  `option["diff_phase"]`  为  `"both"`，且传入了optimizer

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



  ## assign_weight 接口参数

  接口功能：将torch模型权重复制到paddle模型中，可以结合layer_map进行自定义初始化

  函数接口签名：`assign_weight(layer, module, layer_map=LayerMap())`

  -   layer：传入paddle模型

  -   module：传入torch模型

  -   layer_map: 指定paddle与torch的layer映射关系，当模型结构无法完全对齐时需要通过此参数指定layer的映射关系。

      -   layer_map的具体使用方法详见 [LayerMap使用说明](*)

```py
from padiff import assign_weight, LayerMap
import torch
import paddle

layer = SimpleLayer()
module = SimpleModule()
layer_map = LayerMap()

assign_weight(layer, module, layer_map)
```
