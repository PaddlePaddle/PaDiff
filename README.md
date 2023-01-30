# PaDiff
**P**addle **Auto**matically **Diff** precision toolkits.


## 最近更新
- 支持使用自定义损失函数

## 简介
PaDiff是基于PaddlePaddle与PyTorch的模型精度对齐工具。传入Paddle与Torch模型，PaDiff将对训练过程中的所有中间结果以及训练后的模型权重进行对齐检查，并以调用栈的形式提示模型第一次出现精度diff的位置。


## 安装
PaDiff v0.1 版本已发布，可通过如下命令安装：
```
pip install padiff
```

尝鲜版或开发者推荐clone源码并使用如下命令安装：
```
python setup.py install
```
## 使用说明

### auto_diff 使用接口与参数说明

接口函数签名：`auto_diff(layer, module, example_inp, auto_weights=True, layer_mapping={}, options={})`

-   layer：传入paddle模型

-   module：传入torch模型

-   example_inp：传入输入的样例数据，样例数据包含 ( paddle_input, torch_input ) 的结构，其中paddle_input和torch_input是一个dict，包含了需要传入给对应的layer和module的name到value的映射，即最后调用时使用 layer(**paddle_input) 的形式。注意顺序，paddle在前torch在后。

-   auto_weights: 是否使用随机数值统一初始化paddle与torch模型，默认为True

-   layer_mapping: 指定paddle与torch的layer映射关系，当模型结构无法完全对齐时需要通过此参数指定layer的映射关系。

-   options：一个传递参数的字典

       -   "atol": 绝对精度误差上限，默认值为 `0`

       -   "rtol": 相对精度误差上限，默认值为 `1e-7`

       -   "diff_phase": `"both"|"forward"` 默认为 `"both"`。设置为 `"both"` 时，工具将比较前反向的diff；当设置为 `"forward"` 时，仅比较前向diff，且会跳过模型的backward计算过程。

       -   "compare_mode": `"mean"|"strict"` 默认为 `"mean"`。 `"mean"` 表示使用Tensor间误差的均值作为对齐标准； `"strict"` 表示对Tensor进行逐数据（Elementwise）的对齐检查。

       -   "single_step": `True|False` 默认为 `False`。设置为 `True` 时开启单步对齐模式，forward过程中每一个step都会同步模型的输入，可以避免层间误差累积。注意：开启single_step后将不会触发backward过程，"diff_phase"参数将被强制设置为 `"forward"`。

- loss_fn：由paddle和torch使用的损失函数按顺序组成的list。在使用时，要求传入的loss function只接受一个参数。

### 注意事项与样例代码：

#### case1 模型定义与auto_diff基本使用
-   在使用auto_diff时，需要传入paddle模型与torch模型，在模型定义时，需要将forward中所使用的子模型在 `__init__` 函数中定义，并保证其中的子模型定义顺序一致，具体可见下方示例代码

```py
from padiff import auto_diff
import torch
import paddle

# 使用paddle与torch定义相同结构的模型: SimpleLayer 和 SimpleModule
# 样例模型结构为:
#       x -> linear1 -> x -> relu -> x -> add -> linear2 -> output
#       |                                  |
#       |----------------------------------|

# 注意：两个模型定义顺序都是 linear1 linear2 ReLU，顺序必须对齐，submodule内部的定义也是一样。
class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(100, 100)
        self.linear2 = paddle.nn.Linear(100, 10)
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x

class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)
        self.linear2 = torch.nn.Linear(100, 10)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        resdual = x
        x = self.linear1(x)
        x = self.act(x)
        x = x + resdual
        x = self.linear2(x)
        return x


layer = SimpleLayer()
module = SimpleModule()
inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({'x': paddle.to_tensor(inp)},  ## <-- 注意顺序，paddle_input, torch_input 的形式。
       {'y': torch.as_tensor(inp) })
auto_diff(layer, module, inp, auto_weights=True, options={'atol': 1e-4, 'rtol':0, 'compare_mode': 'strict', 'single_step':False})
```

#### case2 使用 layer_mapping
- layer_mapping可以指定两个sublayer之间的对应关系，这样做可以略过sublayer内部的数据对齐
- 在auto_diff内支持部分sublayer的权重初始化，对于不支持的sublayer，将在auto_diff输出信息中进行提示。若出现了相关输出信息，用户需要自行初始化sublayer


```py
# 由于paddle与torch的MultiHeadAttention无法直接对齐
# 需要使用 layer_mapping 功能

class SimpleLayer(paddle.nn.Layer):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.attn = paddle.nn.MultiHeadAttention(16, 1)

    def forward(self, q, k, v):
        x = self.attn(q, k, v)
        return x

class SimpleModule(torch.nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.attn = torch.nn.MultiheadAttention(16, 1, batch_first=True)

    def forward(self, q, k, v):
        x, _ = self.attn(q, k, v)
        return x


layer = SimpleLayer()
module = SimpleModule()

# 在layer_mapping中指定无法对齐的sublayer。注意，在字典中应使用python obj
# layer_mapping对kv顺序没有要求
# 目前 auto_diff 已支持 MultiHeadAttention 的权重自动初始化，因此此处无需其他操作

layer_mapping = {layer.attn: module.attn}

inp = paddle.rand((2, 4, 16)).numpy()
inp = (
{"q": paddle.to_tensor(inp), "k": paddle.to_tensor(inp), "v": paddle.to_tensor(inp)},
{"q": torch.as_tensor(inp), "k": torch.as_tensor(inp), "v": torch.as_tensor(inp)},
)


auto_diff(layer, module, inp, auto_weights=True, layer_mapping=layer_mapping, options={"atol": 1e-4})
```

#### case3 使用loss_fn
- 不指定loss_fn时，将使用auto_diff内置的一个fake loss function进行计算
```py
class SimpleLayer(paddle.nn.Layer):
    # ...
class SimpleModule(torch.nn.Module):
    # ...

layer = SimpleLayer()
module = SimpleModule()
inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({"x": paddle.to_tensor(inp)}, {"x": torch.as_tensor(inp)})
label = paddle.rand([10]).numpy().astype("float32")

# 自定义loss函数，若输入不止一个，可以使用partial或者闭包等方法得到单输入的函数，再传入
def paddle_loss(inp, label):
    label = paddle.to_tensor(label)
    return inp.mean() - label.mean()
def torch_loss(inp, label):
    label = torch.tensor(label)
    return inp.mean() - label.mean()

auto_diff(layer, module, inp, auto_weights=True, options={"atol": 1e-4}, loss_fn=[
       partial(paddle_loss, label=label),
       partial(torch_loss, label=label)
])

# 使用 paddle 和 torch 提供的损失函数时，使用方法一致
paddle_mse = paddle.nn.MSELoss()
torch_mse = torch.nn.MSELoss()

auto_diff(layer, module, inp, auto_weights=True, options={"atol": 1e-4}, loss_fn=[
       partial(paddle_mse, label=paddle.to_tensor(label)),
       partial(torch_mse, target=torch.tensor(label))
])
```



## 输出信息示例

-   正确对齐时的输出信息：
    auto_diff将输出paddle与torch模型输出结果之间的最大diff值

       ```
       [AutoDiff] Start auto_diff, may need a while to generate reports...
       [AutoDiff] Max output diff is 6.103515625e-05

       [AutoDiff] weight and weight.grad is compared.
       [AutoDiff] forward 4 steps compared.
       [AutoDiff] bacward 4 steps compared.
       [AutoDiff] SUCCESS !!!
       ```

-   模型对齐失败时的输出信息：

       -   训练后，模型权重以及梯度的对齐情况，具体信息将记录在当前路径的diff_log文件夹下
       -   注意，每次调用auto_diff后，diff_log下的报告会被覆盖
       -   在训练过程中首先出现diff的位置（在forward过程或backward过程）
       -   paddle与torch的调用栈，可以追溯到第一次出现不对齐的代码位置

       ```
       [AutoDiff] Start auto_diff, may need a while to generate reports...
       [AutoDiff] Max output diff is 3.0571913719177246

       [AutoDiff] Differences in weight or grad !!!
       [AutoDiff] Check reports at `/workspace/diff_log`

       [AutoDiff] FAILED !!!
       [AutoDiff]     Diff found in `Forward  Stagy` in step: 0, net_id is 1 vs 1
       [AutoDiff]     Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>


       Paddle Stacks:
       =========================
              ...
              File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1022    __call__
                     return self._dygraph_call_func(*inputs, **kwargs)
              File pptest.py: 37    forward
                     x = self.linear1(x)
              ...
       Torch  Stacks:
       =========================
              ...
              File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1151    _call_impl
                     hook_result = hook(self, input, result)
              File pptest.py: 58    forward
                     x = self.linear1(x)
              ...
       ```

-   模型对齐失败且失败位置在反向过程时：
    结合输出文本 " [AutoDiff]     Diff found in `Backward Stagy` in step: 0, net_id is 2 vs 2 "，可知模型前向能够对齐，但是反向过程出现diff。结合调用栈信息可以发现，diff出现在linear2对应的反向环节出现diff


       ```
       [AutoDiff] Start auto_diff, may need a while to generate reports...
       [AutoDiff] Max output diff is 1.71661376953125e-05

       [AutoDiff] Differences in weight or grad !!!
       [AutoDiff] Check reports at `/workspace/diff_log`

       [AutoDiff] forward 4 steps compared.
       [AutoDiff] FAILED !!!
       [AutoDiff]     Diff found in `Backward Stagy` in step: 0, net_id is 2 vs 2
       [AutoDiff]     Type of layer is  : <class 'torch.nn.modules.linear.Linear'> vs <class 'paddle.nn.layer.common.Linear'>


       Paddle Stacks:
       =========================
              ...
              File /workspace/env/env3.7/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py: 1022    __call__
                     return self._dygraph_call_func(*inputs, **kwargs)
              File pptest.py: 52    forward
                     x3 = self.linear2(x)
              ...
       Torch  Stacks:
       =========================
              ...
              File /workspace/env/env3.7/lib/python3.7/site-packages/torch/nn/modules/module.py: 1151    _call_impl
                     hook_result = hook(self, input, result)
              File pptest.py: 66    forward
                     x3 = self.linear2(x)
              ...
       ```
## 调试建议

如果遇到了 auto_diff 函数提示某个 layer 没有对齐，可以考虑如下几个 debug 建议：

- 如果报告不是上述的Success或者是Failed，那么说明模型没有满足预定的假设。可以结合 报错信息 进行分析。常见问题是：Torch 模型和 Paddle 模型没有满足Layer定义的一一对应假设。可以通过 print 两个模型来进行假设验证，一个满足一一对应的例子应该如下图（Layer的名字可以不用相同）![e11cd8bfbcdaf5e19a3894cecd22d212](https://user-images.githubusercontent.com/16025309/209917443-e5c21829-f4a6-4bdf-a621-b123c11e83d6.jpg)

- 如果显示精度有diff，先分析Paddle和Torch的调用栈，找到对应的源码并分析他们在逻辑上是否是对应的Layer，如果不是对应的Layer，那么说明 Torch 模型和 Paddle 模型没有满足Layer定义的一一对应假设。如图 <img width="875" alt="3d569899c42f69198f398540dec89012" src="https://user-images.githubusercontent.com/16025309/209917231-717c8e88-b3d8-41bc-b6a9-0330d0d9ed50.png">

- 如果模型没有满足Layer定义的一一对应假设，可以通过`layer_mapping`指定Layer的映射关系。例如下图中共有三个SubLayer没有一一对齐，因此需要通过`layer_mapping`指定三个地方的映射关系。 如图 <img width="788" alt="image" src="https://user-images.githubusercontent.com/40840292/212643420-b30d5d6f-3a26-4a41-8dc2-7b3e6622c1d5.png">

       ```python
       layer = SimpleLayer()
       module = SimpleModule()

       layer_mapping = {
       layer.transformer.encoder.layers[0].self_attn: module.transformer.encoder.layers[0].self_attn,
       layer.transformer.decoder.layers[0].self_attn: module.transformer.decoder.layers[0].self_attn,
       layer.transformer.decoder.layers[0].cross_attn: module.transformer.decoder.layers[0].multihead_attn,
       } # object pair的形式

       auto_diff(layer, module, inp, auto_weights=False, layer_mapping=layer_mapping, options={"atol": 1e-4})
       ```

- 如果不是上述的问题，那么可以考虑进行debug，比如构造最小复现样例或者是pdb调试等等。

- 如果上述无法解决您的问题，或者认为找不到问题，可以考虑给本仓库提一个Issue。
