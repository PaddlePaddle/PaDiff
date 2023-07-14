- [FAQs](#faqs)
  - [自动跳过 wrap layer](#自动跳过-wrap-layer)
  - [assign weight 的错误](#assign-weight-的错误)
  - [出现 CUBLAS ERROR](#出现-cublas-error)
  - [使用包含随机性的op](#使用包含随机性的op)
  - [显存溢出](#显存溢出)
  - [如何设置模型device](#如何设置模型device)
  - [如何进行非fp32模型的对齐](#如何进行非fp32模型的对齐)
  - [由padiff引发的import问题](#由padiff引发的import问题)


# FAQs

## 自动跳过 wrap layer
在老版本的 PaDiff 工具中，会自动过滤 wrap layer （即没有parameter的layer），但这种行为可能会导致用户的困惑，而且可能会改变记录的模型结构，因此在 develop 版本中，我们取消了这种过滤行为，这可能导致老版本能直接对齐的模型在 develop 版本下无法直接对齐。

但在 torch 和 paddle 模型对齐时，不过滤可能需要用户自己手动添加很多 ignore 语句，为了方便与 torch 模型的对齐，可以通过使用环境变量 "PADIFF_SIKP_WRAP_LAYER" 来开启过滤行为（默认是关闭的）。

`export PADIFF_SIKP_WRAP_LAYER=TRUE`

## assign weight 的错误

**assign weight 做了什么**
1. assign weight 首先尝试寻找对应的 layer/module
2. assign weight 尝试在对应的 layer/module 中，同步遍历它们的 parameters
3. assign weight 会略过不包含 parameter 的 layer/module (也会略过 LayerMap 指定的部分)

**因模型权重实现方案不同导致的错误**

padiff允许模型使用 buffer，embedding，但它们的使用必须对应。即一方使用了 parameter/buffer/embedding 另一方也应该使用 parameter/buffer/embedding，否则 padiff 无法判断模型应如何初始化。

例子：
下图中 paddle 和 torch 的 OVDeformableTransformer 是对应的，但由于 torch 模型的 OVDeformableTransformer 使用 parameter，而 paddle 模型的OVDeformableTransformer 使用 Embedding，导致 paddle 模型没有 parameter，被 padiff 略过，因此报错信息指向了 paddle 模型中下方的 Linear

（在当前版本下，被略过的 layer 会被标注 skip）

![Pasted Graphic 5](https://user-images.githubusercontent.com/79986504/227197672-1ecc6b74-d796-447f-8508-2bcf6cbb6bc6.png)


> 解决方案：
>
> 1. 修改模型代码，使用对应的权重实现方案
>
> 2. 使用 layer_map 指定顶层模型的一一对应关系，并自定义初始化函数，详见链接 [自定义初始化函数的方法](SpecialInit.md)

**因模型权重定义顺序导致的错误**

在寻找到对应 layer/module 后，其中的 parameter 定义顺序不一致，导致拷贝权重出错。该问题发生时，一般表现为 parameter shape 不一致。

> 解决方案：
>
> 使用 layer_map 指定顶层模型的一一对应关系，并自定义初始化函数，详见链接 [自定义初始化函数的方法](SpecialInit.md)

**因模型结构不对应导致的错误**

模型结构本就不对应时

> 解决方案：
>
> 使用黑白名单机制忽略部分模型，并在必要时使用 [layer_map 机制]() 添加一一对应关系

**没有找到相应的 special init 函数**

1. 可以检查本 repo 中的[相关文档](SpecialInit.md)中记录的已支持的特殊初始化接口，若需求的接口不在其中，则需要自行添加

2. 当前版本下，special init 对 base_model 和 raw_model 有顺序要求，其语义为将 base_model 的权重拷贝到 raw_model，若传入模型顺序相反，可能导致无法找到对应的 special init 初始化函数



## 出现 CUBLAS ERROR

请确认当前使用的 cuda 版本在当前 paddle 支持的 cuda 版本之内，（cuda 11.4 不在支持范围内）
详见 https://www.paddlepaddle.org.cn/


## 使用包含随机性的op

padiff 无法对齐包含随机性 op 的模型，例如 dropout。

测试时需要自行注释相关代码，使用 padiff 的 api 级别对齐检查可以帮助定位相关 api 的位置。



## 显存溢出

使用 auto_diff 接口出现显存溢出时，请尝试减小 batchsize，或尝试使用离线对齐方案。
在使用离线对齐方案时出现显存溢出，请检查模型原本所需的显存资源量。



## 如何设置模型device

auto_diff 工具的工作与 device 无关，如果需要进行 cpu/gpu 的对齐，只需要传入device 为 cpu/gpu 的模型以及输入即可

-   在调用 paddle 模型构造函数以及 input data 初始化前，使用 `paddle.set_device(xxx)`
-   在构造 torch 模型后，使用 `torch_module = torch_module.to(xxx)`, `torch_input = torch_input.to(xxx)`



## 如何进行非fp32模型的对齐

如果模型的输入以及参数均为非fp32类型，同样可以正常对齐。

在使用离线对齐解决方案时，可以正常使用 amp guard。



## 由padiff引发的import问题

此类问题可能是 padiff 开启 api 级别的对齐检查引起的

1.   尝试将 padiff 的 import 后置
2.   若仍不能避免错误，可以先关闭 api 级别对齐检查再尝试（默认是关闭的）
