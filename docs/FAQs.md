- [FAQs](#faqs)
  - [使用包含随机性的op](#使用包含随机性的op)
  - [显存溢出](#显存溢出)
  - [如何设置模型device](#如何设置模型device)
  - [如何进行非fp32模型的对齐](#如何进行非fp32模型的对齐)
  - [由padiff引发的import问题](#由padiff引发的import问题)
  - [使用torch的checkpoint机制](#使用torch的checkpoint机制)
  - [assign weight 的错误](#assign-weight-的错误)


# FAQs

## 使用包含随机性的op

padiff 无法对齐包含随机性 op 的模型，例如 dropout。

测试时需要自行注释相关代码，使用 padiff 的 api 级别对齐检查可以帮助定位相关 api 的位置。



## 显存溢出

出现显存溢出时，请尝试减小 batchsize

说明：

-   目前 padiff 几乎没有额外的显存开销（除了一个额外的模型）
-   目前 padiff 会进行自动的 device 切换（如果原本模型不在 cpu 上）
    -   device 切换以及显存清理默认打开，可以通过 `export PADIFF_CUDA_MEMORY=OFF` 关闭



## 如何设置模型device

auto_diff 工具的工作与 device 无关，如果需要进行 cpu/gpu 的对齐，只需要传入device 为 cpu/gpu 的模型以及输入即可

-   在调用 paddle 模型构造函数以及 input data 初始化前，使用 `paddle.set_device(xxx)`
-   在构造 torch 模型后，使用 `torch_module = torch_module.to(xxx)`, `torch_input = torch_input.to(xxx)`



## 如何进行非fp32模型的对齐

如果模型的输入以及参数均为非fp32类型，同样可以正常对齐。对于使用 amp 的模型对齐还需进一步测试。



## 由padiff引发的import问题

此类问题可能是 padiff 开启 api 级别的对齐检查引起的

1.   尝试将 padiff 的 import 后置
2.   若仍不能避免错误，可通过  `export PADIFF_API_CHECK=OFF` 关闭API级别的对齐检查，并向我们反馈



## 使用torch的checkpoint机制

目前暂不支持 torch 的 checkpoint 机制（在反向时 rerun 部分前向逻辑）

该机制会导致反向梯度无法捕获，因此 padiff 现在会报错



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
> 2. 使用 LayerMap 指定顶层模型的一一对应关系，并自定义初始化函数，详见链接 [自定义初始化函数的方法](SpecialInit.md)

**因模型权重定义顺序导致的错误**

在寻找到对应 layer/module 后，其中的 parameter 定义顺序不一致，导致拷贝权重出错。该问题发生时，一般表现为 parameter shape 不一致。

> 解决方案：
>
> 使用 LayerMap 指定顶层模型的一一对应关系，并自定义初始化函数，详见链接 [自定义初始化函数的方法](SpecialInit.md)

**因模型结构不对应导致的错误**

模型结构本就不对应，在这种情况下，也需要使用 LayerMap

> 解决方案：
>
> 使用 LayerMap 的 ignore 功能忽略部分模型，并在必要时添加一一对应关系，详见链接 [LayerMap的使用](LayerMap.md)

**没有找到相应的 special init 函数**

1. 可以检查本 repo 中的[相关文档](SpecialInit.md)中记录的已支持的特殊初始化接口，若需求的接口不在其中，则需要自行添加

2. 当前版本下，special init 对 base_model 和 raw_model 有顺序要求，其语义为将 base_model 的权重拷贝到 raw_model，若传入模型顺序相反，可能导致无法找到对应的 special init 初始化函数
