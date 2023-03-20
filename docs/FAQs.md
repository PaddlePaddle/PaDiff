- [FAQs](#faqs)
  - [使用包含随机性的op](#使用包含随机性的op)
  - [显存溢出](#显存溢出)
  - [如何设置模型device](#如何设置模型device)
  - [如何进行非fp32模型的对齐](#如何进行非fp32模型的对齐)
  - [由padiff引发的import问题](#由padiff引发的import问题)
  - [使用torch的checkpoint机制](#使用torch的checkpoint机制)
  - [使用buffer](#使用buffer)
  - [调试建议](#调试建议)


# FAQs

## 使用包含随机性的op

padiff无法对齐包含随机性op的模型，例如dropout。

测试时需要自行注释相关代码，使用padiff的api级别对齐检查可以帮助定位相关api的位置。



## 显存溢出

出现显存溢出时，请尝试减小batchsize

说明：

-   目前padiff几乎没有额外的显存开销（除了一个额外的模型）
-   由于torch和paddle都有自己的显存管理机制，因此显存可能无法立即释放，导致显存溢出的问题



## 如何设置模型device

auto_diff 工具的工作与 device 无关，如果需要进行 cpu/gpu 的对齐，只需要传入device 为 cpu/gpu 的模型以及输入即可

-   在调用paddle模型构造函数以及 input data 初始化前，使用 `paddle.set_device(xxx)`
-   在构造torch模型后，使用 `torch_module = torch_module.to(xxx)`, `torch_input = torch_input.to(xxx)`



## 如何进行非fp32模型的对齐

如果模型的输入以及参数均为非fp32类型，同样可以正常对齐。对于使用 amp 的模型对齐还需进一步测试。



## 由padiff引发的import问题

此类问题可能是 padiff 开启 api 级别的对齐检查引起的

1.   尝试将 padiff 的 import 后置
2.   若仍不能避免错误，可通过  `export PADIFF_API_CHECK=OFF` 关闭API级别的对齐检查，并向我们反馈



## 使用torch的checkpoint机制

目前暂不支持 torch 的 checkpoint 机制（在反向时 rerun 部分前向逻辑）

该机制会导致反向梯度无法捕获，因此 padiff 现在会报错



## 使用buffer

padiff 允许使用 buffer：

1.   torch 与 paddle 的模型写法必须对齐（都使用 buffer 或都使用 param ，不能一边 buffer 一边 param ）
2.   padiff 不会修改 buffer 的值



## 调试建议

如果遇到了 auto_diff 函数提示某个 layer 没有对齐，可以考虑如下几个 debug 建议：

-   如果报告不是Success或者是Failed，那么说明模型没有满足预定的假设。可以结合 报错信息 进行分析。常见问题是：Torch 模型和 Paddle 模型没有满足Layer定义的一一对应假设。可以通过 print 两个模型来进行假设验证，一个满足一一对应的例子应该如下图（Layer的名字可以不用相同）![e11cd8bfbcdaf5e19a3894cecd22d212](https://user-images.githubusercontent.com/16025309/209917443-e5c21829-f4a6-4bdf-a621-b123c11e83d6.jpg)

-   如果显示精度有diff，先分析Paddle和Torch的调用栈，找到对应的源码并分析他们在逻辑上是否是对应的Layer，如果不是对应的Layer，那么说明 Torch 模型和 Paddle 模型没有满足Layer定义的一一对应假设。如图  ![3d569899c42f69198f398540dec89012](https://user-images.githubusercontent.com/16025309/209917231-717c8e88-b3d8-41bc-b6a9-0330d0d9ed50.png)

-   如果模型没有满足Layer定义的一一对应假设，可以通过`layer_map`指定Layer的映射关系。例如下图中共有三个SubLayer没有一一对齐，因此需要通过`layer_map`指定三个地方的映射关系。 如图  ![image](https://user-images.githubusercontent.com/40840292/212643420-b30d5d6f-3a26-4a41-8dc2-7b3e6622c1d5.png)

```
     layer = SimpleLayer()
     module = SimpleModule()

     layer_map = {
     layer.transformer.encoder.layers[0].self_attn: module.transformer.encoder.layers[0].self_attn,
     layer.transformer.decoder.layers[0].self_attn: module.transformer.decoder.layers[0].self_attn,
     layer.transformer.decoder.layers[0].cross_attn: module.transformer.decoder.layers[0].multihead_attn,
     } # object pair的形式

     auto_diff(layer, module, inp, auto_weights=False, layer_map=layer_map, options={"atol": 1e-4})

```

-   如果不是上述的问题，那么可以考虑进行debug，比如构造最小复现样例或者是pdb调试等等。

-   如果上述无法解决您的问题，或者认为找不到问题，可以考虑给本仓库提一个Issue。
