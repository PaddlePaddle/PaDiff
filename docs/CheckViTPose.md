- [使用PaDiff工具对齐ViTPose流程示例](#使用padiff工具对齐vitpose流程示例)
  - [0. 准备工作](#0-准备工作)
  - [1. 单 step 的前向对齐](#1-单-step-的前向对齐)
    - [关于输入数据](#关于输入数据)
    - [关于 LayerMap](#关于-layermap)
    - [关于参数设置](#关于参数设置)
    - [示例代码](#示例代码)
  - [2. 损失函数精度验证](#2-损失函数精度验证)
    - [关于参数设置](#关于参数设置-1)
    - [示例代码](#示例代码-1)
  - [3.  多 step 对齐检查（以及 optimizer 精度验证）](#3--多-step-对齐检查以及-optimizer-精度验证)
    - [工具逻辑说明：关于 optimizer 的精度检查](#工具逻辑说明关于-optimizer-的精度检查)
    - [关于参数设置](#关于参数设置-2)
    - [示例代码](#示例代码-2)
  - [4. 出现 diff 时进行精确定位](#4-出现-diff-时进行精确定位)
    - [工具逻辑说明：关于 single\_step 模式](#工具逻辑说明关于-single_step-模式)
    - [关于参数设置](#关于参数设置-3)
    - [示例代码](#示例代码-3)



# 使用PaDiff工具对齐ViTPose流程示例

**阅读本文档前，建议先阅读[Tutorial](Tutorial.md)，对工具的使用方法有一个基本的认识**

## 0. 准备工作

在使用PaDiff工具前，需要自行编写部分代码，包括：

1.   加载（或定义） paddle 模型以及 torch 模型
2.   准备 dataloader 逻辑（若必要）

完成准备后，使用工具进行对齐的步骤是基本固定的：

1.   得到 paddle 以及 torch 模型
2.   取得模型的输入数据
3.   生成 layer_map 结构
4.   调用 assign_weight 初始化模型权重
5.   调用 auto_diff 接口进行对齐



以下是加载ViTPose模型的代码示例

```py
import torch
import paddle
import numpy as np
from functools import partial

# import 需要的组件

import sys
sys.path.append("./PaddleDetection")
from ppdet.modeling.backbones.vitPosenet import VisionTransformer as vit_paddle
from ppdet.modeling.losses.keypoint_loss import KeyPointMSELoss as paddle_loss
#from ppdet.modeling.backbones.vision_transformer import VisionTransformer as vit_paddle
from ppdet.modeling.heads import TopdownHeatmapSimpleHead as head_paddle
from ppdet.core.workspace import create,load_config
#from ppdet.metrics import KeyPointTopDownCOCOEval


sys.path.append("./ViTPose")
from mmpose.models.backbones import ViT as vit_torch
from mmpose.models.heads import TopdownHeatmapSimpleHead as head_torch
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models.losses.mse_loss import JointsMSELoss as torch_loss
from mmcv import Config

from PaDiff.padiff import auto_diff


# 定义用于对齐的模型

class tiny_pose_paddle(paddle.nn.Layer):
    def __init__(self):
        super(tiny_pose_paddle,self).__init__()
        img_size=[256, 192]
        patch_size=16
        embed_dim=768
        depth=12
        num_heads=12
        ratio=1
        mlp_ratio=4
        qkv_bias=True
        drop_rate=0.0
        drop_path_rate=0.3
        final_norm=True
        with_fpn=False
        use_abs_pos_emb=True
        use_sincos_pos_emb=False
        epsilon=0.000001 # 1e-6
        use_vitPose=True

        in_channels=768
        num_deconv_layers=2
        num_deconv_filters=(256,256)
        num_deconv_kernels=(4,4)
        upsample=0
        extra=dict(final_conv_kernel=1, )

        self.backbone = vit_paddle(img_size=(256,192),
                            qkv_bias=qkv_bias,
                            drop_path_rate=drop_path_rate,
                            epsilon=epsilon)
        self.head = head_paddle(in_channels=in_channels,
                                    num_deconv_layers=num_deconv_layers,
                                    num_deconv_filters=num_deconv_filters,
                                    num_deconv_kernels=num_deconv_kernels,
                                    upsample=upsample,
                                    extra=extra)

    def forward(self,x):
        x = self.backbone.forward_features(x)
        x = self.head(x)

        return x


class tiny_pose_torch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        qkv_bias=True
        drop_path_rate=0.3

        in_channels=768
        num_deconv_layers=2
        num_deconv_filters=(256,256)
        num_deconv_kernels=(4,4)
        upsample=0
        extra=dict(final_conv_kernel=1, )

        self.backbone = vit_torch(img_size=(256,192),
                            qkv_bias=qkv_bias,
                            drop_path_rate=drop_path_rate)
        self.head = head_torch(in_channels=in_channels,
                                    num_deconv_layers=num_deconv_layers,
                                    num_deconv_filters=num_deconv_filters,
                                    num_deconv_kernels=num_deconv_kernels,
                                    upsample=upsample,
                                    extra=extra,
                                    out_channels=17)

    def forward(self,x):
        x = self.backbone.forward_features(x)
        x = self.head(x)
        
        return x
     
# 编写构造 dataloader 的逻辑

def build_torch_data_pipeline():
    data_root= 'coco'
    config_file='ViTPose_base_simple_coco_256x192.py'
    cfg = Config.fromfile(config_file)
    cfg.seed=0
    cfg.data.train.ann_file=f'{data_root}/annotations/person_keypoints_train2017_10.json'
    cfg.data.samples_per_gpu=1
    cfg.data.workers_per_gpu=1
    cfg.data_cfg.use_gt_bbox=True
   
    dataset = build_dataset(cfg.data.train)
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(
            seed=cfg.get('seed'),
            drop_last=False,
            dist=False,
            shuffle=False,
            num_gpus=1),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'samples_per_gpu',
                   'workers_per_gpu',
                   'shuffle',
                   'seed',
                   'drop_last',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))
    
    data_loaders = build_dataloader(dataset, **train_loader_cfg) 

    return dataset,data_loaders

def build_paddle_data_pipeline():
    config_file = 'vitpose_base_simple_coco_256x192.yml'
    cfg = load_config(config_file)
 
    capital_mode = 'train'
    capital_mode = capital_mode.capitalize()
    dataset = cfg['{}Dataset'.format(capital_mode)] = create(
                '{}Dataset'.format(capital_mode))()
    
    loader = create('{}Reader'.format(capital_mode))(
                dataset, cfg.worker_num)
   
    return dataset,loader
```



## 1. 单 step 的前向对齐

单 step 的模型前向对齐检查中，不更新权重，对每一组输入数据进行一次独立的对齐检查。对应代码示例见下方的代码块。

### 关于输入数据

工具提供的 auto_diff 接口不能接受 dataloader 作为输入，若希望验证模型在不同数据上的单 step 精度表现，需要分多次取出数据，然后多次调用auto_diff接口。

注意保证传入的输入数据是对应的。

### 关于 LayerMap

可以通过工具提供的 LayerMap 类的 "auto" 成员函数自动搜索需要的信息。

在调用 auto 接口时，将在终端打印具体的对应情况，若 auto 接口提供的映射不正确，则仍需要手动编写 [LayerMap](LayerMap.md)。同时，也可能需要[自定义初始化函数](SpecialInit.md)来正确初始化模型权重。

### 关于参数设置

除了必需的输入之外，可以注意以下几个参数的设置：

1.   auto_weights

     在下方示例代码中，需要对不同的数据进行单 step 对齐检查，不需要重复进行权重初始化行为。因此设置为 False

2.   options

     -   single_step 选项：在模型对齐的开始，建议关闭 single_step ，确认模型存在 diff 时再打开它来帮助定位具体的 diff 位置
     -   diff_phase 选项：由于目前的任务是单 step 的前向对齐检查，设置 diff_phase 选项为 "forward"，可以只定位模型的前向逻辑，跳过backward 部分（不会更新模型权重）

### 示例代码

```py
from padiff import auto_diff, assign_weight, LayerMap()

def test_forward():

  	# step 1: 得到 paddle 以及 torch 模型
    pretrained = './ViTPose/vitpose-b.pth'
    state_dict = torch.load(pretrained)['state_dict']
    torch_model = tiny_pose_torch()
    torch_model.load_state_dict(torch_new_state_dict)
    torch_model.eval()
    
    paddle_model = tiny_pose_paddle()
    paddle_model.eval()

    # step 2: 构造 dataloader
    print("torch dateloader build!")
    torch_dataset,torch_dataloader = build_torch_data_pipeline()

    print("paddle dateloader build!")
    paddle_dataset,paddle_dataloader = build_paddle_data_pipeline()
    
    # step 3: 构造 layer_map
    layer_map = LayerMap()
    layer_map.auto(paddle_model, torch_model)
    
    # step 4: 使用工具接口初始化 paddle 模型权重 （复制torch权重到paddle模型）
    assign_weight(paddle_model, torch_model, layer_map)

    # step 5: 取得对应的输入数据
    for idx, (paddle_batch, torch_batch
                ) in enumerate(zip(paddle_dataloader, torch_dataloader)):

        inp = ({'x': paddle_batch['image']},  
               {'x': torch_batch['img']})
				
        # step 6: 调用 auto_diff 接口
        result = auto_diff(
          	paddle_model,
          	torch_model,
          	inp,
          	auto_weights=False,
          	layer_map=layer_map,
          	options={
            	  'atol':0.0, 
              	'rtol':1e-5,
              	'single_step':False,
              	'diff_phase':'forward',
            }
        )
        
        if result == False:
          	break
```



## 2. 损失函数精度验证

在确认模型功能初步对齐后，可以添加指定的损失函数参与对齐，以保证损失函数的精度。工具将检查损失函数的输出是否对齐，但不会检查损失函数的内部逻辑，当损失函数的输出精度不对齐时，将打印相应的 log 信息进行提示。

具体使用方法详见[Optimizer的使用](Tutorial.md#32-使用optimizer)，以下提供简要介绍。

### 关于参数设置

需要主动传入 loss_fn 参数，该参数为一个列表，其中依次为 paddle 和 torch 各自的损失函数

注意：损失函数可以是一个 lambda， 它只能接受一个输入参数（即模型的输出），并返回一个 scale

### 示例代码

```py
from padiff import auto_diff, assign_weight, LayerMap()

def test_forward():

  	# step 1: 得到 paddle 以及 torch 模型
    pretrained = './ViTPose/vitpose-b.pth'
    state_dict = torch.load(pretrained)['state_dict']
    torch_model = tiny_pose_torch()
    torch_model.load_state_dict(torch_new_state_dict)
    torch_model.eval()
    
    paddle_model = tiny_pose_paddle()
    paddle_model.eval()

    # step 2: 构造 dataloader
    print("torch dateloader build!")
    torch_dataset,torch_dataloader = build_torch_data_pipeline()

    print("paddle dateloader build!")
    paddle_dataset,paddle_dataloader = build_paddle_data_pipeline()
    
    # step 3: 构造 layer_map
    layer_map = LayerMap()
    layer_map.auto(paddle_model, torch_model)
    
    # step 4: 使用工具接口初始化 paddle 模型权重 （复制torch权重到paddle模型）
    assign_weight(paddle_model, torch_model, layer_map)

    # step 5: 定义损失函数
		def paddle_loss(input):
      	# ...
    
    def torch_loss(input):
      	# ...

    # step 6: 取得对应的输入数据
    for idx, (paddle_batch, torch_batch
                ) in enumerate(zip(paddle_dataloader, torch_dataloader)):

        inp = ({'x': paddle_batch['image']},  
               {'x': torch_batch['img']})
				
        # step 7: 调用 auto_diff 接口
        result = auto_diff(
          	paddle_model,
          	torch_model,
          	inp,
          	auto_weights=False,
          	layer_map=layer_map,
          	options={
            	  'atol':0.0, 
              	'rtol':1e-5,
              	'single_step':False,
              	'diff_phase':'forward',
            },
          	loss_fn=[								# 传入 loss 函数
              	paddle_loss,
              	torch_loss,
            ],
        )
        
        if result == False:
          	break
```



## 3.  多 step 对齐检查（以及 optimizer 精度验证）

多 step 的对齐检查意味着需要在每一个 step 间更新模型权重，然后进行下一个step 的对齐，相对于单 step 的模型对齐检查更复杂 。auto_diff 接口支持传入指定的 optimizer 参与对齐，传入的 optimizer 可以是一个 optimizer 实例，也可以是一个 lambda 函数。

optimizer 参数的具体的使用方法详见 [Tutorial](Tutorial.md)，以下提供简要介绍。

### 工具逻辑说明：关于 optimizer 的精度检查

在检查到输入参数包含了 optimizer 后，auto_diff 接口将按照以下逻辑进行相关检查。

1.   运行模型前反向计算，对比计算过程
2.   在运行完毕模型的 backward 部分后，对比检查模型记录的梯度大小
3.   调用传入的 optimizer ，更新模型权重
4.   更新权重后，检查模型权重间的数值精度误差

因此，在调用 optimizer 后出现的模型权重差异可以确定为 optimizer 精度问题，在 log 信息中将给出相应的提示

### 关于参数设置

1.   auto_weights 

     由于在多 step 对齐检查中，需要更新权重，因此 auto_weights 必须设置为 False，否则在每一个 step 前都会触发权重的拷贝。

2.   optimizer

     进行多 step 的对齐检查时必须显式地提供 optimizer ，否则工具将不知道如何更新模型权重。关于 optimizer 的设置和使用。

3.   steps

     auto_diff 接口允许设置 steps 参数，在 steps > 1 的情况下，auto_diff 将使用相同的输入数据进行多 step 的对齐检查。使用这个参数可以方便地进行快速检查。

     若准备了数据集进行训练，则不能设置 steps 参数。

### 示例代码

1.   使用 dataloader 进行多 step 的对齐检查（每一个step使用不同的input data）

```py
from padiff import auto_diff, assign_weight, LayerMap()

def test_forward():

  	# step 1: 得到 paddle 以及 torch 模型
    pretrained = './ViTPose/vitpose-b.pth'
    state_dict = torch.load(pretrained)['state_dict']
    torch_model = tiny_pose_torch()
    torch_model.load_state_dict(torch_new_state_dict)
    torch_model.eval()
    
    paddle_model = tiny_pose_paddle()
    paddle_model.eval()

    # step 2: 构造 dataloader
    print("torch dateloader build!")
    torch_dataset,torch_dataloader = build_torch_data_pipeline()

    print("paddle dateloader build!")
    paddle_dataset,paddle_dataloader = build_paddle_data_pipeline()
    
    # step 3: 构造 layer_map
    layer_map = LayerMap()
    layer_map.auto(paddle_model, torch_model)
    
    # step 4: 使用工具接口初始化 paddle 模型权重 （复制torch权重到paddle模型）
    assign_weight(paddle_model, torch_model, layer_map)

    # step 5: 取得对应的输入数据
    for idx, (paddle_batch, torch_batch
                ) in enumerate(zip(paddle_dataloader, torch_dataloader)):

        inp = ({'x': paddle_batch['image']},  
               {'x': torch_batch['img']})

        # step 6: 调用 auto_diff 接口，提供对应的 optimizer
        result = auto_diff(
          	paddle_model,
          	torch_model,
          	inp,
          	auto_weights=False,
          	layer_map=layer_map,
          	options={
            	  'atol':0.0, 
              	'rtol':1e-5,
              	'single_step':False,
              	'diff_phase':'both',
            }
          	optimizer=[paddle_opt, torch_opt]
        )
        
        if result == False:
          	break
```

2.   使用同一组输入进行多 step 对齐检查

```py
from padiff import auto_diff, assign_weight, LayerMap()

def test_forward():

  	# step 1: 得到 paddle 以及 torch 模型
    pretrained = './ViTPose/vitpose-b.pth'
    state_dict = torch.load(pretrained)['state_dict']
    torch_model = tiny_pose_torch()
    torch_model.load_state_dict(torch_new_state_dict)
    torch_model.eval()
    
    paddle_model = tiny_pose_paddle()
    paddle_model.eval()

    # step 2: 构造 input
    inp = ({'x': paddle_batch['image']},  
           {'x': torch_batch['img']})

    # step 3: 构造 layer_map
    layer_map = LayerMap()
    layer_map.auto(paddle_model, torch_model)
    
    # step 4: 使用工具接口初始化 paddle 模型权重 （复制torch权重到paddle模型）
    assign_weight(paddle_model, torch_model, layer_map)

    # step 5: 调用 auto_diff 接口 (使用 steps 参数)
    result = auto_diff(
        paddle_model,
        torch_model,
        inp,
        auto_weights=False,
        layer_map=layer_map,
        options={
            'atol':0.0, 
            'rtol':1e-5,
            'single_step':False,
            'diff_phase':'both',
        }
        optimizer=[paddle_opt, torch_opt],
      	steps=10,
    )
```





## 4. 出现 diff 时进行精确定位

在对齐检查的过程中可能出现这种情况：auto_diff 接口发现了精度 diff，但 log 信息中定位到的位置却是 Linear 等常见的 API，检查后未发现 Linear 存在 diff。

这可能是由于精度误差累积引起的，可以通过使用 single_step 对齐模式进行精度对齐定位，具体方法是：

1.   在接口参数中打开 single_step 模式的开关
2.   进行对齐检查的同时不断调整 atol， rtol 等参数，以找出 diff 出现的具体位置（开启 single_step 模式后，diff 数值的数量级可能下降）

### 工具逻辑说明：关于 single_step 模式

《图片》

### 关于参数设置

使用 single_step 模式，只需要将 options 参数中的 "single_step" 选项设置为 True，此时，"diff_phase" 选项将控制single_step 的行为，它能够被指定为 "forward",  "backward", "both" 3种可能

注意：当需要进行 "backward" 的 single_step 对齐时，auto_diff 会额外运行一次前向网络。

### 示例代码

```py
from padiff import auto_diff, assign_weight, LayerMap()

def test_forward():		
    # ... 

    result = auto_diff(
        paddle_model,
        torch_model,
        inp,
        auto_weights=False,
        layer_map=layer_map,
        options={
            'atol':0.0, 
            'rtol':1e-5,
            'single_step':False,
            'diff_phase':'both',
        }
    )
```




