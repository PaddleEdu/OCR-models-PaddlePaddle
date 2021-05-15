## CRAFT
![](https://ai-studio-static-online.cdn.bcebos.com/4f4800c1d9fa4de4be137ef3cd56577215ae335c1e2d4d9d997bb13f1d0e6b8b)

本项目基于paddlepaddle框架复现CRAFT，并参加百度第三届论文复现赛，将在2021年5月15日比赛完后提供AIStudio链接～敬请期待

参考项目：

[CRAFT: Character-Region Awareness For Text detection](https://github.com/clovaai/CRAFT-pytorch)

## 项目配置
```shell script
pip install -r requirements.txt
```
你应该具有以下目录
```
/home/aistudio/CRAFT(工程目录)
/home/aistudio/Data(数据集文件)
```
数据集文件已挂载，自行解压即可

## 训练

**The code for training is not included in this repository, and we cannot release the full training code for IP reason.**


*作者并未提供训练代码*

## 权重转换
这里用到了`X2Paddle`神器，转换代码如下，具体使用文档参见[X2Paddle](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/user_guides/pytorch2paddle.md)

```python
from craft import CRAFT
import torch
from collections import OrderedDict
import imgproc
import numpy as np
import cv2

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

# 构建输入
input_data = np.random.rand(1, 3, 736, 1280).astype("float32")
net = CRAFT()
net.load_state_dict(copyStateDict(torch.load('craft_mlt_25k.pth')))
net = net.cuda()
net.eval()

# 进行转换
from x2paddle.convert import pytorch2paddle
pytorch2paddle(net, 
          save_dir="paddlemodel", 
          jit_type="trace", 
          input_examples=[torch.tensor(input_data).cuda()])
```
完成后你会出现如下文件目录
```
/home/aistudio/CRAFT/paddlemodel
└───inference_model
└──────model.pdiparams
└──────model.pdiparams.info
└──────model.pdmodel
└───model.pdparams
└───x2paddle_code.py
```
使用同样的方式转换`refinenet`

## 测试

[模型下载](https://pan.baidu.com/s/1gqVaA2eVgwMFjNPCJI79NQ)

提取码：4yy1

[AIStudio链接](https://aistudio.baidu.com/aistudio/projectdetail/1927739?channel=0&channelType=0&shared=1)

```shell script
cd /home/aistudio/CRAFT
python test.py
```

<img src="https://ai-studio-static-online.cdn.bcebos.com/7ffe3b44046a4840a1df2c7c99c6f05de187afe56b634dbf86aeb88920bb69b5" width="800"/>

 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
 | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | craft_mlt_25k
IC15 | SynthText, IC15 | Eng | For IC15 only | craft_ic15_20k
LinkRefiner | CTW1500 | - | Used with the General Model | craft_refiner_CTW1500

**下图是实际测试效果**

<img src="https://ai-studio-static-online.cdn.bcebos.com/b2755b3bdd784c0f890ba168bb860182d2a776a8c239431ba487da02dad7b165" width="800"/>
<img src="https://ai-studio-static-online.cdn.bcebos.com/a669d797d2bd46ba867856be4da1c2485aea9e935ed2443c8711c528630a3f0e" width="800"/>


## 评估

**可以采用以下代码进行评估**
```script shell
cd /home/aistudio/CRAFT
python eval.py
cd /home/aistudio/CRAFT/outputs/submit_ic15/
zip ../submit_ic15.zip *
cd /home/aistudio/CRAFT/eval
`./eval_ic15.sh` or `bash eval_ic15.sh`
```

| Method | Dataset | Backbone | refiner | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| basenet | ICDAR2015 | VGG16_BN | N |82.2 | 77.9 | 80.0| craft_ic15_20k|
| basenet | ICDAR2015 | VGG16_BN | N |85.1 | 79.4 | 82.2| craft_mlt_25k|
| basenet | ICDAR2015 | VGG16_BN | Y |61.9 | 45.1 | 52.2| craft_ic15_20k|
| basenet | ICDAR2015 | VGG16_BN | Y |63.1 | 43.3 | 51.4| craft_mlt_25k|

**评估total_text数据集可参见我的PSNET项目eval文件价下的评估代码**


## **关于作者**

<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| 主页        | [Deep Hao的主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
