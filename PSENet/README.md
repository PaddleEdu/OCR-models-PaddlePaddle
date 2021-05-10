# 介绍
本项目基于paddlepaddle框架复现PSENet，并参加百度第三届论文复现赛，将在2021年5月15日比赛完进行开源～敬请期待。

参考论文：
- [1] W. Wang, E. Xie, X. Li, W. Hou, T. Lu, G. Yu, and S. Shao. Shape robust text detection with progressive scale expansion network. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 9336–9345, 2019.<br>

参考项目：
- [https://github.com/whai362/PSENet](https://github.com/whai362/PSENet)

# 对比
本项目基于paddlepaddle深度学习框架复现，对比于作者论文的代码：
- 我们将提供更加细节的训练流程，以及更加全面的预训练模型。
- 我们将提供基于aistudio平台可在线运行的项目地址，您不需要在您的机器上配置paddle环境可以直接浏览器在线运行全套流程。
- 我们的提供模型在total_text数据集上超越原作者论文最好模型3%左右，在ICDAR2015数据集上超过作者开源的模型3%左右（差作者未公开的论文模型0.6%）。
- 模型速度上，在total_text数据集上FPS达到10.1是作者论文的2.5倍左右，在ICDAR2015上FPS为1.8与作者论文的FPS1.6相当。


# 细节
>该列指标在ICDAR2015的测试集测试

train from scratch细节：


| |epoch|opt|short_size|batch_size|dataset|memory|card|precision|recall|hmean|FPS|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|pretrain_1|33|Adam|1024|16|ICDAR2017|32G|1|0.68290|0.68850|0.68569|5.0|
|pretrain_2|46|Adam|1024|16|ICDAR2013、ICDAR2017、COCO_TEXT|32G|4|0.69678|0.69812|0.69745|5.0|
|pretrain_3|68|Adam|1260|16|ICDAR2013、ICDAR2015、ICDAR2017、COCO_TEXT|32G|1|0.86526|0.80693|0.83508|2.0|

## ICDAR2015
>该列指标在ICDAR2015的测试集测试

训练细节：

| |pretrain|epoch|opt|short_size|batch_size|dataset|memory|card|precision|recall|hmean|FPS|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|finetune_1|pretrain_1|491|Adam|1024|16|ICDAR2015|32G|1|0.86463|0.80260|0.83246|5.0|
|finetune_2|pretrain_3|-|Adam|1260|16|ICDAR2015|32G|1|0.87024|0.81367|0.84101|2.0|
|finetune_3|finetune_2|401|SGD|1480|16|ICDAR2015|32G|1|<font color='red'>0.88060</font>|<font color='red'>0.82378</font>|<font color='red'>0.85124</font>|<font color='red'>1.8</font>|

## Total_text
>该列指标在Total_Text的测试集测试

训练细节：

| |pretrain|epoch|opt|short_size|batch_size|dataset|memory|card|precision|recall|hmean|FPS|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|finetune_1|None|331|Adam|736|16|Total_Text|32G|1|0.84823|0.76007|0.80173|10.1|
|finetune_2|pretrain_2|290|Adam|736|16|Total_Text|32G|1|<font color='red'>0.88482</font>|<font color='red'>0.79002</font>|<font color='red'>0.83474</font>|<font color='red'>10.1</font>|
