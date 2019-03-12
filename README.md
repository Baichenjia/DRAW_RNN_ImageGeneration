# DRAW_RNN_ImageGeneration
TensorFlow Eager Implementation of "DRAW: A Recurrent Neural Network For Image Generation" 2015

## 概述
1. 本文依据文章 "DRAW: A Recurrent Neural Network For Image Generation"， 参考 "https://github.com/ericjang/draw" 实现了一个简单版
的 DRAW 模型。
2. 为了简单，该模型没有使用空间的 Attention 机制。使用 T=10 个时间步来生成一幅图像。
3. 使用 Tensorflow 2.0 类似的 eager 方式实现。

## 训练
1. 训练的过程写在文件 `DRAW.py` 中，训练过程中将权重存储在 `weights` 文件夹下。
2. 训练周期一般需要 20 个周期左右。
3. 训练数据是Mnist，二值化后作为输入。

## 生成图像
1. 已经训练好的权重保存在 `weights/model_weight.h5` 中，通过 test.py 中导入该权重可以用于生成图片。
2. 首先从正态分布中采样 8 个点，随后生成图片，如下图所示。
3. 其中，列表示 8 个不同的随机采样点的生成结果；行 表示每个时间步的生成结果。可以看到，在顺序生成的过程中，图片越来越清晰。

## 示例
![](/generated_img/img1.jpg)

![](/generated_img/img2.jpg)

![](/generated_img/img5.jpg)







