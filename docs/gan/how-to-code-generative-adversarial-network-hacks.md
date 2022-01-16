# 如何在 Keras 中实现 GAN Hacks 来训练稳定模型

> 原文：<https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/>

最后更新于 2019 年 7 月 12 日

[生成对抗网络](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)，或称 GANs，训练起来很有挑战性。

这是因为该架构同时涉及一个生成器和一个鉴别器模型，它们在零和游戏中竞争。这意味着一个模型的改进是以另一个模型的表现下降为代价的。结果是一个非常不稳定的训练过程，经常会导致失败，例如，一个生成器总是生成相同的图像或生成无意义的图像。

因此，有许多试探法或最佳实践(称为“[](https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/)*”)可以在配置和训练您的 GAN 模型时使用。多年来，实践者在一系列问题上测试和评估数百或数千个配置操作组合，来之不易。*

 *其中一些试探法可能很难实现，尤其是对初学者来说。

此外，它们中的一些或全部对于给定的项目可能是必需的，尽管可能不清楚应该采用哪一子集的试探法，这需要实验。这意味着一个实践者必须准备好实现一个给定的启发，而不需要太多的注意。

在本教程中，您将发现如何实现一套最佳实践或 GAN 黑客，您可以直接复制并粘贴到您的 GAN 项目。

阅读本教程后，您将知道:

*   开发生成对抗网络时实用启发式或黑客攻击的最佳来源。
*   如何从零开始实现深度卷积 GAN 模型架构的七个最佳实践？
*   如何实现 Soumith Chintala 的 GAN Hacks 演示文稿和列表中的四个附加最佳实践。

**用我的新书[Python 生成对抗网络](https://machinelearningmastery.com/generative_adversarial_networks/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Implement Hacks to Train Stable Generative Adversarial Networks](img/3f25bd34a780b484ed3b7021f4fe0954.png)

如何实现黑客训练稳定的生成对抗网络
图片由 [BLM 内华达](https://www.flickr.com/photos/blmnevada/29076765722/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  训练稳定遗传神经网络的启发式方法
2.  深度卷积遗传算法的最佳实践
    1.  使用条纹卷积进行下采样
    2.  使用条纹卷积进行上采样
    3.  用漏嘴
    4.  使用批处理规范化
    5.  使用高斯权重初始化
    6.  使用亚当随机梯度下降
    7.  将图像缩放到范围[-1，1]
3.  我是史密斯钦塔拉的 GAN Hacks
    1.  使用高斯潜在空间
    2.  真假图像分离手表
    3.  使用标签平滑
    4.  使用有噪声的标签

## 训练稳定遗传神经网络的启发式方法

GANs 很难训练。

在撰写本文时，关于如何设计和训练 GAN 模型，还没有很好的理论基础，但是已经有了启发式的既定文献，或者“T0”黑客，它们已经被经验证明在实践中运行良好。

因此，在开发 GAN 模型时，有一系列最佳实践需要考虑和实现。

建议的配置和训练参数的两个最重要的来源可能是:

1.  亚历克·拉德福德等人 2015 年的论文，介绍了 DCGAN 架构。
2.  soumith Chintala 2016 年的演讲和相关的“ *GAN Hacks* ”列表。

在本教程中，我们将探索如何从这两个来源实现最重要的最佳实践。

## 深度卷积遗传算法的最佳实践

在设计和训练稳定的 GAN 模型方面，最重要的一步可能是亚历克·拉德福德等人在 2015 年发表的论文，题为“利用深度卷积生成对抗网络的[无监督表示学习](https://arxiv.org/abs/1511.06434)”

在论文中，他们描述了深度卷积 GAN，或 DCGAN，这种 GAN 开发方法已经成为事实上的标准。

在本节中，我们将研究如何为 DCGAN 模型架构实现七种最佳实践。

### 1.使用条纹卷积进行下采样

鉴别器模型是一个标准的卷积神经网络模型，它将图像作为输入，并且必须输出一个关于它是真的还是假的二进制分类。

深度卷积网络的标准做法是使用池化层对具有网络深度的输入和要素图进行下采样。

DCGAN 不建议这样做，相反，他们建议使用[条纹卷积](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)进行下采样。

这包括按照常规定义卷积层，但不是使用默认的二维步长(1，1)将其更改为(2，2)。这具有对输入进行下采样的效果，特别是将输入的宽度和高度减半，从而产生面积为四分之一的输出要素图。

下面的示例用单个隐藏卷积层来演示这一点，该隐藏卷积层通过将“*步长*参数设置为(2，2)来使用下采样的步进卷积。效果是模型将输入从 64×64 下采样到 32×32。

```py
# example of downsampling with strided convolutions
from keras.models import Sequential
from keras.layers import Conv2D
# define model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
# summarize model
model.summary()
```

运行该示例显示了卷积层输出的形状，其中要素图有四分之一的面积。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 64)        1792
=================================================================
Total params: 1,792
Trainable params: 1,792
Non-trainable params: 0
_________________________________________________________________
```

### 2.使用条纹卷积进行上采样

生成器模型必须从潜在空间的随机点生成作为输入的输出图像。

实现这一点的推荐方法是使用带有交错卷积的转置卷积层。这是一种特殊类型的层，可以反向执行卷积运算。直观地说，这意味着设置 2×2 的步幅将产生相反的效果，对输入进行上采样，而不是在正常卷积层的情况下对其进行下采样。

通过堆叠具有交错卷积的转置卷积层，生成器模型能够将给定的输入缩放到期望的输出维度。

下面的例子用一个隐藏的转置卷积层演示了这一点，该层通过将“*步长*参数设置为(2，2)来使用上采样的步进卷积。

效果是模型将输入从 64×64 增加到 128×128。

```py
# example of upsampling with strided convolutions
from keras.models import Sequential
from keras.layers import Conv2DTranspose
# define model
model = Sequential()
model.add(Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2), padding='same', input_shape=(64,64,3)))
# summarize model
model.summary()
```

运行该示例显示了卷积层输出的形状，其中特征图的面积是四倍。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_transpose_1 (Conv2DTr (None, 128, 128, 64)      3136
=================================================================
Total params: 3,136
Trainable params: 3,136
Non-trainable params: 0
_________________________________________________________________
```

### 3.用漏嘴

[整流线性激活单元](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)，简称 ReLU，是一个简单的计算，直接返回作为输入提供的值，如果输入小于等于 0.0，则返回 0.0。

一般开发深度卷积神经网络已经成为一种最佳实践。

GANs 的最佳实践是使用 ReLU 的变体，它允许一些值小于零，并学习每个节点的截止点。这被称为泄漏整流线性激活单元，简称 LeakyReLU。

可以为泄漏率指定负斜率，建议默认值为 0.2。

最初，ReLU 被推荐用于生成器模型，LeakyReLU 被推荐用于鉴别器模型，尽管最近，LeakyReLU 被推荐用于这两个模型。

下面的示例演示了在鉴别器模型的卷积层之后使用默认斜率为 0.2 的 LeakyReLU。

```py
# example of using leakyrelu in a discriminator model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
# define model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
model.add(LeakyReLU(0.2))
# summarize model
model.summary()
```

运行该示例演示了模型的结构，该模型有一个卷积层，后面是激活层。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 64)        1792
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 32, 32, 64)        0
=================================================================
Total params: 1,792
Trainable params: 1,792
Non-trainable params: 0
_________________________________________________________________
```

### 4.使用批处理规范化

[批量标准化](https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/)标准化前一层的激活，使其均值和单位方差为零。这具有稳定训练过程的效果。

在鉴别器和生成器模型中分别激活卷积层和转置卷积层之后，使用批量归一化。

它是在隐藏层之后，但在激活之前添加到模型中的，例如 LeakyReLU。

下面的示例演示了在鉴别器模型中的 Conv2D 层之后但在激活之前添加批处理规范化层。

```py
# example of using batch norm in a discriminator model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
# define model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(0.2))
# summarize model
model.summary()
```

运行该示例显示了卷积层的输出和激活函数之间批处理范数的期望使用。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 64)        1792
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 64)        256
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 32, 32, 64)        0
=================================================================
Total params: 2,048
Trainable params: 1,920
Non-trainable params: 128
_________________________________________________________________
```

### 5.使用高斯权重初始化

在训练神经网络之前，必须将模型权重(参数)初始化为小的随机变量。

论文中报告的 DCAGAN 模型的最佳实践是使用标准偏差为 0.02 的零中心高斯分布(正态或钟形分布)初始化所有权重。

下面的例子演示了定义一个均值为 0、标准差为 0.02 的随机高斯权重初始值，用于生成器模型中的转置卷积层。

给定模型中的每一层都可以使用相同的权重初始化器实例。

```py
# example of gaussian weight initialization in a generator model
from keras.models import Sequential
from keras.layers import Conv2DTranspose
from keras.initializers import RandomNormal
# define model
model = Sequential()
init = RandomNormal(mean=0.0, stddev=0.02)
model.add(Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=(64,64,3)))
```

### 6.使用亚当随机梯度下降

随机梯度下降，简称 SGD，是用于优化卷积神经网络模型权重的标准算法。

训练算法有许多变体。训练 DCGAN 模型的最佳做法是使用 [Adam 版本的随机梯度下降](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)，学习率为 0.0002，beta1 动量值为 0.5，而不是默认值 0.9。

在优化鉴别器和生成器模型时，建议使用具有这种配置的 Adam 优化算法。

下面的示例演示了如何配置 Adam 随机梯度下降优化算法来训练鉴别器模型。

```py
# example of using adam when training a discriminator model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
# define model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', input_shape=(64,64,3)))
# compile model
opt = Adam(lr=0.0002, beta_1=0.5)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
```

### 7.将图像缩放到范围[-1，1]

建议使用双曲正切激活函数作为生成器模型的输出。

因此，还建议对用于训练鉴别器的真实图像进行[缩放，使得它们的像素值](https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/)在[-1，1]的范围内。这使得鉴别器将总是接收像素值在相同范围内的输入图像，真实的和假的。

典型地，图像数据被加载为 NumPy 数组，使得像素值是在[0，255]范围内的 8 位无符号整数(uint8)值。

首先，必须将数组转换为浮点值，然后重新缩放到所需的范围。

下面的示例提供了一个函数，该函数可以将加载图像数据的 NumPy 数组适当地缩放到[-1，1]的所需范围。

```py
# example of a function for scaling images

# scale image data from [0,255] to [-1,1]
def scale_images(images):
	# convert from unit8 to float32
	images = images.astype('float32')
	# scale from [0,255] to [-1,1]
	images = (images - 127.5) / 127.5
	return images
```

## 我是史密斯钦塔拉的 GAN Hacks

DCGAN 论文的合著者之一 Soumith Chintala 在 NIPS 2016 上做了题为“如何训练 GAN？“总结了很多小技巧和窍门。

该视频可在 YouTube 上获得，强烈推荐。这些技巧的总结也可以作为 GitHub 资源库获得，标题为“如何训练 GAN？让 GANs 发挥作用的提示和技巧。”

这些提示借鉴了 DCGAN 论文以及其他地方的建议。

在本节中，我们将回顾如何实现前一节中未涉及的另外四种 GAN 最佳实践。

### 1.使用高斯潜在空间

潜在空间定义了用于生成新图像的生成器模型的输入的形状和分布。

DCGAN 建议从均匀分布中取样，这意味着潜在空间的形状是超立方体。

最近的最佳实践是从标准[高斯分布](https://machinelearningmastery.com/statistical-data-distributions/)中采样，这意味着潜在空间的形状是一个超球面，平均值为零，标准偏差为 1。

下面的例子演示了如何从一个 100 维的潜在空间中生成 500 个[随机高斯点](https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/)，该潜在空间可以用作生成器模型的输入；每个点都可以用来生成图像。

```py
# example of sampling from a gaussian latent space
from numpy.random import randn

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape((n_samples, latent_dim))
	return x_input

# size of latent space
n_dim = 100
# number of samples to generate
n_samples = 500
# generate samples
samples = generate_latent_points(n_dim, n_samples)
# summarize
print(samples.shape, samples.mean(), samples.std())
```

运行该示例总结了 500 个点的生成，每个点由 100 个随机高斯值组成，平均值接近于零，标准偏差接近于 1，例如标准高斯分布。

```py
(500, 100) -0.004791256735601787 0.9976912528950904
```

### 2.分开批次的真实和虚假图像

鉴别器模型使用随机梯度下降和小批量训练。

最佳做法是用不同批次的真实图像和伪造图像来更新鉴别器，而不是将真实图像和伪造图像合并成一个批次。

这可以通过两次单独调用 train_on_batch()函数来更新鉴别器模型的模型权重来实现。

下面的代码片段演示了在训练鉴别器模型时，如何在代码的内部循环中做到这一点。

```py
...
# get randomly selected 'real' samples
X_real, y_real = ...
# update discriminator model weights
discriminator.train_on_batch(X_real, y_real)
# generate 'fake' examples
X_fake, y_fake = ...
# update discriminator model weights
discriminator.train_on_batch(X_fake, y_fake)
```

### 3.使用标签平滑

在训练鉴别器模型时，通常使用类别标签 1 来表示真实图像，使用类别标签 0 来表示假图像。

这些被称为硬标签，因为标签值是精确的或清晰的。

使用软标签是一个很好的做法，例如真实图像和假图像的值分别略大于或小于 1.0 或略大于 0.0，其中每个图像的变化是随机的。

这通常被称为标签平滑，并且在训练模型时可以产生[正则化效果](https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/)。

下面的示例演示了为正类(类=1)定义 1，000 个标签，并按照建议将标签值均匀地平滑到范围[0.7，1.2]内。

```py
# example of positive label smoothing
from numpy import ones
from numpy.random import random

# example of smoothing class=1 to [0.7, 1.2]
def smooth_positive_labels(y):
	return y - 0.3 + (random(y.shape) * 0.5)

# generate 'real' class labels (1)
n_samples = 1000
y = ones((n_samples, 1))
# smooth labels
y = smooth_positive_labels(y)
# summarize smooth labels
print(y.shape, y.min(), y.max())
```

运行该示例总结了平滑值的最小值和最大值，显示它们接近预期值。

```py
(1000, 1) 0.7003103006957805 1.1997858934066357
```

有人建议只需要正类标签平滑，并且值小于 1.0。尽管如此，您也可以平滑负类标签。

下面的示例演示了为负类(类=0)生成 1，000 个标签，并按照建议将标签值均匀地平滑到范围[0.0，0.3]内。

```py
# example of negative label smoothing
from numpy import zeros
from numpy.random import random

# example of smoothing class=0 to [0.0, 0.3]
def smooth_negative_labels(y):
	return y + random(y.shape) * 0.3

# generate 'fake' class labels (0)
n_samples = 1000
y = zeros((n_samples, 1))
# smooth labels
y = smooth_negative_labels(y)
# summarize smooth labels
print(y.shape, y.min(), y.max())
```

### 4.使用有噪声的标签

训练鉴别器模型时使用的标签总是正确的。

这意味着假图像总是被标记为 0 类，而真实图像总是被标记为 1 类。

建议在这些标签中引入一些错误，一些假图像被标记为真实，一些真实图像被标记为虚假。

如果您使用单独的批次来更新真假图像的鉴别器，这可能意味着向该批真实图像中随机添加一些假图像，或者向该批假图像中随机添加一些真实图像。

如果您正在用一批真实和虚假的图像更新鉴别器，那么这可能会涉及到随机翻转一些图像上的标签。

下面的示例通过创建 1000 个真实(类=1)标签样本并以 5%的概率翻转它们来演示这一点，然后对 1000 个虚假(类=0)标签样本进行同样的操作。

```py
# example of noisy labels
from numpy import ones
from numpy import zeros
from numpy.random import choice

# randomly flip some labels
def noisy_labels(y, p_flip):
	# determine the number of labels to flip
	n_select = int(p_flip * y.shape[0])
	# choose labels to flip
	flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
	# invert the labels in place
	y[flip_ix] = 1 - y[flip_ix]
	return y

# generate 'real' class labels (1)
n_samples = 1000
y = ones((n_samples, 1))
# flip labels with 5% probability
y = noisy_labels(y, 0.05)
# summarize labels
print(y.sum())

# generate 'fake' class labels (0)
y = zeros((n_samples, 1))
# flip labels with 5% probability
y = noisy_labels(y, 0.05)
# summarize labels
print(y.sum())
```

试着运行这个例子几次。

结果显示，对于正标签，大约 50“1”翻转为 1(例如，1，000 的 5%)，对于负标签，大约 50“0”翻转为 1。

```py
950.049.0
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [深度卷积生成对抗网络的无监督表示学习](https://arxiv.org/abs/1511.06434)，2015。
*   [教程:生成对抗网络，NIPS](https://arxiv.org/abs/1701.00160) ，2016。
*   [训练 GANs 的改进技术](https://arxiv.org/abs/1606.03498)，2016。

### 应用程序接口

*   [响亮的 API](https://keras.io) 。
*   [NumPy 随机采样(numpy.random) API](https://docs.scipy.org/doc/numpy/reference/routines.random.html)
*   [NumPy 数组操作例程](https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)

### 文章

*   [ganhacks:如何训练一个 GAN？让 GANs 发挥作用的提示和技巧](https://github.com/soumith/ganhacks)
*   [Ian Goodfellow，GANs 入门，NIPS 2016](https://www.youtube.com/watch?v=9JpdAg6uMXs) 。
*   [苏史密斯·钦塔拉，如何训练一个 GAN，NIPS 2016 对抗训练研讨会](https://www.youtube.com/watch?v=X1mUN6dD8uE)。

## 摘要

在本教程中，您发现了如何实现一套最佳实践或 GAN 黑客，您可以直接复制并粘贴到您的 GAN 项目。

具体来说，您了解到:

*   开发生成对抗网络时实用启发式或黑客攻击的最佳来源。
*   如何从零开始实现深度卷积 GAN 模型架构的七个最佳实践？
*   如何实现 Soumith Chintala 的 GAN Hacks 演示文稿和列表中的四个附加最佳实践。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。*