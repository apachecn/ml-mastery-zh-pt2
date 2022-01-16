# 如何在 Keras 中使用 UpSampling2D 和 Conv2D 转置层

> 原文：<https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/>

最后更新于 2019 年 7 月 12 日

生成对抗网络是一种用于训练生成模型的体系结构，例如用于生成图像的深度卷积神经网络。

GAN 架构由一个生成器和一个鉴别器模型组成。生成器负责创建新的输出，如图像，这些输出很可能来自原始数据集。生成器模型通常使用深度卷积神经网络和结果专用层来实现，结果专用层学习填充图像中的特征，而不是从输入图像中提取特征。

可以在生成器模型中使用的两种常见类型的层是简单地将输入的维度加倍的上采样层(upsample layer，UpSampling2D)和执行反卷积运算的转置卷积层(Conv2DTranspose)。

在本教程中，您将发现如何在生成图像时使用创成式对抗网络中的增强 2D 和转换 2D 置换层。

完成本教程后，您将知道:

*   GAN 架构中的生成模型需要对输入数据进行上采样，以生成输出图像。
*   上采样层是一个没有权重的简单层，它将使输入的维度加倍，并且可以在传统卷积层之后的生成模型中使用。
*   转置卷积层是一个反向卷积层，它将对输入进行上采样，并学习如何在模型训练过程中填充细节。

**用我的新书[Python 生成对抗网络](https://machinelearningmastery.com/generative_adversarial_networks/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![A Gentle Introduction to Upsampling and Transpose Convolution Layers for Generative Adversarial Networks](img/e76581acd96e98162b209ad97d06218e.png)

生成对抗网络的上采样和转置卷积层简介
图片由 [BLM 内华达](https://www.flickr.com/photos/blmnevada/29182914095/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

*   需要在 GANs 中进行上采样
*   如何使用上采样层
*   如何使用转置卷积层

## 生成对抗网络中的上采样需求

生成对抗网络是一种用于训练生成模型的神经网络体系结构。

该体系结构由一个生成器和一个鉴别器模型组成，这两个模型都实现为一个深度卷积神经网络。鉴别器负责将图像分类为真实的(来自域)或虚假的(生成的)。生成器负责从问题域生成新的似是而非的例子。

生成器通过从潜在空间中选取一个随机点作为输入，并以一次拍摄的方式输出完整的图像来工作。

用于图像分类和相关任务的传统卷积神经网络将使用[池化层](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)对输入图像进行下采样。例如，平均池层或最大池层将在每个维度上将卷积图的特征图减少一半，从而产生四分之一输入面积的输出。

卷积层本身也通过对输入图像或特征图应用每个滤波器来执行一种形式的下采样；生成的激活是一个输出特征图，由于边界效应，它变小了。通常[填充](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)用来抵消这种效果。

GAN 中的生成器模型需要传统卷积层中池层的反向操作。它需要一个层来从粗糙的显著特征转换成更密集和详细的输出。

一个简单版本的非冷却或相反的池化层被称为上采样层。它通过重复输入的行和列来工作。

一种更复杂的方法是执行反向卷积运算，最初称为反卷积，这是不正确的，但更常见的是称为分数卷积层或转置卷积层。

这两个层都可以在 GAN 上使用，以执行所需的上采样操作，将小输入转换为大图像输出。

在接下来的部分中，我们将仔细研究每一个，并对它们如何工作形成直觉，以便我们可以在 GAN 模型中有效地使用它们。

## 如何使用放大的二维层

也许对输入进行上采样的最简单方法是每一行和每一列加倍。

例如，形状为 2×2 的输入图像将输出为 4×4。

```py
         1, 2
Input = (3, 4)

          1, 1, 2, 2
Output = (1, 1, 2, 2)
          3, 3, 4, 4
          3, 3, 4, 4
```

### 使用上放大二维层的工作示例

Keras 深度学习库在一个名为*upsmampling 2d*的层中提供了这种能力。

它可以被添加到卷积神经网络，并在输出中重复作为输入提供的行和列。例如:

```py
...
# define model
model = Sequential()
model.add(UpSampling2D())
```

我们可以用一个简单的人为例子来演示这个层的行为。

首先，我们可以定义一个 2×2 像素的人为输入图像。我们可以为每个像素使用特定的值，这样在上采样后，我们就可以确切地看到操作对输入的影响。

```py
...
# define input data
X = asarray([[1, 2],
			 [3, 4]])
# show input data for context
print(X)
```

一旦定义了图像，我们必须添加一个通道维度(例如灰度)和一个样本维度(例如我们有 1 个样本)，这样我们就可以将其作为输入传递给模型。

```py
...
# reshape input data into one sample a sample with a channel
X = X.reshape((1, 2, 2, 1))
```

我们现在可以定义我们的模型了。

该模型只有*上采样 2D* 层，直接以 2×2 灰度图像作为输入，输出上采样操作的结果。

```py
...
# define model
model = Sequential()
model.add(UpSampling2D(input_shape=(2, 2, 1)))
# summarize the model
model.summary()
```

然后，我们可以使用该模型进行预测，即对提供的输入图像进行上采样。

```py
...
# make a prediction with the model
yhat = model.predict(X)
```

输出将像输入一样有四个维度，因此，我们可以将其转换回 2×2 数组，以便于查看结果。

```py
...
# reshape output to remove channel to make printing easier
yhat = yhat.reshape((4, 4))
# summarize output
print(yhat)
```

将所有这些结合在一起，下面提供了在 Keras 中使用*upsmampling 2d*层的完整示例。

```py
# example of using the upsampling layer
from numpy import asarray
from keras.models import Sequential
from keras.layers import UpSampling2D
# define input data
X = asarray([[1, 2],
			 [3, 4]])
# show input data for context
print(X)
# reshape input data into one sample a sample with a channel
X = X.reshape((1, 2, 2, 1))
# define model
model = Sequential()
model.add(UpSampling2D(input_shape=(2, 2, 1)))
# summarize the model
model.summary()
# make a prediction with the model
yhat = model.predict(X)
# reshape output to remove channel to make printing easier
yhat = yhat.reshape((4, 4))
# summarize output
print(yhat)
```

运行该示例首先创建并总结我们的 2×2 输入数据。

接下来，对模型进行总结。我们可以看到，它将输出我们所期望的 4×4 结果，重要的是，该层没有参数或模型权重。这是因为它没有学到任何东西；这只是投入的两倍。

最后，该模型用于对我们的输入进行上采样，如我们所料，导致我们的输入数据的每一行和每一列都翻倍。

```py
[[1 2]
 [3 4]]

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
up_sampling2d_1 (UpSampling2 (None, 4, 4, 1)           0
=================================================================
Total params: 0
Trainable params: 0
Non-trainable params: 0
_________________________________________________________________

[[1\. 1\. 2\. 2.]
 [1\. 1\. 2\. 2.]
 [3\. 3\. 4\. 4.]
 [3\. 3\. 4\. 4.]]
```

默认情况下，*向上放大 2D* 将使每个输入尺寸加倍。这是由设置为元组(2，2)的“*大小*参数定义的。

您可能希望在每个维度上使用不同的因子，例如宽度的两倍和高度的三倍。这可以通过将“*大小*参数设置为(2，3)来实现。将该操作应用于 2×2 图像的结果将是 4×6 输出图像(例如 2×2 和 2×3)。例如:

```py
...
# example of using different scale factors for each dimension
model.add(UpSampling2D(size=(2, 3)))
```

此外，默认情况下，*upsmampling 2d*层将使用最近邻算法来填充新的行和列。这具有简单地将行和列加倍的效果，如所述，并且由设置为最接近的“*插值*参数指定。

或者，可以使用利用多个周围点的双线性插值方法。这可以通过将“*插值*参数设置为“*双线性*来指定。例如:

```py
...
# example of using bilinear interpolation when upsampling
model.add(UpSampling2D(interpolation='bilinear'))
```

### 带有上放大 2D 层的简单生成器模型

*upsmamping 2d*层简单有效，虽然不执行任何学习。

它不能在向上采样操作中填充有用的细节。为了在 GAN 中有用，每个*upsmampling 2d*层后面必须跟有一个 [Conv2D 层](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)，该层将学习解释加倍的输入，并被训练将其转化为有意义的细节。

我们可以用一个例子来证明。

在这种情况下，我们的小 GAN 生成器模型必须生成一个 10×10 的图像，并从潜在空间中获取一个 100 元素的向量作为输入。

首先，可以使用一个密集的完全连接层来解释输入向量，并创建足够数量的激活(输出)，这些激活可以被重塑为我们输出图像的低分辨率版本，在本例中，是 5×5 图像的 128 个版本。

```py
...
# define model
model = Sequential()
# define input shape, output enough activations for for 128 5x5 image
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((5, 5, 128)))
```

接下来，可以将 5×5 的要素图上采样为 10×10 的要素图。

```py
...
# double input from 128 5x5 to 1 10x10 feature map
model.add(UpSampling2D())
```

最后，上采样的要素图可以由一个 Conv2D 层解释并填充有用的细节。

Conv2D 有一个单一的功能图作为输出，以创建我们需要的单一图像。

```py
...
# fill in detail in the upsampled feature maps
model.add(Conv2D(1, (3,3), padding='same'))
```

将这些联系在一起，完整的示例如下所示。

```py
# example of using upsampling in a simple generator model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.layers import Conv2D
# define model
model = Sequential()
# define input shape, output enough activations for for 128 5x5 image
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((5, 5, 128)))
# double input from 128 5x5 to 1 10x10 feature map
model.add(UpSampling2D())
# fill in detail in the upsampled feature maps and output a single image
model.add(Conv2D(1, (3,3), padding='same'))
# summarize model
model.summary()
```

运行该示例会创建模型并总结每个层的输出形状。

我们可以看到密集层输出 3200 次激活，然后这些激活被重新整形为 128 个形状为 5×5 的特征图。

通过*上放大 2D* 层，宽度和高度增加了一倍，达到 10×10，从而生成面积增加了四倍的要素地图。

最后，Conv2D 处理这些特征图并添加细节，输出单个 10×10 图像。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 3200)              323200
_________________________________________________________________
reshape_1 (Reshape)          (None, 5, 5, 128)         0
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 10, 10, 128)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 10, 10, 1)         1153
=================================================================
Total params: 324,353
Trainable params: 324,353
Non-trainable params: 0
_________________________________________________________________
```

## 如何使用转换 2 置换层

conv2d 转置或转置卷积层比简单的上采样层更复杂。

一个简单的方法是，它既执行上采样操作，又解释粗输入数据，以便在上采样时填充细节。它就像一个层，将*upsmampling 2d*和 Conv2D 层组合成一个层。这是一个粗略的理解，但却是一个实际的出发点。

> 对转置卷积的需求通常源于使用与正常卷积方向相反的变换的期望，即从具有某个卷积输出形状的事物到具有其输入形状的事物，同时保持与所述卷积兼容的连通性模式

——[深度学习卷积算法指南](https://arxiv.org/abs/1603.07285)，2016 年。

事实上，转置卷积层执行反卷积运算。

具体来说，卷积层的前向和后向通路是反向的。

> 一种说法是，内核定义了一个卷积，但是它是直接卷积还是转置卷积取决于向前和向后传递的计算方式。

——[深度学习卷积算法指南](https://arxiv.org/abs/1603.07285)，2016 年。

它有时被称为反卷积或反卷积层，使用这些层的模型可以被称为反卷积网络或反卷积集。

> 一个解卷积网络可以被认为是一个 convnet 模型，它使用相同的组件(过滤、池化)，但方向相反，所以与其将像素映射到特征，不如反过来做。

——[可视化和理解卷积网络](https://arxiv.org/abs/1311.2901)，2013。

将该操作称为反褶积在技术上是不正确的，因为反褶积是该层不执行的特定数学操作。

事实上，传统的[卷积层](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)在技术上并不执行卷积运算，而是执行互相关。

> 人们通常所指的反卷积层首次出现在泽勒的论文中，作为反卷积网络的一部分，但没有具体的名称。[……]它也有许多名称，包括(但不限于)子像素或分数卷积层、转置卷积层、逆卷积层、上卷积层或后卷积层。

——[反卷积层和卷积层一样吗？](https://arxiv.org/abs/1609.07009)，2016 年。

这是一个非常灵活的层，尽管我们将重点关注它在对输入图像进行上采样的生成模型中的使用。

转置卷积层很像普通卷积层。它要求您指定过滤器的数量和每个过滤器的内核大小。这一层的关键是步幅。

典型地，卷积层的[步距是(1×1)，也就是说，对于从左到右的每次读取，滤波器沿着一个像素水平移动，然后对于下一行读取向下移动一个像素。正常卷积层上 2×2 的步长具有对输入进行下采样的效果，很像](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)[池化层](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)。事实上，在鉴别器模型中，可以使用 2×2 步长来代替池化层。

转置卷积层类似于逆卷积层。因此，你会直觉地认为 2×2 的步幅会对输入进行上采样，而不是下采样，这正是发生的情况。

跨步是指传统卷积层中滤波器扫描输入的方式。而在转置卷积层中，步幅指的是特征图中输出的布局方式。

这种效果可以通过使用分数输入步幅(f)的普通卷积层来实现，例如步幅为 f=1/2。反转时，输出步幅设置为该分数的分子，例如 f=2。

> 从某种意义上说，用 f 因子进行上采样是用 1/f 的分数输入步长进行卷积。只要 f 是整数，自然的上采样方式就是用 f 的输出步长进行反向卷积(有时称为反卷积)

——[语义分割的全卷积网络](https://arxiv.org/abs/1411.4038)，2014。

使用普通卷积层可以实现这种效果的一种方式是在输入数据中插入 0.0 值的新行和新列。

> 最后要注意的是，总是可以用直接卷积来模拟转置卷积。缺点是它通常需要向输入中添加许多列和行的零…

——[深度学习卷积算法指南](https://arxiv.org/abs/1603.07285)，2016 年。

让我们用一个例子来具体说明。

考虑大小为 2×2 的输入图像，如下所示:

```py
         1, 2
Input = (3, 4)
```

假设单个滤波器具有 1×1 内核和模型权重，在输出时不会导致输入发生变化(例如，模型权重为 1.0，偏差为 0.0)，输出步长为 1×1 的转置卷积运算将按原样再现输出:

```py
          1, 2
Output = (3, 4)
```

输出步长为(2，2)时，1×1 卷积需要在输入图像中插入额外的行和列，以便可以执行读取操作。因此，输入如下所示:

```py
         1, 0, 2, 0
Input = (0, 0, 0, 0)
         3, 0, 4, 0
         0, 0, 0, 0
```

然后，模型可以使用(2，2)的输出步幅读取该输入，并将输出 4×4 的图像，在这种情况下没有变化，因为我们的模型权重不受设计影响:

```py
          1, 0, 2, 0
Output = (0, 0, 0, 0)
          3, 0, 4, 0
          0, 0, 0, 0
```

### 使用 con v2 置换层的工作示例

Keras 通过 con v2 转置层提供转置卷积功能。

可以直接添加到您的模型中；例如:

```py
...
# define model
model = Sequential()
model.add(Conv2DTranspose(...))
```

我们可以用一个简单的人为例子来演示这个层的行为。

首先，我们可以定义一个 2×2 像素的人为输入图像，就像我们在上一节中所做的那样。我们可以为每个像素使用特定的值，以便在转置卷积运算之后，我们可以确切地看到运算对输入有什么影响。

```py
...
# define input data
X = asarray([[1, 2],
			 [3, 4]])
# show input data for context
print(X)
```

一旦定义了图像，我们必须添加一个通道维度(例如灰度)和一个样本维度(例如我们有 1 个样本)，这样我们就可以将其作为输入传递给模型。

```py
...
# reshape input data into one sample a sample with a channel
X = X.reshape((1, 2, 2, 1))
```

我们现在可以定义我们的模型了。

模型只有*conv2d 转置*层，直接以 2×2 灰度图像作为输入，输出运算结果。

con v2 转置两个上采样并执行卷积。因此，我们必须指定过滤器的数量和过滤器的大小，就像我们为 Conv2D 层所做的那样。此外，我们必须指定(2，2)的步距，因为上采样是通过输入卷积的步距行为实现的。

指定步幅为(2，2)具有分隔输入的效果。具体来说，插入 0.0 值的行和列以实现期望的步幅。

在本例中，我们将使用一个滤波器，其内核为 1×1，步长为 2×2，因此 2×2 的输入图像被上采样到 4×4。

```py
...
# define model
model = Sequential()
model.add(Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=(2, 2, 1)))
# summarize the model
model.summary()
```

为了弄清楚*conv2d 转置*层在做什么，我们将把单个过滤器中的单个权重固定为 1.0 的值，并使用 0.0 的偏差值。

这些权重以及内核大小(1，1)意味着输入中的值将乘以 1 并按原样输出，通过 2×2 的步长添加的新行和新列中的 0 值将输出为 0(例如，在每种情况下为 1 * 0)。

```py
...
# define weights that they do nothing
weights = [asarray([[[[1]]]]), asarray([0])]
# store the weights in the model
model.set_weights(weights)
```

然后，我们可以使用该模型进行预测，即对提供的输入图像进行上采样。

```py
...
# make a prediction with the model
yhat = model.predict(X)
```

输出将像输入一样有四个维度，因此，我们可以将其转换回 2×2 数组，以便于查看结果。

```py
...
# reshape output to remove channel to make printing easier
yhat = yhat.reshape((4, 4))
# summarize output
print(yhat)
```

将所有这些结合在一起，下面提供了在 Keras 中使用*conv2d 转置*层的完整示例。

```py
# example of using the transpose convolutional layer
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2DTranspose
# define input data
X = asarray([[1, 2],
			 [3, 4]])
# show input data for context
print(X)
# reshape input data into one sample a sample with a channel
X = X.reshape((1, 2, 2, 1))
# define model
model = Sequential()
model.add(Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=(2, 2, 1)))
# summarize the model
model.summary()
# define weights that they do nothing
weights = [asarray([[[[1]]]]), asarray([0])]
# store the weights in the model
model.set_weights(weights)
# make a prediction with the model
yhat = model.predict(X)
# reshape output to remove channel to make printing easier
yhat = yhat.reshape((4, 4))
# summarize output
print(yhat)
```

运行该示例首先创建并总结我们的 2×2 输入数据。

接下来，对模型进行总结。我们可以看到，它将输出我们期望的 4×4 的结果，重要的是，第二层参数或模型权重。一个用于单个 1×1 滤波器，一个用于偏置。与*上采样 2D* 层不同的是，*conv2d 转置*将在训练期间学习，并将尝试填充细节作为上采样过程的一部分。

最后，该模型用于对我们的输入进行上采样。我们可以看到，包含实值作为输入的单元格的计算导致实值作为输出(例如 1×1、1×2 等)。).我们可以看到，在以 2×2 的步幅插入新行和新列的情况下，它们的 0.0 值乘以单个 1×1 过滤器中的 1.0 值会在输出中产生 0 值。

```py
[[1 2]
 [3 4]]

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_transpose_1 (Conv2DTr (None, 4, 4, 1)           2
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________

[[1\. 0\. 2\. 0.]
 [0\. 0\. 0\. 0.]
 [3\. 0\. 4\. 0.]
 [0\. 0\. 0\. 0.]]
```

**记住**:这是一个人为的例子，我们人为地指定了模型权重，这样我们就可以看到转置卷积运算的效果。

实际上，我们将使用大量的过滤器(例如 64 或 128 个)，更大的内核(例如 3×3、5×5 等)。)，该层将使用随机权重进行初始化，该权重将学习如何在训练过程中有效地对细节进行上采样。

事实上，您可能会想象不同大小的内核会导致不同大小的输出，超过输入的宽度和高度的两倍。在这种情况下，层的“*填充*参数可以设置为“*相同*，以强制输出具有所需的(双倍)输出形状；例如:

```py
...
# example of using padding to ensure that the output is only doubled
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding='same', input_shape=(2, 2, 1)))
```

### 带有 Model 转置层的简单生成器模型

*conv2d 转置*比*upsmampling 2d*层更复杂，但在 GAN 模型中使用时也很有效，尤其是发电机模型。

这两种方法都可以使用，尽管首选 *Conv2DTranspose* 层，这可能是因为更简单的生成器模型和可能更好的结果，尽管众所周知 GAN 的表现和技能很难量化。

我们可以用另一个简单的例子来演示在生成器模型中使用*conv2d 转置*层。

在这种情况下，我们的小 GAN 生成器模型必须生成一个 10×10 的图像，并从潜在空间中获取一个 100 元素的向量作为输入，就像前面的*upsmamping 2d*示例一样。

首先，可以使用一个密集的完全连接层来解释输入向量，并创建足够数量的激活(输出)，这些激活可以被重塑为我们输出图像的低分辨率版本，在本例中，是 5×5 图像的 128 个版本。

```py
...
# define model
model = Sequential()
# define input shape, output enough activations for for 128 5x5 image
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((5, 5, 128)))
```

接下来，可以将 5×5 的要素图上采样为 10×10 的要素图。

我们将对单个过滤器使用 3×3 的内核大小，这将导致输出要素图中的宽度和高度略大于两倍(11×11)。

因此，我们将“*填充*”设置为“相同”，以确保输出尺寸按要求为 10×10。

```py
...
# double input from 128 5x5 to 1 10x10 feature map
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding='same'))
```

将这些联系在一起，完整的示例如下所示。

```py
# example of using transpose conv in a simple generator model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import Conv2D
# define model
model = Sequential()
# define input shape, output enough activations for for 128 5x5 image
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((5, 5, 128)))
# double input from 128 5x5 to 1 10x10 feature map
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding='same'))
# summarize model
model.summary()
```

运行该示例会创建模型并总结每个层的输出形状。

我们可以看到密集层输出 3200 次激活，然后这些激活被重新整形为 128 个形状为 5×5 的特征图。

conv2d 转置层将宽度和高度增加了一倍，达到 10×10，从而形成面积增加了四倍的单一要素地图。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 3200)              323200
_________________________________________________________________
reshape_1 (Reshape)          (None, 5, 5, 128)         0
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 10, 10, 1)         1153
=================================================================
Total params: 324,353
Trainable params: 324,353
Non-trainable params: 0
_________________________________________________________________
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [深度学习卷积算法指南](https://arxiv.org/abs/1603.07285)，2016。
*   [去进化网络](https://ieeexplore.ieee.org/document/5539957)，2010。
*   [反卷积层和卷积层一样吗？](https://arxiv.org/abs/1609.07009)，2016 年。
*   [可视化和理解卷积网络](https://arxiv.org/abs/1311.2901)，2013。
*   [语义分割的全卷积网络](https://arxiv.org/abs/1411.4038)，2014。

### 应用程序接口

*   [Keras 卷积层应用编程接口](https://keras.io/layers/convolutional/)

### 文章

*   [卷积运算项目，GitHub](https://github.com/vdumoulin/conv_arithmetic) 。
*   [什么是反进化层？，数据科学堆栈交换](https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers)。

## 摘要

在本教程中，您发现了在生成图像时，如何在创成式对抗网络中使用 UpSampling2D 和 conv2d 转置层。

具体来说，您了解到:

*   GAN 架构中的生成模型需要对输入数据进行上采样，以生成输出图像。
*   上采样层是一个没有权重的简单层，它将使输入的维度加倍，并且可以在传统卷积层之后的生成模型中使用。
*   转置卷积层是一个反向卷积层，它将对输入进行上采样，并学习如何在模型训练过程中填充细节。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。