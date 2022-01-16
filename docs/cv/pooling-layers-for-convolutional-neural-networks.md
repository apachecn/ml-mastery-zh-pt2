# 卷积神经网络池化层的简单介绍

> 原文：<https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/>

最后更新于 2019 年 7 月 5 日

[卷积神经网络中的卷积层](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)总结输入图像中特征的存在。

输出要素地图的一个问题是，它们对输入中要素的位置很敏感。解决这种敏感性的一种方法是对特征图进行下采样。这具有使所得到的下采样特征图对图像中特征位置的变化更鲁棒的效果，这由技术短语“*局部平移不变性*来指代。”

通过汇总要素图的面片中存在的要素，池化层提供了一种对要素图进行下采样的方法。两种常见的池化方法是平均池化和最大池化，它们分别总结了特征的平均存在和特征的最活跃存在。

在本教程中，您将发现池操作如何工作，以及如何在卷积神经网络中实现它。

完成本教程后，您将知道:

*   需要使用池对要素地图中的要素检测进行下采样。
*   如何计算和实现卷积神经网络中的平均池和最大池？
*   如何在卷积神经网络中使用全局池？

**用我的新书[计算机视觉深度学习](https://machinelearningmastery.com/deep-learning-for-computer-vision/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![A Gentle Introduction to Pooling Layers for Convolutional Neural Networks](img/7185a125006624c395aaf33ac31ebbe0.png)

卷积神经网络的池化层简介
尼古拉斯·a·托内利摄，版权所有。

## 教程概述

本教程分为五个部分；它们是:

1.  联营
2.  检测垂直线
3.  平均池层数
4.  最大池层数
5.  全局池层

## 池化层

卷积神经网络中的卷积层系统地将学习的滤波器应用于输入图像，以便创建总结输入中存在的那些特征的特征图。

卷积层被证明非常有效，在深度模型中堆叠卷积层允许接近输入的层学习低级特征(例如线)，而在模型中更深的层学习高阶或更抽象的特征，例如形状或特定对象。

卷积层的特征映射输出的一个限制是，它们记录输入中特征的精确位置。这意味着输入图像中要素位置的微小移动将导致不同的要素图。这可能发生在对输入图像进行重新裁剪、旋转、移动和其他微小更改时。

从信号处理中解决这个问题的一种常见方法叫做下采样。这是在输入信号的较低分辨率版本被创建的地方，它仍然包含大的或重要的结构元素，没有可能对任务不那么有用的精细细节。

通过改变图像上卷积的[步距](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)，可以用卷积层实现下采样。一种更健壮和常见的方法是使用池层。

池化层是在卷积层之后添加的新层。具体地，在非线性(例如，ReLU)已经被应用于卷积层输出的特征图之后；例如，模型中的层可能如下所示:

1.  输入图像
2.  卷积层
3.  非线性
4.  池化层

在卷积层之后添加池化层是用于在卷积神经网络内排序层的常见模式，在给定的模型中可以重复一次或多次。

池化层分别对每个要素地图进行操作，以创建一组相同数量的池化要素地图。

池化包括选择池化操作，很像应用于要素地图的过滤器。池操作或过滤器的大小小于要素地图的大小；具体来说，几乎总是以 2 像素的步幅应用 2×2 像素。

这意味着池化层将始终将每个要素地图的大小减少 2 倍，例如每个维度减半，将每个要素地图中的像素或值的数量减少到大小的四分之一。例如，应用于 6×6 (36 像素)要素图的池化层将产生 3×3 (9 像素)的输出池化要素图。

池操作是指定的，而不是学习的。池操作中使用的两个常见功能是:

*   **平均池化**:计算要素图上每个斑块的平均值。
*   **最大池化(或最大池化)**:计算要素图每个面片的最大值。

使用池化层并创建下采样或池化要素图的结果是在输入中检测到的要素的汇总版本。它们是有用的，因为卷积层检测到的输入中的特征位置的微小变化将导致具有相同位置的特征的池化特征图。通过池化增加的这种能力被称为模型对本地翻译的不变性。

> 在所有情况下，池化有助于使表示对于输入的小翻译变得近似不变。对翻译的不变性意味着，如果我们翻译少量的输入，大多数合并输出的值不会改变。

—第 342 页，[深度学习](https://amzn.to/2Dl124s)，2016。

现在我们已经熟悉了池层的需求和好处，让我们看一些具体的例子。

## 检测垂直线

在我们看一些池层及其效果的例子之前，让我们开发一个输入图像和卷积层的小例子，稍后我们可以添加和评估池层。

在本例中，我们定义了一个输入图像或样本，它有一个通道，是一个 8 像素乘 8 像素的正方形，所有值都为 0，中间有一条两像素宽的垂直线。

```py
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
data = data.reshape(1, 8, 8, 1)
```

接下来，我们可以定义一个模型，该模型期望输入样本具有形状(8，8，1)，并且具有单个隐藏卷积层，该隐藏卷积层具有形状为 3 像素乘 3 像素的单个滤波器。

然后，将校正后的线性激活函数(简称 ReLU)应用于特征图中的每个值。这是一种简单而有效的非线性，在这种情况下不会改变要素图中的值，但它的存在是因为我们稍后将添加后续的池化层，并且在将非线性应用于要素图后添加池化，例如最佳实践。

```py
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# summarize model
model.summary()
```

作为模型初始化的一部分，使用随机权重初始化过滤器。

相反，我们将硬编码我们自己的 3×3 过滤器，它将检测垂直线。也就是说，当检测到垂直线时，过滤器将被强激活，当检测不到垂直线时，过滤器将被弱激活。我们期望通过在输入图像上应用该过滤器，输出特征图将显示垂直线被检测到。

```py
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
```

接下来，我们可以通过调用模型上的 *predict()* 函数将过滤器应用于我们的输入图像。

```py
# apply filter to input data
yhat = model.predict(data)
```

结果是一个四维输出，包含一个批处理、给定数量的行和列以及一个过滤器，或[批处理、行、列、过滤器]。我们可以在单个特征图中打印激活，以确认检测到该行。

```py
# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

将所有这些结合在一起，下面列出了完整的示例。

```py
# example of vertical line detection with a convolutional layer
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
data = data.reshape(1, 8, 8, 1)
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
# summarize model
model.summary()
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# apply filter to input data
yhat = model.predict(data)
# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

运行示例首先总结模型的结构。

值得注意的是，单个隐藏卷积层将采用 8×8 像素的输入图像，并将产生尺寸为 6×6 的特征图。

我们还可以看到该层有 10 个参数:即滤波器的 9 个权重(3×3)和偏差的 1 个权重。

最后，打印单一要素地图。

通过查看 6×6 矩阵中的数字，我们可以看到手动指定的过滤器确实检测到了输入图像中间的垂直线。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 6, 6, 1)           10
=================================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________________________________

[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
```

我们现在可以看看一些常见的池化方法，以及它们如何影响输出要素图。

## 平均池层

在二维要素地图上，池化通常应用于跨度为(2，2)的 2×2 要素地图块。

平均池包括计算要素地图每个面片的平均值。这意味着特征图的每个 2×2 的正方形被下采样到正方形中的平均值。

例如，上一节中的线检测器卷积滤波器的输出是 6×6 特征图。我们可以考虑将平均池操作手动应用到要素地图的第一行。

输出要素图的第一行(前两行六列)如下所示:

```py
[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
```

第一个池操作应用如下:

```py
average(0.0, 0.0) = 0.0
        0.0, 0.0
```

假设步幅为 2，操作将向左移动两列，并计算平均值:

```py
average(3.0, 3.0) = 3.0
        3.0, 3.0
```

同样，该操作向左移动两列，并计算平均值:

```py
average(0.0, 0.0) = 0.0
        0.0, 0.0
```

这是第一条联营线。结果是平均池操作的第一行:

```py
[0.0, 3.0, 0.0]
```

给定(2，2)步距，操作将向下移动两行，回到第一列，过程继续。

因为下采样操作将每个维度减半，所以我们期望应用于 6×6 要素图的池化输出是新的 3×3 要素图。给定要素图输入的水平对称性，我们期望每行具有相同的平均池值。因此，我们预计前一部分中检测到的线要素图的平均池如下所示:

```py
[0.0, 3.0, 0.0]
[0.0, 3.0, 0.0]
[0.0, 3.0, 0.0]
```

我们可以通过更新上一节中的示例来确认这一点，以使用平均池。

这可以在 Keras 中通过使用*平均池 2D* 层来实现。层的默认 pool_size(如内核大小或过滤器大小)是(2，2)，默认的*步长*是*无*，这种情况下意味着使用 *pool_size* 作为*步长*，将是(2，2)。

```py
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
model.add(AveragePooling2D())
```

下面列出了平均池的完整示例。

```py
# example of average pooling
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
data = data.reshape(1, 8, 8, 1)
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
model.add(AveragePooling2D())
# summarize model
model.summary()
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# apply filter to input data
yhat = model.predict(data)
# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

运行示例首先总结模型。

我们可以从模型摘要中看到，池化层的输入将是一个具有形状(6，6)的单个要素地图，平均池化层的输出将是一个每个维度减半的单个要素地图，具有形状(3，3)。

应用平均池会产生一个新的要素图，该图仍然可以检测到线，尽管是以向下采样的方式，这与我们通过手动计算操作所预期的完全一样。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 6, 6, 1)           10
_________________________________________________________________
average_pooling2d_1 (Average (None, 3, 3, 1)           0
=================================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________________________________

[0.0, 3.0, 0.0]
[0.0, 3.0, 0.0]
[0.0, 3.0, 0.0]
```

平均池运行良好，尽管使用最大池更常见。

## 最大池层

最大池化或最大池化是一种池化操作，用于计算每个要素地图的每个面片中的最大值。

结果是向下采样或池化的要素图，突出显示补丁中最常见的要素，而不是平均池化情况下要素的平均存在。在实践中发现，这种方法比计算机视觉任务(如图像分类)的平均池化更有效。

> 简而言之，原因是要素往往会在要素地图的不同区块上编码某种模式或概念的空间存在(因此，术语“要素地图”)，并且查看不同要素的最大存在比查看它们的平均存在更能提供信息。

—第 129 页，[Python 深度学习](https://amzn.to/2Dnshvc)，2017。

我们可以通过再次将其应用于线检测器卷积运算的输出特征图并手动计算合并特征图的第一行来使最大合并操作具体化。

输出要素图的第一行(前两行六列)如下所示:

```py
[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
[0.0, 0.0, 3.0, 3.0, 0.0, 0.0]
```

第一个最大池操作应用如下:

```py
max(0.0, 0.0) = 0.0
    0.0, 0.0
```

如果步幅为 2，操作将向左移动两列，并计算最大值:

```py
max(3.0, 3.0) = 3.0
    3.0, 3.0
```

同样，操作会向左移动两列，并计算最大值:

```py
max(0.0, 0.0) = 0.0
    0.0, 0.0
```

这是第一条联营线。

结果是最大池操作的第一行:

```py
[0.0, 3.0, 0.0]
```

同样，考虑到为池化提供的要素图的水平对称性，我们期望池化的要素图如下所示:

```py
[0.0, 3.0, 0.0]
[0.0, 3.0, 0.0]
[0.0, 3.0, 0.0]
```

当使用平均池和最大池进行下采样时，所选的线检测器图像和特征图会产生相同的输出。

通过添加 Keras API 提供的*maxpool2d*层，可以将最大池操作添加到工作示例中。

```py
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
model.add(MaxPooling2D())
```

下面列出了最大池化垂直线检测的完整示例。

```py
# example of max pooling
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
data = data.reshape(1, 8, 8, 1)
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
model.add(MaxPooling2D())
# summarize model
model.summary()
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# apply filter to input data
yhat = model.predict(data)
# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

运行示例首先总结模型。

我们可以看到，正如我们现在可能预期的那样，最大池层的输出将是每个维度减半的单个要素地图，形状为(3，3)。

应用最大池会产生一个新的要素图，该图仍然可以检测到线，尽管是以向下采样的方式。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 6, 6, 1)           10
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 3, 3, 1)           0
=================================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________________________________

[0.0, 3.0, 0.0]
[0.0, 3.0, 0.0]
[0.0, 3.0, 0.0]
```

## 全局池层

还有另一种类型的池，有时被称为全局池。

全局池不是对输入要素图的面片进行向下采样，而是将整个要素图向下采样为单个值。这与将*池大小*设置为输入要素图的大小相同。

全局池可用于模型中，以积极总结图像中某个特征的存在。它有时也用于模型中，作为使用完全连接的层从要素地图过渡到模型输出预测的替代方法。

Keras 分别通过*global average pool 2d*和*global maxpool 2d*类支持全局平均池和全局最大池。

例如，我们可以向用于垂直线检测的卷积模型添加全局最大池。

```py
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
model.add(GlobalMaxPooling2D())
```

结果将是一个单一的值，它将总结输入图像中垂直线的最强激活或存在。

完整的代码列表如下。

```py
# example of using global max pooling
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import GlobalMaxPooling2D
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
data = data.reshape(1, 8, 8, 1)
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
model.add(GlobalMaxPooling2D())
# summarize model
model.summary()
# # define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# apply filter to input data
yhat = model.predict(data)
# enumerate rows
print(yhat)
```

运行示例首先总结模型

我们可以看到，正如预期的那样，全局池层的输出是一个单一值，它总结了单一要素地图中要素的存在。

接下来，打印模型的输出，在要素图上显示全局最大池化的效果，打印单个最大激活。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 6, 6, 1)           10
_________________________________________________________________
global_max_pooling2d_1 (Glob (None, 1)                 0
=================================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________________________________

[[3.]]
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 邮件

*   [机器学习卷积神经网络速成班](https://machinelearningmastery.com/crash-course-convolutional-neural-networks/)

### 书

*   第九章:卷积网络，[深度学习](https://amzn.to/2Dl124s)，2016。
*   第五章:计算机视觉深度学习，[Python 深度学习](https://amzn.to/2Dnshvc)，2017。

### 应用程序接口

*   [Keras 卷积层应用编程接口](https://keras.io/layers/convolutional/)
*   [Keras 池化层应用编程接口](https://keras.io/layers/pooling/)

## 摘要

在本教程中，您发现了池操作如何工作，以及如何在卷积神经网络中实现它。

具体来说，您了解到:

*   需要使用池对要素地图中的要素检测进行下采样。
*   如何计算和实现卷积神经网络中的平均池和最大池？
*   如何在卷积神经网络中使用全局池？

你有什么问题吗？
在下面的评论中提问，我会尽力回答。