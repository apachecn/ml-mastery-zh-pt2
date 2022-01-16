# 卷积神经网络填充和步长的温和介绍

> 原文：<https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/>

最后更新于 2019 年 8 月 16 日

卷积神经网络中的[卷积层](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)系统地对输入应用滤波器，并创建输出特征图。

虽然卷积层非常简单，但它能够实现复杂而令人印象深刻的结果。然而，对于过滤器的形状如何影响输出特征图的形状以及相关的配置超参数(例如填充和步幅)应该如何配置，开发直觉可能是一项挑战。

在本教程中，您将发现卷积神经网络中滤波器大小、填充需求和步长的直觉。

完成本教程后，您将知道:

*   过滤器大小或内核大小如何影响输出要素图的形状。
*   过滤器大小如何在要素地图中创建边框效果，以及如何通过填充来克服它。
*   如何使用输入图像上过滤器的步幅对输出要素图的大小进行下采样。

**用我的新书[计算机视觉深度学习](https://machinelearningmastery.com/deep-learning-for-computer-vision/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![A Gentle Introduction to Padding and Stride for Convolutional Neural Networks](img/edb84d9f2728096fb2506ea2e86c0760.png)

卷积神经网络填充和步幅的温和介绍
图片由[红星](https://www.flickr.com/photos/rainriver/4138418274/)提供，版权所有。

## 教程概述

本教程分为五个部分；它们是:

1.  卷积层
2.  边界效应问题
3.  过滤器大小的影响(内核大小)
4.  使用填充修复边框效果问题
5.  带步幅的下采样输入

## 卷积层

在卷积神经网络中，卷积层负责对输入系统应用一个或多个滤波器。

滤波器与输入图像的相乘产生单个输出。输入通常是三维图像(例如，行、列和通道)，反过来，过滤器也是三维的，具有与输入图像相同数量的通道和更少的行和列。这样，过滤器被重复应用于输入图像的每个部分，产生激活的二维输出图，称为特征图。

Keras 提供了一个卷积层的实现，称为 Conv2D。

它要求您根据行(高度)、列(宽度)和通道(深度)或*【行、列、通道】*来指定输入图像的预期形状。

过滤器包含在层的训练过程中必须学习的权重。过滤器权重表示过滤器将检测到的结构或特征，激活强度表示检测到特征的程度。

该层要求指定过滤器的数量和形状。

我们可以用一个小例子来证明这一点。在本例中，我们定义了一个输入图像或样本，它有一个通道，是一个 8 像素乘 8 像素的正方形，所有值都为 0，中间有一条 2 像素宽的垂直线。

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

接下来，我们可以定义一个模型，该模型期望输入样本具有形状(8，8，1)，并且具有单个隐藏卷积层，该卷积层具有三个像素乘三个像素的形状的单个滤波器。

```py
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
# summarize model
model.summary()
```

作为模型初始化的一部分，使用随机权重初始化过滤器。我们将覆盖随机权重并硬编码我们自己的 3×3 过滤器，它将检测垂直线。

也就是说，当检测到垂直线时，过滤器将被强激活，当检测不到垂直线时，过滤器将被弱激活。我们预计，通过在输入图像上应用此过滤器，输出特征图将显示检测到垂直线。

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

结果是一个四维输出，具有一个批次、给定数量的行和列以及一个过滤器，或*【批次、行、列、过滤器】*。

我们可以在单个特征图中打印激活，以确认检测到该行。

```py
# enumerate rows
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
```

将所有这些结合在一起，下面列出了完整的示例。

```py
# example of using a single convolutional layer
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
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
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

值得注意的是，单个隐藏卷积层将采用 8×8 像素的输入图像，并将产生尺寸为 6×6 的特征图。我们将在下一节探讨为什么会出现这种情况。

我们还可以看到该层有 10 个参数，即滤波器的 9 个权重(3×3)和偏差的 1 个权重。

最后，打印特征图。通过查看 6×6 矩阵中的数字，我们可以看到手动指定的过滤器确实检测到了输入图像中间的垂直线。

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

## 边界效应问题

在上一节中，我们定义了一个三像素高、三像素宽(行、列)的单个过滤器。

我们看到，对 8×8 输入图像应用 3×3 滤波器(在 Keras 中称为核大小)产生了大小为 6×6 的特征图。

也就是说，具有 64 个像素的输入图像被简化为具有 36 个像素的特征图。其他 28 像素去哪了？

滤波器被系统地应用于输入图像。它从图像的左上角开始，一次从左向右移动一个像素列，直到滤镜的边缘到达图像的边缘。

对于应用于 8×8 输入图像的 3×3 像素滤波器，我们可以看到它只能应用 6 次，导致输出特征图中的宽度为 6。

例如，让我们逐一查看输入图像(左)点积(")的六个补丁操作员)过滤器(右):

```py
0, 0, 0   0, 1, 0
0, 0, 0 . 0, 1, 0 = 0
0, 0, 0   0, 1, 0
```

向右移动一个像素:

```py
0, 0, 1   0, 1, 0
0, 0, 1 . 0, 1, 0 = 0
0, 0, 1   0, 1, 0
```

向右移动一个像素:

```py
0, 1, 1   0, 1, 0
0, 1, 1 . 0, 1, 0 = 3
0, 1, 1   0, 1, 0
```

向右移动一个像素:

```py
1, 1, 0   0, 1, 0
1, 1, 0 . 0, 1, 0 = 3
1, 1, 0   0, 1, 0
```

向右移动一个像素:

```py
1, 0, 0   0, 1, 0
1, 0, 0 . 0, 1, 0 = 0
1, 0, 0   0, 1, 0
```

向右移动一个像素:

```py
0, 0, 0   0, 1, 0
0, 0, 0 . 0, 1, 0 = 0
0, 0, 0   0, 1, 0
```

这给出了输出要素图的第一行和每一列:

```py
0.0, 0.0, 3.0, 3.0, 0.0, 0.0
```

要素图输入大小的减少称为边界效果。它是由滤镜与图像边框的相互作用引起的。

这对于大图像和小滤镜来说通常不是问题，但对于小图像来说可能是问题。一旦堆叠了多个卷积层，这也会成为一个问题。

例如，下面是更新为具有两个堆叠卷积层的相同模型。

这意味着对 8×8 输入图像应用 3×3 滤波器，以产生如前一节所述的 6×6 特征图。然后将 3×3 滤波器应用于 6×6 特征图。

```py
# example of stacked convolutional layers
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
model.add(Conv2D(1, (3,3)))
# summarize model
model.summary()
```

运行该示例总结了每一层的输出形状。

我们可以看到，将过滤器应用于第一层的要素图输出，反过来会产生更小的 4×4 要素图。

当我们开发具有几十或几百层的非常深的卷积神经网络模型时，这可能成为一个问题。我们的要素地图中的数据将会用完，无法继续操作。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 6, 6, 1)           10
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 1)           10
=================================================================
Total params: 20
Trainable params: 20
Non-trainable params: 0
_________________________________________________________________
```

## 过滤器大小的影响(内核大小)

不同大小的过滤器将检测输入图像中不同大小的特征，并进而产生不同大小的特征图。

对于较大的输入图像，通常使用 3×3 尺寸的滤波器，或许 5×5 甚至 7×7 尺寸的滤波器。

例如，下面是一个模型的例子，其中单个过滤器被更新为使用 5×5 像素的过滤器大小。

```py
# example of a convolutional layer
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (5,5), input_shape=(8, 8, 1)))
# summarize model
model.summary()
```

运行示例表明，5×5 过滤器只能应用于 8×8 输入图像 4 次，从而产生 4×4 的要素图输出。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 4, 4, 1)           26
=================================================================
Total params: 26
Trainable params: 26
Non-trainable params: 0
_________________________________________________________________
```

观察两种极端情况可能有助于进一步发展过滤器尺寸和输出特征图之间关系的直觉。

第一种是 1×1 像素大小的滤镜。

```py
# example of a convolutional layer
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (1,1), input_shape=(8, 8, 1)))
# summarize model
model.summary()
```

运行该示例表明，输出要素图与输入要素图具有相同的大小，特别是 8×8。这是因为滤波器只有一个权重(和一个偏差)。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 8, 8, 1)           2
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________
```

另一个极端是与输入相同大小的滤波器，在本例中为 8×8 像素。

```py
# example of a convolutional layer
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (8,8), input_shape=(8, 8, 1)))
# summarize model
model.summary()
```

运行该示例，我们可以看到，如您所料，输入图像中的每个像素都有一个权重(偏差为 64 + 1)，输出是一个具有单个像素的特征图。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 1, 1, 1)           65
=================================================================
Total params: 65
Trainable params: 65
Non-trainable params: 0
_________________________________________________________________
```

现在，我们已经熟悉了过滤器大小对生成的要素图大小的影响，让我们看看如何才能避免丢失像素。

## 使用填充修复边框效果问题

默认情况下，滤镜从图像的左侧开始，滤镜的左侧位于图像的最左侧像素。然后，过滤器一次一列地跨过图像，直到过滤器的右侧位于图像的最右侧像素上。

对图像应用滤镜的另一种方法是确保图像中的每个像素都有机会位于滤镜的中心。

默认情况下，情况并非如此，因为输入边缘的像素只暴露在滤波器的边缘。通过在图像帧外启动过滤器，它为图像边界上的像素提供了更多与过滤器交互的机会，也为过滤器检测特征提供了更多机会，进而提供了与输入图像形状相同的输出特征图。

例如，在对 8×8 输入图像应用 3×3 滤波器的情况下，我们可以在图像外部周围添加一个像素的边框。这具有人工创建 10×10 输入图像的效果。当应用 3×3 滤波器时，会产生 8×8 的特征图。当应用过滤器时，添加的像素值可以具有对点积运算没有影响的零值。

```py
x, x, x   0, 1, 0
x, 0, 0 . 0, 1, 0 = 0
x, 0, 0   0, 1, 0
```

向图像边缘添加像素称为填充。

在 Keras 中，这是通过 Conv2D 层上的“*填充*”参数指定的，该参数的默认值为“*有效*”(无填充)。这意味着过滤器仅适用于输入的有效方式。

“*相同的“*填充*”值*计算输入图像(或特征图)所需的填充，并将其相加，以确保输出与输入具有相同的形状。

在我们的工作示例中，下面的示例向卷积层添加了填充。

```py
# example a convolutional layer with padding
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), padding='same', input_shape=(8, 8, 1)))
# summarize model
model.summary()
```

运行该示例演示了输出要素图的形状与输入图像相同:填充具有所需的效果。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 8, 8, 1)           10
=================================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________________________________
```

填充的增加允许以这样一种方式开发非常深的模型，使得特征地图不会缩小到零。

下面的例子用三个堆叠的卷积层演示了这一点。

```py
# example a deep cnn with padding
from keras.models import Sequential
from keras.layers import Conv2D
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), padding='same', input_shape=(8, 8, 1)))
model.add(Conv2D(1, (3,3), padding='same'))
model.add(Conv2D(1, (3,3), padding='same'))
# summarize model
model.summary()
```

运行该示例，我们可以看到，通过添加填充，输出要素图的形状保持固定在 8×8 甚至三层深。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 8, 8, 1)           10
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 1)           10
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 1)           10
=================================================================
Total params: 30
Trainable params: 30
Non-trainable params: 0
_________________________________________________________________
```

## 带步幅的下采样输入

滤镜在图像上从左到右、从上到下移动，水平移动时改变一个像素的列，垂直移动时改变一个像素的行。

滤波器对输入图像的应用之间的移动量被称为步幅，并且在高度和宽度维度上几乎总是对称的。

二维中的默认步幅是(1，1)表示高度和宽度移动，在需要时执行。这种默认在大多数情况下运行良好。

步幅可以改变，这对于如何将过滤器应用于图像以及由此产生的特征图的大小都有影响。

例如，步幅可以更改为(2，2)。这具有这样的效果，即在创建要素地图时，过滤器的每次水平移动向右移动两个像素，过滤器的每次垂直移动向下移动两个像素。

我们可以用一个例子来演示这一点，该例子使用带有垂直线(左)点积(")的 8×8 图像运算符),垂直线滤镜(右)的步长为两个像素:

```py
0, 0, 0   0, 1, 0
0, 0, 0 . 0, 1, 0 = 0
0, 0, 0   0, 1, 0
```

向右移动两个像素:

```py
0, 1, 1   0, 1, 0
0, 1, 1 . 0, 1, 0 = 3
0, 1, 1   0, 1, 0
```

向右移动两个像素:

```py
1, 0, 0   0, 1, 0
1, 0, 0 . 0, 1, 0 = 0
1, 0, 0   0, 1, 0
```

我们可以看到，3×3 滤波器对于 8×8 输入图像只有三种有效的应用，步长为 2。这在垂直维度上是相同的。

这具有以这样的方式应用滤波器的效果，即正常特征图输出(6×6)被下采样，使得每个维度的大小减少一半(3×3)，导致像素数量的 1/4(36 个像素减少到 9 个)。

步幅可以通过“*步幅*参数在 *Conv2D* 层上的 Keras 中指定，并指定为具有高度和宽度的元组。

该示例演示了我们的手动垂直线滤波器在 8×8 输入图像上的应用，其中卷积层的步长为 2。

```py
# example of vertical line filter with a stride of 2
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
model.add(Conv2D(1, (3,3), strides=(2, 2), input_shape=(8, 8, 1)))
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

运行该示例，我们可以从模型的摘要中看到，输出要素图的形状将为 3×3。

将手工制作的过滤器应用于输入图像并打印生成的激活特征图，我们可以看到，实际上，过滤器仍然检测到垂直线，并且可以用更少的信息来表示这一发现。

在模型中使用的滤波器或模型体系结构的更深入的知识允许在结果特征图中进行一些压缩的一些情况下，下采样可能是期望的。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 3, 3, 1)           10
=================================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________________________________

[0.0, 3.0, 0.0]
[0.0, 3.0, 0.0]
[0.0, 3.0, 0.0]
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

## 摘要

在本教程中，您发现了卷积神经网络中滤波器大小、填充需求和步长的直觉。

具体来说，您了解到:

*   过滤器大小或内核大小如何影响输出要素图的形状。
*   过滤器大小如何在要素地图中创建边框效果，以及如何通过填充来克服它。
*   如何使用输入图像上过滤器的步幅对输出要素图的大小进行下采样。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。