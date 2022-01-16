# 通道在前和通道在后图像格式的温和介绍

> 原文：<https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/>

最后更新于 2019 年 9 月 12 日

彩色图像具有高度、宽度和颜色通道尺寸。

当表示为[三维阵列](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)时，图像数据的通道维度默认是最后一个，但可能会移动到第一个维度，通常是出于表现调整的原因。

使用这两种“*通道排序格式*”并准备数据以满足特定的首选通道排序可能会让初学者感到困惑。

在本教程中，您将发现通道排序格式，如何准备和操作图像数据以满足格式，以及如何为不同的通道排序配置 Keras 深度学习库。

完成本教程后，您将知道:

*   图像的三维阵列结构以及通道第一和通道最后的阵列格式。
*   如何添加通道尺寸以及如何在通道格式之间转换图像。
*   Keras 深度学习库如何管理首选渠道订购，以及如何更改和查询此首选项。

**用我的新书[计算机视觉深度学习](https://machinelearningmastery.com/deep-learning-for-computer-vision/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2019 年 4 月更新**:修正代码评论错别字(感谢安东尼奥)。
*   **2019 年 9 月更新**:针对 Keras 2.2.5 API 的小改动进行了更新。将 *set_image_dim_ordering()* 的用法更改为 *set_image_data_format()。*

## 教程概述

本教程分为三个部分；它们是:

1.  作为三维阵列的图像
2.  操纵图像通道
3.  Keras 通道排序

## 作为三维阵列的图像

图像可以作为三维数组存储在内存中。

通常，图像格式有一个行维度(高度)、一个列维度(宽度)和一个通道维度。

如果图像是黑白的(灰度)，通道尺寸可能不显式存在，例如，图像中的每个(行、列)坐标都有一个无符号整数像素值。

彩色图像通常有三个通道，用于红色、绿色和蓝色分量(行、列)坐标处的像素值。

深度学习神经网络要求图像数据以三维阵列的形式提供。

即使您的图像是灰度图像，这也适用。在这种情况下，必须增加单色通道的附加尺寸。

有两种方法可以将图像数据表示为三维数组。第一种方法是将通道作为数组中的最后一个或第三维。这叫做“*通道最后*”。第二种方法是将通道作为阵列中的第一维，称为“T2 通道优先”。

*   **通道最后**。图像数据以三维阵列表示，其中最后一个通道表示彩色通道，例如*【行】【列】【通道】*。
*   **通道优先**。图像数据以三维阵列表示，其中第一通道表示彩色通道，例如*【通道】【行】【列】*。

一些图像处理和深度学习库更喜欢通道优先排序，一些更喜欢通道最后排序。因此，熟悉这两种表示图像的方法非常重要。

## 操纵图像通道

您可能需要更改或操作图像通道或通道排序。

这可以使用 NumPy python 库轻松实现。

让我们看一些例子。

在本教程中，我们将使用拉里·科斯特拍摄的菲利普岛企鹅游行的照片，版权所有。

![Phillip Island Penguin Parade](img/79ba7995d6e81ef4e0db78e2ba1202d2.png)

菲利普岛企鹅游行
拉里·科斯特摄，版权所有。

下载图片，放入当前工作目录，文件名为“ *penguin_parade.jpg* ”。

*   [下载照片(企鹅 _ 游行. jpg)](https://machinelearningmastery.com/wp-content/uploads/2019/01/penguin_arade.jpg)

本教程中的代码示例假设安装了[Pillow 库](https://pillow.readthedocs.io/en/stable/)。

### 如何向灰度图像添加通道

灰度图像作为二维数组加载。

在将它们用于建模之前，您可能必须向图像添加一个显式通道尺寸。这不会添加新数据；相反，它改变了数组数据结构，增加了一维的第三个轴来保存灰度像素值。

例如，尺寸为[行][列]的灰度图像可以更改为[行][列][通道]或[通道][行][列]，其中新的[通道]轴为一维。

这可以使用 [expand_dims() NumPy 功能](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html)来实现。“*轴*”参数允许您指定新维度将添加到第一个维度的位置，例如，通道第一个的第一个或通道最后一个的最后一个。

以下示例使用 Pillow 库作为灰度图像加载企鹅游行照片，并演示如何添加通道尺寸。

```py
# example of expanding dimensions
from numpy import expand_dims
from numpy import asarray
from PIL import Image
# load the image
img = Image.open('penguin_arade.jpg')
# convert the image to grayscale
img = img.convert(mode='L')
# convert to numpy array
data = asarray(img)
print(data.shape)
# add channels first
data_first = expand_dims(data, axis=0)
print(data_first.shape)
# add channels last
data_last = expand_dims(data, axis=2)
print(data_last.shape)
```

运行该示例首先使用 Pillow 库加载照片，然后将其转换为灰度图像。

图像对象被转换成一个 NumPy 数组，我们确认数组的形状是二维的，特别是(424，640)。

然后使用 *expand_dims()* 功能通过*轴=0* 将通道添加到数组的前面，并用形状(1，424，640)确认该变化。然后使用相同的函数将通道添加到阵列的末端或第三维，其中*轴=2* ，并且用形状(424，640，1)确认该变化。

```py
(424, 640)
(1, 424, 640)
(424, 640, 1)
```

扩展数组维度的另一种流行方法是使用[重塑()NumPy 函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)并用新形状指定元组；例如:

```py
data = data.reshape((424, 640, 1))
```

### 如何更改图像通道排序

将彩色图像加载为三维阵列后，可以更改通道顺序。

这可以使用[移动轴()NumPy 功能](https://docs.scipy.org/doc/numpy/reference/generated/numpy.moveaxis.html)来实现。它允许您指定源轴和目标轴的索引。

此功能可用于将通道最后格式的数组(如*【行】【列】【通道】*)更改为通道第一格式(如*【通道】【行】【列】*)或相反。

以下示例以通道最后格式加载企鹅游行照片，并使用 *moveaxis()* 功能将其更改为通道第一格式。

```py
# change image from channels last to channels first format
from numpy import moveaxis
from numpy import asarray
from PIL import Image
# load the color image
img = Image.open('penguin_arade.jpg')
# convert to numpy array
data = asarray(img)
print(data.shape)
# change channels last to channels first format
data = moveaxis(data, 2, 0)
print(data.shape)
# change channels first to channels last format
data = moveaxis(data, 0, 2)
print(data.shape)
```

运行该示例首先使用 Pillow 库加载照片，并将其转换为 NumPy 数组，确认图像是以通道的最后格式加载的，形状为(424，640，3)。

然后使用 *moveaxis()* 功能将通道轴从位置 2 移动到位置 0，并确认结果显示通道第一格式(3，424，640)。然后，这种情况发生逆转，将位置 0 的通道再次移动到位置 2。

```py
(424, 640, 3)
(3, 424, 640)
(424, 640, 3)
```

## Keras 通道排序

Keras 深度学习库不知道您希望如何以通道优先或最后一种格式表示图像，但是在使用该库时，必须指定并遵守首选项。

Keras 包装了许多数学库，每个库都有一个首选的通道排序。下面列出了 Keras 可能包装的三个主要库及其首选渠道订购:

*   **TensorFlow** :通道最后顺序。
*   **安诺**:通道一阶。
*   **CNTK** :通道最后顺序。

默认情况下，Keras 被配置为使用 TensorFlow，并且通道排序也默认为最后一个通道。您可以将任一渠道订购用于任何库和 Keras 库。

一些库声称首选通道排序会导致表现的巨大差异。例如，使用 MXNet 数学库作为 Keras 的后端，建议使用通道优先排序以获得更好的表现。

> 我们强烈建议将 image_data_format 更改为 channels _ first。MXNet 在 channels _ first 数据上要快得多。

— [带 MXNet 后端的表现调优 Keras，Apache MXNet](https://github.com/awslabs/keras-apache-mxnet/wiki/Performance-Tuning---Keras-with-MXNet-Backend)

### 默认通道排序

库和首选通道排序列在 Keras 配置文件中，存储在您的主目录下 *~/。keras/keras.json* 。

首选通道排序存储在“ *image_data_format* ”配置设置中，可以设置为“*通道 _ 最后一个*”或“*通道 _ 第一个*”。

例如，下面是一个 *keras.json* 配置文件的内容。在其中，可以看到系统配置为使用张量流和*通道 _last* 顺序。

```py
{
    "image_data_format": "channels_last",
    "backend": "tensorflow",
    "epsilon": 1e-07,
    "floatx": "float32"
}
```

根据您首选的通道排序，您必须准备图像数据以匹配首选排序。

具体而言，这将包括以下任务:

*   调整或扩展任何训练、验证和测试数据的维度以满足期望。
*   定义模型时指定样本的预期输入形状(例如 *input_shape=(28，28，1)* )。

### 特定型号的渠道订购

此外，那些被设计用于处理图像的神经网络层，如 Conv2D，也提供了一个名为“ *data_format* ”的参数，允许您指定通道顺序。例如:

```py
...
model.add(Conv2D(..., data_format='channels_first'))
```

默认情况下，这将使用 Keras 配置文件的“ *image_data_format* ”值中指定的首选顺序。然而，您可以更改给定模型的通道顺序，反过来，数据集和输入形状也必须更改，以使用新的模型通道顺序。

当加载一个用于传输学习的模型，该模型的通道排序不同于您的首选通道排序时，这将非常有用。

### 查询通道排序

您可以通过打印 *image_data_format()* 功能的结果来确认您当前的首选通道订购。下面的例子演示了。

```py
# show preferred channel order
from keras import backend
print(backend.image_data_format())
```

运行该示例将打印您在 Keras 配置文件中配置的首选通道排序。在这种情况下，使用通道最后一种格式。

```py
channels_last
```

如果您希望根据系统首选的通道顺序自动构建模型或以不同的方式准备数据，访问此属性可能会有所帮助；例如:

```py
if backend.image_data_format() == 'channels_last':
	...
else:
	...
```

### 强制通道排序

最后，可以针对特定节目强制进行通道排序。

这可以通过调用 Keras 后端的 *set_image_data_format()* 函数来实现，对于通道优先排序，调用“*channel _ first*”(an ano)或者对于通道最后排序，调用“*channel _ last*”(tensorflow)。

如果您希望一个程序或模型能够一致地运行，而不管 Keras 的默认通道排序配置如何，这将非常有用。

```py
# force a channel ordering
from keras import backend
# force channels-first ordering
backend.set_image_data_format('channels_first')
print(backend.image_data_format())
# force channels-last ordering
backend.set_image_data_format('channels_last')
print(backend.image_data_format())
```

运行示例首先强制通道先排序，然后通道后排序，通过在更改后打印通道排序模式来确认每个配置。

```py
channels_first
channels_last
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [枕蟒库](https://pillow.readthedocs.io/en/stable/)
*   [numpy.expand_dims API](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html)
*   num py . reshape API
*   num py . move axis API
*   [硬后端接口](https://keras.io/backend/)
*   [Keras 卷积层应用编程接口](https://keras.io/layers/convolutional/)

## 摘要

在本教程中，您发现了通道排序格式，如何准备和操作图像数据以满足格式要求，以及如何为不同的通道排序配置 Keras 深度学习库。

具体来说，您了解到:

*   图像的三维阵列结构以及通道第一和通道最后的阵列格式。
*   如何添加通道尺寸以及如何在通道格式之间转换图像。
*   Keras 深度学习库如何管理首选渠道订购，以及如何更改和查询此首选项。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。