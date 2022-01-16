# 如何在 Keras 中对图像像素归一化、居中和标准化

> 原文：<https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/>

最后更新于 2019 年 7 月 5 日

在训练或评估模型期间，在将图像作为输入提供给深度学习神经网络模型之前，必须缩放图像中的像素值。

传统上，图像必须在模型开发之前进行缩放，并以缩放后的格式存储在内存或磁盘上。

另一种方法是在训练或模型评估过程中及时使用首选的缩放技术来缩放图像。Keras 通过 ImageDataGenerator 类和 API 支持这种类型的图像数据的数据准备。

在本教程中，您将发现如何使用 ImageDataGenerator 类在拟合和评估深度学习神经网络模型时及时缩放像素数据。

完成本教程后，您将知道:

*   如何配置和使用 ImageDataGenerator 类来训练、验证和测试图像数据集。
*   在拟合和评估卷积神经网络模型时，如何使用 ImageDataGenerator 对像素值进行归一化。
*   在拟合和评估卷积神经网络模型时，如何使用 ImageDataGenerator 对像素值进行中心化和标准化。

**用我的新书[计算机视觉深度学习](https://machinelearningmastery.com/deep-learning-for-computer-vision/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Normalize, Center, and Standardize Images With the ImageDataGenerator in Keras](img/a077f9778b206ed8d0bb1846380f7b74.png)

如何使用 Keras 中的 ImageDataGenerator 对图像进行规范化、居中和标准化[萨加尔](https://www.flickr.com/photos/27568572@N06/14580422837/)拍摄的照片，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  MNIST 手写图像类别数据集
2.  用于像素缩放的图像数据生成器类
3.  如何使用图像数据生成器规范化图像
4.  如何使用图像数据生成器将图像居中
5.  如何使用图像数据生成器标准化图像

## MNIST 手写图像类别数据集

在我们深入研究 ImageDataGenerator 类用于准备图像数据之前，我们必须选择一个图像数据集来测试生成器。

[MNIST 问题](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/)，是一个由 70，000 幅手写数字图像组成的图像分类问题。

该问题的目标是将手写数字的给定图像分类为从 0 到 9 的整数。因此，这是一个多类图像分类问题。

该数据集作为 Keras 库的一部分提供，可以通过调用[Keras . datasets . mnist . load _ data()函数](https://keras.io/datasets/#mnist-database-of-handwritten-digits)自动下载(如果需要)并加载到内存中。

该函数返回两个元组:一个用于训练输入和输出，一个用于测试输入和输出。例如:

```py
# example of loading the MNIST dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

我们可以加载 MNIST 数据集并汇总数据集。下面列出了完整的示例。

```py
# load and summarize the MNIST dataset
from keras.datasets import mnist
# load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# summarize dataset shape
print('Train', train_images.shape, train_labels.shape)
print('Test', (test_images.shape, test_labels.shape))
# summarize pixel values
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
```

运行该示例首先将数据集加载到内存中。然后报告训练和测试数据集的形状。

我们可以看到，所有图像都是 28×28 像素，黑白图像只有一个通道。训练数据集有 60，000 幅图像，测试数据集有 10，000 幅图像。

我们还可以看到，像素值是介于 0 和 255 之间的整数值，并且两个数据集之间像素值的平均值和标准差相似。

```py
Train (60000, 28, 28) (60000,)
Test ((10000, 28, 28), (10000,))
Train 0 255 33.318421449829934 78.56748998339798
Test 0 255 33.791224489795916 79.17246322228644
```

我们将使用这个数据集来探索使用 Keras 中的 ImageDataGenerator 类的不同像素缩放方法。

## 用于像素缩放的图像数据生成器类

Keras 中的 ImageDataGenerator 类提供了一套在建模之前缩放图像数据集中像素值的技术。

该类将包装您的图像数据集，然后在需要时，它将在训练、验证或评估期间分批将图像返回给算法，并及时应用缩放操作。这为使用神经网络建模时缩放图像数据提供了一种高效便捷的方法。

ImageDataGenerator 类的用法如下。

*   1.加载数据集。
*   2.配置 ImageDataGenerator(例如，构建一个实例)。
*   3.计算图像统计(如调用 *fit()* 函数)。
*   4.使用生成器来拟合模型(例如，将实例传递给 *fit_generator()* 函数)。
*   5.使用生成器评估模型(例如，将实例传递给 *evaluate_generator()* 函数)。

ImageDataGenerator 类支持多种像素缩放方法，以及一系列[数据扩充技术](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)。我们将集中讨论像素缩放技术，并将数据扩充方法留给后面的讨论。

ImageDataGenerator 类支持的三种主要像素缩放技术如下:

*   **像素归一化**:将像素值缩放到 0-1 的范围。
*   **像素居中**:缩放像素值至零平均值。
*   **像素标准化**:缩放像素值，使其平均值和单位方差为零。

像素标准化在两个层次上得到支持:每图像(称为样本方式)或每数据集(称为特征方式)。具体而言，标准化像素值所需的均值和/或均值和标准偏差统计量可以仅从每个图像中的像素值(样本方式)或跨整个训练数据集(特征方式)来计算。

支持其他像素缩放方法，如 ZCA、增亮等，但我们将重点关注这三种最常见的方法。

通过在构造实例时为 ImageDataGenerator 指定参数来选择像素缩放；例如:

```py
# create and configure the data generator
datagen = ImageDataGenerator(...)
```

接下来，如果所选的缩放方法要求跨训练数据集计算统计数据，则可以通过调用 *fit()* 函数来计算和存储这些统计数据。

评估和选择模型时，通常会在训练数据集上计算这些统计数据，然后将它们应用于验证和测试数据集。

```py
# calculate scaling statistics on the training dataset
datagen.fit(trainX)
```

数据生成器一旦准备好，就可以用来拟合神经网络模型，方法是调用 *flow()* 函数来检索返回成批样本的迭代器，并将其传递给 *fit_generator()* 函数。

```py
# get batch iterator
train_iterator = datagen.flow(trainX, trainy)
# fit model
model.fit_generator(train_iterator, ...)
```

如果需要验证数据集，可以从相同的数据生成器创建单独的批处理迭代器，该迭代器将执行相同的像素缩放操作，并使用在训练数据集上计算的任何所需统计信息。

```py
# get batch iterator for training
train_iterator = datagen.flow(trainX, trainy)
# get batch iterator for validation
val_iterator = datagen.flow(valX, valy)
# fit model
model.fit_generator(train_iterator, validation_data=val_iterator, ...)
```

一旦适合，就可以通过为测试数据集创建批处理迭代器并在模型上调用 *evaluate_generator()* 函数来评估模型。

同样，将执行相同的像素缩放操作，并且如果需要，将使用在训练数据集上计算的任何统计数据。

```py
# get batch iterator for testing
test_iterator = datagen.flow(testX, testy)
# evaluate model loss on test dataset
loss = model.evaluate_generator(test_iterator, ...)
```

现在我们已经熟悉了如何使用 ImageDataGenerator 类来缩放像素值，让我们看一些具体的例子。

## 如何使用图像数据生成器规范化图像

ImageDataGenerator 类可用于将像素值从 0-255 范围重新缩放到神经网络模型首选的 0-1 范围。

将数据缩放到 0-1 的范围通常被称为归一化。

这可以通过将重新缩放参数设置为一个比率来实现，通过该比率可以将每个像素相乘以获得所需的范围。

在这种情况下，比率是 1/255 或大约 0.0039。例如:

```py
# create generator (1.0/255.0 = 0.003921568627451)
datagen = ImageDataGenerator(rescale=1.0/255.0)
```

在这种情况下，不需要安装 ImageDataGenerator，因为没有需要计算的全局统计信息。

接下来，可以使用生成器为训练和测试数据集创建迭代器。我们将使用 64 的批量。这意味着图像的每个训练和测试数据集被分成 64 个图像组，当从迭代器返回时，这些图像组将被缩放。

通过打印每个迭代器的长度，我们可以看到一个纪元中有多少批次，例如一次通过训练数据集。

```py
# prepare an iterators to scale images
train_iterator = datagen.flow(trainX, trainY, batch_size=64)
test_iterator = datagen.flow(testX, testY, batch_size=64)
print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))
```

然后，我们可以通过检索第一批缩放图像并检查最小和最大像素值来确认像素归一化是否已按预期执行。

```py
# confirm the scaling works
batchX, batchy = train_iterator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
```

接下来，我们可以使用数据生成器来拟合和评估模型。我们将定义一个简单的卷积神经网络模型，并将其拟合到 5 个时期的 *train_iterator* 上，其中 60，000 个样本除以每批 64 个样本，即每时期约 938 个批次。

```py
# fit model with generator
model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=5)
```

一旦拟合，我们将在测试数据集上评估模型，大约 10，000 幅图像除以每批 64 个样本，或者在单个时期内大约 157 个步骤。

```py
_, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)
print('Test Accuracy: %.3f' % (acc * 100))
```

我们可以把这一切联系在一起；下面列出了完整的示例。

```py
# example of using ImageDataGenerator to normalize images
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# reshape dataset to have a single channel
width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))
# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)
# confirm scale of pixels
print('Train min=%.3f, max=%.3f' % (trainX.min(), trainX.max()))
print('Test min=%.3f, max=%.3f' % (testX.min(), testX.max()))
# create generator (1.0/255.0 = 0.003921568627451)
datagen = ImageDataGenerator(rescale=1.0/255.0)
# prepare an iterators to scale images
train_iterator = datagen.flow(trainX, trainY, batch_size=64)
test_iterator = datagen.flow(testX, testY, batch_size=64)
print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))
# confirm the scaling works
batchX, batchy = train_iterator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit model with generator
model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=5)
# evaluate model
_, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)
print('Test Accuracy: %.3f' % (acc * 100))
```

运行该示例首先报告训练集和测试集上的最小和最大像素值。这证实了原始数据确实具有 0-255 范围内的像素值。

接下来，创建数据生成器并准备迭代器。我们可以看到，训练数据集每个时期有 938 个批次，测试数据集每个时期有 157 个批次。

我们从数据集中检索第一批，并确认它包含 64 幅图像，高度和宽度(行和列)为 28 个像素和 1 个通道，新的最小和最大像素值分别为 0 和 1。这证实了正常化取得了预期的效果。

```py
Train min=0.000, max=255.000
Test min=0.000, max=255.000
Batches train=938, test=157
Batch shape=(64, 28, 28, 1), min=0.000, max=1.000
```

然后将该模型拟合到归一化的图像数据上。在 CPU 上训练不需要很长时间。最后，在测试数据集中评估模型，应用相同的规范化。

```py
Epoch 1/5
938/938 [==============================] - 12s 13ms/step - loss: 0.1841 - acc: 0.9448
Epoch 2/5
938/938 [==============================] - 12s 13ms/step - loss: 0.0573 - acc: 0.9826
Epoch 3/5
938/938 [==============================] - 12s 13ms/step - loss: 0.0407 - acc: 0.9870
Epoch 4/5
938/938 [==============================] - 12s 13ms/step - loss: 0.0299 - acc: 0.9904
Epoch 5/5
938/938 [==============================] - 12s 13ms/step - loss: 0.0238 - acc: 0.9928
Test Accuracy: 99.050
```

既然我们已经熟悉了如何使用 ImageDataGenerator 进行图像规范化，那么让我们来看看像素居中和标准化的例子。

## 如何使用图像数据生成器将图像居中

另一种流行的像素缩放方法是计算整个训练数据集的平均像素值，然后从每个图像中减去它。

这被称为居中，并且具有将像素值的分布居中为零的效果:也就是说，居中图像的平均像素值将为零。

ImageDataGenerator 类指的是定心，它使用在训练数据集上计算的平均值作为特征定心。它要求在缩放之前对训练数据集计算统计数据。

```py
# create generator that centers pixel values
datagen = ImageDataGenerator(featurewise_center=True)
# calculate the mean on the training dataset
datagen.fit(trainX)
```

它不同于计算每个图像的平均像素值，Keras 称之为样本中心化，不需要在训练数据集上计算任何统计数据。

```py
# create generator that centers pixel values
datagen = ImageDataGenerator(samplewise_center=True)
```

在本节中，我们将演示按特征居中。一旦在训练数据集上计算出统计量，我们就可以通过访问和打印来确认该值；例如:

```py
# print the mean calculated on the training dataset.
print(datagen.mean)
```

我们还可以通过计算批处理迭代器返回的一批图像的平均值来确认缩放过程已经达到了预期的效果。我们希望平均值是一个接近于零的小值，但不是零，因为批次中的图像数量很少。

```py
# get a batch
batchX, batchy = iterator.next()
# mean pixel value in the batch
print(batchX.shape, batchX.mean())
```

更好的检查是将批次大小设置为训练数据集的大小(例如 60，000 个样本)，检索一个批次，然后计算平均值。它应该是一个非常小的接近于零的值。

```py
# try to flow the entire training dataset
iterator = datagen.flow(trainX, trainy, batch_size=len(trainX), shuffle=False)
# get a batch
batchX, batchy = iterator.next()
# mean pixel value in the batch
print(batchX.shape, batchX.mean())
```

下面列出了完整的示例。

```py
# example of centering a image dataset
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# reshape dataset to have a single channel
width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))
# report per-image mean
print('Means train=%.3f, test=%.3f' % (trainX.mean(), testX.mean()))
# create generator that centers pixel values
datagen = ImageDataGenerator(featurewise_center=True)
# calculate the mean on the training dataset
datagen.fit(trainX)
print('Data Generator Mean: %.3f' % datagen.mean)
# demonstrate effect on a single batch of samples
iterator = datagen.flow(trainX, trainy, batch_size=64)
# get a batch
batchX, batchy = iterator.next()
# mean pixel value in the batch
print(batchX.shape, batchX.mean())
# demonstrate effect on entire training dataset
iterator = datagen.flow(trainX, trainy, batch_size=len(trainX), shuffle=False)
# get a batch
batchX, batchy = iterator.next()
# mean pixel value in the batch
print(batchX.shape, batchX.mean())
```

运行该示例首先报告训练和测试数据集的平均像素值。

MNIST 数据集只有一个通道，因为图像是黑白的(灰度)，但是如果图像是彩色的，则训练数据集中所有图像的所有通道的平均像素值将被计算，即每个通道没有单独的平均值。

图像数据生成器适合训练数据集，我们可以确认平均像素值与我们自己的手动计算相匹配。

检索单批居中的图像，我们可以确认平均像素值是接近于零的小值。使用整个训练数据集作为批次大小来重复测试，在这种情况下，缩放数据集的平均像素值是非常接近于零的数字，这证实了居中具有期望的效果。

```py
Means train=33.318, test=33.791
Data Generator Mean: 33.318
(64, 28, 28, 1) 0.09971977
(60000, 28, 28, 1) -1.9512918e-05
```

我们可以用上一节开发的卷积神经网络来演示居中。

下面列出了按特征对中的完整示例。

```py
# example of using ImageDataGenerator to center images
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# reshape dataset to have a single channel
width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))
# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)
# create generator to center images
datagen = ImageDataGenerator(featurewise_center=True)
# calculate mean on training dataset
datagen.fit(trainX)
# prepare an iterators to scale images
train_iterator = datagen.flow(trainX, trainY, batch_size=64)
test_iterator = datagen.flow(testX, testY, batch_size=64)
print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))
# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit model with generator
model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=5)
# evaluate model
_, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)
print('Test Accuracy: %.3f' % (acc * 100))
```

运行该示例将准备 ImageDataGenerator，使用在训练数据集上计算的统计信息对图像进行居中。

我们可以看到，表现开始很差，但确实有所提高。居中的像素值将具有大约-227 到 227 的范围，并且神经网络通常使用小输入来更有效地训练。在实践中，正常化之后是居中将是更好的方法。

重要的是，在测试数据集上评估模型，其中使用在训练数据集上计算的平均值将测试数据集中的图像居中。这是为了避免任何数据泄露。

```py
Batches train=938, test=157
Epoch 1/5
938/938 [==============================] - 12s 13ms/step - loss: 12.8824 - acc: 0.2001
Epoch 2/5
938/938 [==============================] - 12s 13ms/step - loss: 6.1425 - acc: 0.5958
Epoch 3/5
938/938 [==============================] - 12s 13ms/step - loss: 0.0678 - acc: 0.9796
Epoch 4/5
938/938 [==============================] - 12s 13ms/step - loss: 0.0464 - acc: 0.9857
Epoch 5/5
938/938 [==============================] - 12s 13ms/step - loss: 0.0373 - acc: 0.9880
Test Accuracy: 98.540
```

## 如何使用图像数据生成器标准化图像

标准化是一种数据缩放技术，它假设数据的分布是高斯分布，并将数据的分布转换为平均值为零，标准偏差为 1。

具有这种分布的数据称为标准高斯分布。当训练神经网络时，这可能是有益的，因为数据集和为零，并且输入是大约-3.0 到 3.0 的粗略范围内的小值(例如， [99.7 的值将落在平均值的三个标准偏差内](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule))。

图像的标准化是通过减去平均像素值并将结果除以像素值的标准偏差来实现的。

均值和标准差统计可以在训练数据集上计算，正如上一节所讨论的，Keras 将其称为特征统计。

```py
# feature-wise generator
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# calculate mean and standard deviation on the training dataset
datagen.fit(trainX)
```

还可以计算统计数据，然后分别用于标准化每个图像，Keras 称之为采样标准化。

```py
# sample-wise standardization
datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
```

在本节中，我们将演示图像标准化的前一种方法或特征方法。效果将是一批批图像，其近似平均值为零，标准偏差为一。

与前一部分一样，我们可以通过一些简单的实验来证实这一点。下面列出了完整的示例。

```py
# example of standardizing a image dataset
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# reshape dataset to have a single channel
width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))
# report pixel means and standard deviations
print('Statistics train=%.3f (%.3f), test=%.3f (%.3f)' % (trainX.mean(), trainX.std(), testX.mean(), testX.std()))
# create generator that centers pixel values
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# calculate the mean on the training dataset
datagen.fit(trainX)
print('Data Generator mean=%.3f, std=%.3f' % (datagen.mean, datagen.std))
# demonstrate effect on a single batch of samples
iterator = datagen.flow(trainX, trainy, batch_size=64)
# get a batch
batchX, batchy = iterator.next()
# pixel stats in the batch
print(batchX.shape, batchX.mean(), batchX.std())
# demonstrate effect on entire training dataset
iterator = datagen.flow(trainX, trainy, batch_size=len(trainX), shuffle=False)
# get a batch
batchX, batchy = iterator.next()
# pixel stats in the batch
print(batchX.shape, batchX.mean(), batchX.std())
```

运行该示例首先报告训练和测试数据集中像素值的平均值和标准偏差。

然后将数据生成器配置为基于特征的标准化，并在训练数据集上计算统计数据，这与我们手动计算统计数据时的预期相匹配。

然后检索单批 64 幅标准化图像，我们可以确认这个小样本的均值和标准差接近预期的标准高斯。

然后在整个训练数据集上重复测试，我们可以确认平均值确实是一个非常小的接近 0.0 的值，标准偏差是一个非常接近 1.0 的值。

```py
Statistics train=33.318 (78.567), test=33.791 (79.172)
Data Generator mean=33.318, std=78.567
(64, 28, 28, 1) 0.010656365 1.0107679
(60000, 28, 28, 1) -3.4560264e-07 0.9999998
```

现在，我们已经确认，像素值的标准化正在按照我们的预期进行，我们可以在拟合和评估卷积神经网络模型的同时应用像素缩放。

下面列出了完整的示例。

```py
# example of using ImageDataGenerator to standardize images
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# reshape dataset to have a single channel
width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))
# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)
# create generator to standardize images
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# calculate mean on training dataset
datagen.fit(trainX)
# prepare an iterators to scale images
train_iterator = datagen.flow(trainX, trainY, batch_size=64)
test_iterator = datagen.flow(testX, testY, batch_size=64)
print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))
# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit model with generator
model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=5)
# evaluate model
_, acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)
print('Test Accuracy: %.3f' % (acc * 100))
```

运行该示例将 ImageDataGenerator 类配置为标准化图像，仅计算训练集所需的统计信息，然后分别准备训练和测试迭代器来拟合和评估模型。

```py
Epoch 1/5
938/938 [==============================] - 12s 13ms/step - loss: 0.1342 - acc: 0.9592
Epoch 2/5
938/938 [==============================] - 12s 13ms/step - loss: 0.0451 - acc: 0.9859
Epoch 3/5
938/938 [==============================] - 12s 13ms/step - loss: 0.0309 - acc: 0.9906
Epoch 4/5
938/938 [==============================] - 13s 13ms/step - loss: 0.0230 - acc: 0.9924
Epoch 5/5
938/938 [==============================] - 13s 14ms/step - loss: 0.0182 - acc: 0.9941
Test Accuracy: 99.120
```

## 扩展ˌ扩张

本节列出了一些您可能希望探索的扩展教程的想法。

*   **颜色**。更新一个示例，将图像数据集用于彩色图像，并确认缩放是在整个图像上执行的，而不是按通道执行的。
*   **采样**。演示像素图像的样本式居中或标准化示例。
*   **ZCA 美白**。演示使用 ZCA 方法准备图像数据的示例。

如果你探索这些扩展，我很想知道。
在下面的评论中发表你的发现。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 应用程序接口

*   [MNIST 手写数字数据库，Keras API](https://keras.io/datasets/#mnist-database-of-handwritten-digits) 。
*   [图像预处理 Keras API](https://keras.io/preprocessing/image/)
*   [序列模型 Keras API](https://keras.io/models/sequential/)

### 文章

*   [MNIST 数据库，维基百科](https://en.wikipedia.org/wiki/MNIST_database)。
*   [68–95–99.7 规则，维基百科](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)。

## 摘要

在本教程中，您发现了如何使用 ImageDataGenerator 类在拟合和评估深度学习神经网络模型时及时缩放像素数据。

具体来说，您了解到:

*   如何配置和使用 ImageDataGenerator 类来训练、验证和测试图像数据集。
*   在拟合和评估卷积神经网络模型时，如何使用 ImageDataGenerator 对像素值进行归一化。
*   在拟合和评估卷积神经网络模型时，如何使用 ImageDataGenerator 对像素值进行中心化和标准化。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。