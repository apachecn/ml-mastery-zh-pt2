# 如何使用测试时间扩充做出更好的预测

> 原文：<https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/>

最后更新于 2020 年 4 月 3 日

[数据增广](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)是一种在为计算机视觉问题训练神经网络模型时，经常用来提高表现和减少泛化误差的技术。

当用拟合模型进行预测时，也可以应用图像数据扩充技术，以便允许模型对测试数据集中每个图像的多个不同版本进行预测。可以对增强图像上的预测进行平均，这可以导致更好的预测表现。

在本教程中，您将发现用于提高图像分类任务模型表现的测试时间扩展。

完成本教程后，您将知道:

*   测试时间扩充是数据扩充技术的应用，通常在进行预测的训练中使用。
*   如何在 Keras 中从零开始实现测试时间扩充？
*   如何在标准图像分类任务中使用测试时间扩充来提高卷积神经网络模型的表现？

**用我的新书[计算机视觉深度学习](https://machinelearningmastery.com/deep-learning-for-computer-vision/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Use Test-Time Augmentation to Improve Model Performance for Image Classification](img/86f163e68dad707035baf952d1d6127c.png)

如何使用测试时间扩充来提高图像分类的模型表现
图片由[达维宁](https://www.flickr.com/photos/daveynin/7206430966/)提供，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  测试时间扩充
2.  Keras 试验时间扩充
3.  数据集和基线模型
4.  测试时间扩充示例
5.  如何调整测试时间扩充配置

## 测试时间扩充

数据扩充是一种通常在模型训练期间使用的方法，它使用来自训练数据集的样本的修改副本来扩展训练集。

数据扩充通常使用图像数据来执行，其中训练数据集中的图像副本是通过执行一些图像操作技术来创建的，例如缩放、翻转、移位等等。

人工扩展的训练数据集可以产生更熟练的模型，因为深度学习模型的表现通常会随着训练数据集的大小而不断扩展。此外，训练数据集中图像的修改或增强版本有助于模型以对其位置、光照等不变的方式提取和学习特征。

测试时间扩充，简称 TTA，是数据扩充在测试数据集中的应用。

具体来说，它包括为测试集中的每个图像创建多个增强副本，让模型为每个副本做出预测，然后返回这些预测的集合。

选择增强是为了给模型提供对给定图像进行正确分类的最佳机会，模型必须预测的图像副本数量通常很少，例如少于 10 或 20 个。

通常，执行一个简单的测试时间扩充，如移位、裁剪或图像翻转。

在他们 2015 年发表的题为“用于大规模图像识别的非常深卷积网络”的论文中，作者使用了水平翻转测试时间扩充:

> 我们还通过水平翻转图像来扩充测试集；原始图像和翻转图像的软最大类后验值被平均以获得图像的最终分数。

类似地，在他们 2015 年发表的名为“重新思考计算机视觉的初始架构”的关于初始架构的论文中，谷歌的作者使用了裁剪测试时间扩充，他们称之为多裁剪评估。

## Keras 试验时间扩充

Keras 深度学习库中并没有提供测试时增强，但是可以轻松实现。

[ImageDataGenerator 类](https://keras.io/preprocessing/image/)可用于配置测试时间扩充的选择。例如，下面的数据生成器被配置用于水平翻转图像数据扩充。

```py
# configure image data augmentation
datagen = ImageDataGenerator(horizontal_flip=True)
```

然后，可以对测试数据集中的每个样本分别应用增强。

首先，对于单个图像，单个图像的维度可以从*【行】【列】【通道】*扩展到*【样本】【行】【列】【通道】*，其中样本数为 1。这将图像的数组转换为包含一个图像的样本数组。

```py
# convert image into dataset
samples = expand_dims(image, 0)
```

接下来，可以为样本创建一个迭代器，批量大小可以用来指定要生成的增强图像的数量，例如 10 个。

```py
# prepare iterator
it = datagen.flow(samples, batch_size=10)
```

迭代器然后可以传递给模型的 *predict_generator()* 函数，以便进行预测。具体来说，将生成一批 10 幅增强图像，模型将对每幅图像进行预测。

```py
# make predictions for each augmented image
yhats = model.predict_generator(it, steps=10, verbose=0)
```

最后，可以进行集合预测。在图像多类别分类的情况下，对每个图像进行预测，并且每个预测包含图像属于每个类别的概率。

可以使用[软投票](https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/)进行集成预测，其中在预测中对每个类别的概率求和，并且通过计算求和预测的 [argmax()](https://machinelearningmastery.com/argmax-in-machine-learning/) 来进行类别预测，返回最大求和概率的指数或类别号。

```py
# sum across predictions
summed = numpy.sum(yhats, axis=0)
# argmax across classes
return argmax(summed)
```

我们可以将这些元素绑定到一个函数中，该函数将采用一个已配置的数据生成器、拟合模型和单个图像，并将使用测试时间扩充返回一个类预测(整数)。

```py
# make a prediction using test-time augmentation
def tta_prediction(datagen, model, image, n_examples):
	# convert image into dataset
	samples = expand_dims(image, 0)
	# prepare iterator
	it = datagen.flow(samples, batch_size=n_examples)
	# make predictions for each augmented image
	yhats = model.predict_generator(it, steps=n_examples, verbose=0)
	# sum across predictions
	summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	return argmax(summed)
```

现在，我们知道了如何在 Keras 中使用测试时间扩充进行预测，让我们通过一个例子来演示这种方法。

## 数据集和基线模型

我们可以用一个标准的计算机视觉数据集和一个卷积神经网络来演示测试时间的扩充。

在此之前，我们必须选择一个数据集和一个基线模型。

我们将使用 [CIFAR-10 数据集](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/)，由来自 10 个类别的 60，000 张 32×32 像素彩色照片组成，如青蛙、鸟类、猫、船只等。CIFAR-10 是一个众所周知的数据集，广泛用于机器学习领域的计算机视觉算法基准测试。问题是“*解决了*”通过深度学习卷积神经网络，在测试数据集上的分类准确率达到 96%或 97%以上，在该问题上取得了最佳表现。

我们还将使用卷积神经网络，或 CNN，模型能够在这个问题上获得好的(比随机的更好的)结果，但不是最先进的结果。这将足以证明测试时间扩充所能提供的表现提升。

通过调用 *cifar10.load_data()* 函数，可以通过 Keras API 轻松加载 CIFAR-10 数据集，该函数返回一个元组，其中训练和测试数据集被拆分为输入(图像)和输出(类标签)组件。

```py
# load dataset
(trainX, trainY), (testX, testY) = load_data()
```

在建模之前，最好将像素值从 0-255 范围归一化到 0-1 范围。这确保了输入很小并且接近于零，并且反过来意味着模型的权重将保持很小，从而导致更快和更好的学习。

```py
# normalize pixel values
trainX = trainX.astype('float32') / 255
testX = testX.astype('float32') / 255
```

类标签是整数，在建模之前必须转换成一个热编码。

这可以使用*到 _ classic()*Keras 效用函数来实现。

```py
# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)
```

我们现在准备为这个多类分类问题定义一个模型。

该模型有一个卷积层，包含 32 个具有 3×3 内核的滤波器映射，使用[整流器线性激活](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/)、“*相同的*填充，因此输出与输入大小相同，并且 *He 权重初始化*。接下来是批处理规范化层和最大池层。

尽管滤波器的数量扩充到了 64 个，但这种模式在卷积层、批量范数层和最大池层重复使用。然后，输出在被密集层解释之前被展平，并最终被提供给输出层以进行预测。

```py
# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
```

随机梯度下降的[亚当变异](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)用于寻找模型权重。

使用[分类交叉熵损失函数](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/)，多类分类需要，训练时监控分类准确率。

```py
# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

该模型适用于三个训练时期，并且使用了 128 幅图像的大批量。

```py
# fit model
model.fit(trainX, trainY, epochs=3, batch_size=128)
```

一旦拟合，就在测试数据集上评估模型。

```py
# evaluate model
_, acc = model.evaluate(testX, testY, verbose=0)
print(acc)
```

下面列出了完整的示例，几分钟后就可以在 CPU 上轻松运行。

```py
# baseline cnn model for the cifar10 problem
from keras.datasets.cifar10 import load_data
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
# load dataset
(trainX, trainY), (testX, testY) = load_data()
# normalize pixel values
trainX = trainX.astype('float32') / 255
testX = testX.astype('float32') / 255
# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)
# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainY, epochs=3, batch_size=128)
# evaluate model
_, acc = model.evaluate(testX, testY, verbose=0)
print(acc)
```

运行实例表明，该模型能够很好地快速学习问题。

测试集的准确率达到了 66%左右，这是可以的，但并不可怕。所选择的模型配置已经开始过度调整，并且可以受益于[正则化](https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/)的使用和进一步的调整。然而，这为演示测试时间扩充提供了一个很好的起点。

```py
Epoch 1/3
50000/50000 [==============================] - 64s 1ms/step - loss: 1.2135 - acc: 0.5766
Epoch 2/3
50000/50000 [==============================] - 63s 1ms/step - loss: 0.8498 - acc: 0.7035
Epoch 3/3
50000/50000 [==============================] - 63s 1ms/step - loss: 0.6799 - acc: 0.7632
0.6679
```

神经网络是随机算法，同一个模型多次拟合同一个数据可能会找到不同的一组权重，反过来，每次都会有不同的表现。

为了平衡对模型表现的估计，我们可以更改示例来多次重新运行模型的拟合和评估，并在测试数据集上报告分数分布的均值和标准差。

首先，我们可以定义一个名为 *load_dataset()* 的函数，该函数将加载 CIFAR-10 数据集并为建模做准备。

```py
# load and return the cifar10 dataset ready for modeling
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = load_data()
	# normalize pixel values
	trainX = trainX.astype('float32') / 255
	testX = testX.astype('float32') / 255
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
```

接下来，我们可以定义一个名为 define_model()的函数，该函数将为 CIFAR-10 数据集定义一个模型，准备进行拟合，然后进行评估。

```py
# define the cnn model for the cifar10 dataset
def define_model():
	# define model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dense(10, activation='softmax'))
	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model
```

接下来，定义 *evaluate_model()* 函数，该函数将在训练数据集上拟合定义的模型，然后在测试数据集上对其进行评估，返回运行的估计分类准确率。

```py
# fit and evaluate a defined model
def evaluate_model(model, trainX, trainY, testX, testY):
	# fit model
	model.fit(trainX, trainY, epochs=3, batch_size=128, verbose=0)
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	return acc
```

接下来，我们可以定义一个具有新行为的函数来重复定义、拟合和评估一个新模型，并返回准确度分数的分布。

下面的*repeat _ evaluation()*函数实现了这一点，取数据集，默认为 10 次重复评估。

```py
# repeatedly evaluate model, return distribution of scores
def repeated_evaluation(trainX, trainY, testX, testY, repeats=10):
	scores = list()
	for _ in range(repeats):
		# define model
		model = define_model()
		# fit and evaluate model
		accuracy = evaluate_model(model, trainX, trainY, testX, testY)
		# store score
		scores.append(accuracy)
		print('> %.3f' % accuracy)
	return scores
```

最后，我们可以调用 *load_dataset()* 函数来准备数据集，然后*repeat _ evaluation()*得到一个准确率分数的分布，可以通过报告均值和标准差来总结。

```py
# load dataset
trainX, trainY, testX, testY = load_dataset()
# evaluate model
scores = repeated_evaluation(trainX, trainY, testX, testY)
# summarize result
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

将所有这些联系在一起，下面列出了在 MNIST 数据集上重复评估 CNN 模型的完整代码示例。

```py
# baseline cnn model for the cifar10 problem, repeated evaluation
from numpy import mean
from numpy import std
from keras.datasets.cifar10 import load_data
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization

# load and return the cifar10 dataset ready for modeling
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = load_data()
	# normalize pixel values
	trainX = trainX.astype('float32') / 255
	testX = testX.astype('float32') / 255
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# define the cnn model for the cifar10 dataset
def define_model():
	# define model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dense(10, activation='softmax'))
	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# fit and evaluate a defined model
def evaluate_model(model, trainX, trainY, testX, testY):
	# fit model
	model.fit(trainX, trainY, epochs=3, batch_size=128, verbose=0)
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=0)
	return acc

# repeatedly evaluate model, return distribution of scores
def repeated_evaluation(trainX, trainY, testX, testY, repeats=10):
	scores = list()
	for _ in range(repeats):
		# define model
		model = define_model()
		# fit and evaluate model
		accuracy = evaluate_model(model, trainX, trainY, testX, testY)
		# store score
		scores.append(accuracy)
		print('> %.3f' % accuracy)
	return scores

# load dataset
trainX, trainY, testX, testY = load_dataset()
# evaluate model
scores = repeated_evaluation(trainX, trainY, testX, testY)
# summarize result
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

在现代中央处理器硬件上运行该示例可能需要一段时间，在图形处理器硬件上要快得多。

对于每次重复评估，报告模型的准确性，并报告最终的平均模型表现。

在这种情况下，我们可以看到所选模型配置的平均准确率约为 68%，接近单次模型运行的估计值。

```py
> 0.690
> 0.662
> 0.698
> 0.681
> 0.686
> 0.680
> 0.697
> 0.696
> 0.689
> 0.679
Accuracy: 0.686 (0.010)
```

现在我们已经为标准数据集开发了一个基线模型，让我们看看如何更新这个例子来使用测试时增强。

## 测试时间扩充示例

我们现在可以在 CIFAR-10 上更新我们对 CNN 模型的重复评估，以使用测试时间扩充。

可以直接使用上面关于如何在 Keras 中实现测试时间扩充的部分中开发的 *tta_prediction()* 函数。

```py
# make a prediction using test-time augmentation
def tta_prediction(datagen, model, image, n_examples):
	# convert image into dataset
	samples = expand_dims(image, 0)
	# prepare iterator
	it = datagen.flow(samples, batch_size=n_examples)
	# make predictions for each augmented image
	yhats = model.predict_generator(it, steps=n_examples, verbose=0)
	# sum across predictions
	summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	return argmax(summed)
```

我们可以通过定义 *ImageDataGenerator* 配置来开发一个驱动测试时间扩充的函数，并为测试数据集中的每个图像调用 *tta_prediction()* 。

重要的是，要考虑可能有利于模型适合 CIFAR-10 数据集的图像扩充类型。对照片进行细微修改的增强可能是有用的。这可能包括放大，如缩放、移动和水平翻转。

在这个例子中，我们将只使用水平翻转。

```py
# configure image data augmentation
datagen = ImageDataGenerator(horizontal_flip=True)
```

我们将配置图像生成器来创建七张照片，根据这些照片，将对测试集中的每个示例进行平均预测。

下面的 *tta_evaluate_model()* 函数配置 *ImageDataGenerator* 然后枚举测试数据集，对测试数据集中的每个图像进行类标签预测。然后通过将预测的类别标签与测试数据集中的类别标签进行比较来计算准确度。这要求我们通过使用 *argmax()* 来反转在 *load_dataset()* 中执行的一个热编码。

```py
# evaluate a model on a dataset using test-time augmentation
def tta_evaluate_model(model, testX, testY):
	# configure image data augmentation
	datagen = ImageDataGenerator(horizontal_flip=True)
	# define the number of augmented images to generate per test set image
	n_examples_per_image = 7
	yhats = list()
	for i in range(len(testX)):
		# make augmented prediction
		yhat = tta_prediction(datagen, model, testX[i], n_examples_per_image)
		# store for evaluation
		yhats.append(yhat)
	# calculate accuracy
	testY_labels = argmax(testY, axis=1)
	acc = accuracy_score(testY_labels, yhats)
	return acc
```

然后可以更新 *evaluate_model()* 函数来调用 *tta_evaluate_model()* ，以获得模型准确率分数。

```py
# fit and evaluate a defined model
def evaluate_model(model, trainX, trainY, testX, testY):
	# fit model
	model.fit(trainX, trainY, epochs=3, batch_size=128, verbose=0)
	# evaluate model using tta
	acc = tta_evaluate_model(model, testX, testY)
	return acc
```

将所有这些联系在一起，下面列出了一个完整的例子，它重复评估了美国有线电视新闻网对 CIFAR-10 的测试时间扩充。

```py
# cnn model for the cifar10 problem with test-time augmentation
import numpy
from numpy import argmax
from numpy import mean
from numpy import std
from numpy import expand_dims
from sklearn.metrics import accuracy_score
from keras.datasets.cifar10 import load_data
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization

# load and return the cifar10 dataset ready for modeling
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = load_data()
	# normalize pixel values
	trainX = trainX.astype('float32') / 255
	testX = testX.astype('float32') / 255
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# define the cnn model for the cifar10 dataset
def define_model():
	# define model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dense(10, activation='softmax'))
	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# make a prediction using test-time augmentation
def tta_prediction(datagen, model, image, n_examples):
	# convert image into dataset
	samples = expand_dims(image, 0)
	# prepare iterator
	it = datagen.flow(samples, batch_size=n_examples)
	# make predictions for each augmented image
	yhats = model.predict_generator(it, steps=n_examples, verbose=0)
	# sum across predictions
	summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	return argmax(summed)

# evaluate a model on a dataset using test-time augmentation
def tta_evaluate_model(model, testX, testY):
	# configure image data augmentation
	datagen = ImageDataGenerator(horizontal_flip=True)
	# define the number of augmented images to generate per test set image
	n_examples_per_image = 7
	yhats = list()
	for i in range(len(testX)):
		# make augmented prediction
		yhat = tta_prediction(datagen, model, testX[i], n_examples_per_image)
		# store for evaluation
		yhats.append(yhat)
	# calculate accuracy
	testY_labels = argmax(testY, axis=1)
	acc = accuracy_score(testY_labels, yhats)
	return acc

# fit and evaluate a defined model
def evaluate_model(model, trainX, trainY, testX, testY):
	# fit model
	model.fit(trainX, trainY, epochs=3, batch_size=128, verbose=0)
	# evaluate model using tta
	acc = tta_evaluate_model(model, testX, testY)
	return acc

# repeatedly evaluate model, return distribution of scores
def repeated_evaluation(trainX, trainY, testX, testY, repeats=10):
	scores = list()
	for _ in range(repeats):
		# define model
		model = define_model()
		# fit and evaluate model
		accuracy = evaluate_model(model, trainX, trainY, testX, testY)
		# store score
		scores.append(accuracy)
		print('> %.3f' % accuracy)
	return scores

# load dataset
trainX, trainY, testX, testY = load_dataset()
# evaluate model
scores = repeated_evaluation(trainX, trainY, testX, testY)
# summarize result
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

考虑到重复评估和用于评估每个模型的较慢的手动测试时间扩充，运行该示例可能需要一些时间。

在这种情况下，我们可以看到表现从没有测试时间扩充的测试集的大约 68.6%适度提升到有测试时间扩充的测试集的大约 69.8%的准确性。

```py
> 0.719
> 0.716
> 0.709
> 0.694
> 0.690
> 0.694
> 0.680
> 0.676
> 0.702
> 0.704
Accuracy: 0.698 (0.013)
```

## 如何调整测试时间扩充配置

选择对模型表现提升最大的增强配置可能是一项挑战。

不仅有许多增强方法可供选择，而且每种方法都有配置选项，而且在一组配置选项上调整和评估模型的时间可能会很长，即使适合快速图形处理器。

相反，我建议拟合模型一次并保存到文件中。例如:

```py
# save model
model.save('model.h5')
```

然后从一个单独的文件中加载模型，并在一个小的验证数据集或测试集的小子集上评估不同的测试时增强方案。

例如:

```py
...
# load model
model = load_model('model.h5')
# evaluate model
datagen = ImageDataGenerator(...)
...
```

一旦你找到了一组提升最大的增强选项，你就可以在整个测试集上对模型进行评估，或者尝试一个如上所述的重复评估实验。

测试时间扩充配置不仅包括*图像数据生成器*的选项，还包括生成的图像数量，将根据这些图像对测试集中的每个示例进行平均预测。

我使用这种方法来选择上一节中的测试时间扩充，发现七个例子比三个或五个更好，随机缩放和随机移动似乎降低了模型的准确性。

请记住，如果您也对训练数据集使用图像数据扩充，并且该增强使用一种涉及计算数据集统计数据的像素缩放类型(例如，您称之为 *datagen.fit()* )，那么在测试时增强期间也必须使用相同的统计数据和像素缩放技术。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 应用程序接口

*   [图像预处理 Keras API](https://keras.io/preprocessing/image/) 。
*   [Keras 顺序模型 API](https://keras.io/models/sequential/) 。
*   num py . argmax API

### 文章

*   [用 Keras 扩充测试时间的图像分割](https://www.depends-on-the-definition.com/test-time-augmentation-keras/)
*   [keras_tta，keras python 库的简单测试时间扩充(TTA)](https://github.com/tsterbak/keras_tta)。
*   [tta_wrapper，用于 Keras 模型的测试时间图像扩充(TTA)wrapper](https://github.com/qubvel/tta_wrapper)。

## 摘要

在本教程中，您发现了用于提高图像分类任务模型表现的测试时间扩展。

具体来说，您了解到:

*   测试时间扩充是数据扩充技术的应用，通常在进行预测的训练中使用。
*   如何在 Keras 中从零开始实现测试时间扩充？
*   如何在标准图像分类任务中使用测试时间扩充来提高卷积神经网络模型的表现？

你有什么问题吗？
在下面的评论中提问，我会尽力回答。