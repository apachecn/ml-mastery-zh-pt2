# 如何为 MNIST 手写数字分类开发 CNN

> 原文：<https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/>

最后更新于 2021 年 11 月 14 日

#### 如何从头开发卷积神经网络用于 MNIST 手写数字分类？

MNIST 手写数字分类问题是用于计算机视觉和深度学习的标准数据集。

虽然数据集得到了有效解决，但它可以作为学习和实践如何从零开始开发、评估和使用卷积深度学习神经网络进行图像分类的基础。这包括如何开发一个健壮的测试工具来评估模型的表现，如何探索模型的改进，以及如何保存模型并在以后加载它来对新数据进行预测。

在本教程中，您将发现如何从零开始开发用于手写数字分类的卷积神经网络。

完成本教程后，您将知道:

*   如何开发一个测试工具来开发一个健壮的模型评估，并为分类任务建立一个表现基线。
*   如何探索基线模型的扩展，以提高学习和模型能力。
*   如何开发最终模型，评估最终模型的表现，并使用它对新图像进行预测。

**用我的新书[计算机视觉深度学习](https://machinelearningmastery.com/deep-learning-for-computer-vision/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2019 年 12 月更新**:更新了 TensorFlow 2.0 和 Keras 2.3 的示例。
*   **2020 年 1 月更新**:修复了在交叉验证循环之外定义模型的 bug。
*   **2021 年 11 月更新**:更新使用 Tensorflow 2.6

![How to Develop a Convolutional Neural Network From Scratch for MNIST Handwritten Digit Classification](img/491ed03da56a655d483c2ce1d444227e.png)

如何从零开始开发卷积神经网络用于 MNIST 手写数字分类
图片由[理查德·阿拉维](https://www.flickr.com/photos/geographyalltheway_photos/2918451763/)提供，版权所有。

## 教程概述

本教程分为五个部分；它们是:

1.  MNIST 手写数字类别数据集
2.  模型评估方法
3.  如何开发基线模型
4.  如何开发改进的模型
5.  如何最终确定模型并做出预测

### 发展环境

本教程假设您正在使用独立的 Keras，它运行在 Python 3 的 TensorFlow 之上。如果您需要设置开发环境的帮助，请参阅本教程:

*   [如何用 Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 设置机器学习的 Python 环境

## MNIST 手写数字类别数据集

[MNIST 数据集](https://en.wikipedia.org/wiki/MNIST_database)是一个首字母缩略词，代表修改后的国家标准和技术研究所数据集。

它是一个由 60，000 个 0 到 9 之间的手写单个数字的 28×28 像素小正方形灰度图像组成的数据集。

任务是将手写数字的给定图像分类为代表从 0 到 9 的整数值的 10 个类别之一。

这是一个被广泛使用和深入理解的数据集，并且在很大程度上是“*解决了*”表现最好的模型是深度学习卷积神经网络，其分类准确率达到 99%以上，在等待测试数据集上的错误率在 0.4%到 0.2%之间。

以下示例使用 Keras API 加载 MNIST 数据集，并创建训练数据集中前九幅图像的图。

```py
# example of loading the mnist dataset
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()
```

运行该示例将加载 MNIST 训练和测试数据集，并打印它们的形状。

我们可以看到训练数据集中有 6 万个例子，测试数据集中有 1 万个例子，图像确实是 28×28 像素的正方形。

```py
Train: X=(60000, 28, 28), y=(60000,)
Test: X=(10000, 28, 28), y=(10000,)
```

还创建了数据集中前九幅图像的图，显示了要分类的图像的自然手写特性。

![Plot of a Subset of Images From the MNIST Dataset](img/3d1188cc32173d1f20d352f24a0c23e4.png)

从 MNIST 数据集绘制图像子集

## 模型评估方法

尽管有效地解决了 MNIST 数据集，但是它可以成为开发和实践使用卷积神经网络解决图像分类任务的方法的有用的起点。

我们可以从零开始开发一个新的模型，而不是回顾数据集上表现良好的模型的文献。

数据集已经有了我们可以使用的定义良好的训练和测试数据集。

为了估计给定训练运行的模型表现，我们可以进一步将训练集拆分为训练和验证数据集。然后，可以绘制每次运行的训练和验证数据集的表现，以提供学习曲线和对模型学习问题的深入了解。

Keras API 通过在训练模型时为 *model.fit()* 函数指定“ *validation_data* ”参数来支持这一点，该函数将返回一个对象，该对象描述所选损失的模型表现和每个训练时期的度量。

```py
# record model performance on a validation dataset during training
history = model.fit(..., validation_data=(valX, valY))
```

为了估计一个模型在一般问题上的表现，我们可以使用 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)，也许是五倍交叉验证。这将给出关于训练和测试数据集的差异以及学习算法的随机性质的模型方差的一些说明。在给定标准差的情况下，模型的表现可以作为 k 倍的平均表现，如果需要，可以用来估计置信区间。

我们可以使用 Sklearn API 中的 [KFold 类](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.KFold.html)来实现给定神经网络模型的 K 折交叉验证评估。有许多方法可以实现这一点，尽管我们可以选择一种灵活的方法，其中 *KFold* 类仅用于指定用于每个 spit 的行索引。

```py
# example of k-fold cv for a neural net
data = ...
# prepare cross validation
kfold = KFold(5, shuffle=True, random_state=1)
# enumerate splits
for train_ix, test_ix in kfold.split(data):
        model = ...
	...
```

我们将保留实际的测试数据集，并将其用作最终模型的评估。

## 如何开发基线模型

第一步是开发一个基线模型。

这一点至关重要，因为它既涉及到为测试工具开发基础设施，以便我们设计的任何模型都可以在数据集上进行评估，又建立了问题模型表现的基线，通过该基线可以比较所有的改进。

测试线束的设计是模块化的，我们可以为每个部件开发单独的功能。如果我们愿意的话，这允许测试装具的给定方面独立于其他方面进行修改或交互改变。

我们可以用五个关键元素来开发这个测试工具。它们是数据集的加载、数据集的准备、模型的定义、模型的评估和结果的呈现。

### 加载数据集

我们知道一些关于数据集的事情。

例如，我们知道图像都是预对齐的(例如，每个图像只包含一个手绘数字)，图像都具有相同的 28×28 像素的正方形大小，并且图像是灰度级的。

因此，我们可以加载图像并重塑数据阵列，使其具有单一颜色通道。

```py
# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
```

我们还知道有 10 个类，类被表示为唯一的整数。

因此，我们可以对每个样本的类元素使用一个热编码，将整数转换为一个 10 元素的二进制向量，类值的索引为 1，所有其他类的索引为 0。我们可以通过*到 _ classic()*效用函数来实现。

```py
# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)
```

*load_dataset()* 函数实现了这些行为，可以用来加载数据集。

```py
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
```

### 准备像素数据

我们知道，数据集中每个图像的像素值都是介于黑白或 0 到 255 之间的无符号整数。

我们不知道缩放用于建模的像素值的最佳方式，但是我们知道需要一些缩放。

一个好的起点是[归一化灰度图像的像素值](https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/)，例如将它们重新缩放到范围[0，1]。这包括首先将数据类型从无符号整数转换为浮点数，然后将像素值除以最大值。

```py
# convert from integers to floats
train_norm = train.astype('float32')
test_norm = test.astype('float32')
# normalize to range 0-1
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0
```

下面的 *prep_pixels()* 函数实现了这些行为，并提供了需要缩放的训练和测试数据集的像素值。

```py
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
```

在任何建模之前，必须调用该函数来准备像素值。

### 定义模型

接下来，我们需要为这个问题定义一个基线卷积神经网络模型。

该模型有两个主要方面:由卷积层和池化层组成的特征提取前端，以及进行预测的分类器后端。

对于卷积前端，我们可以从单个[卷积层](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)开始，该卷积层具有小的滤波器大小(3，3)和适度数量的滤波器(32)，然后是[最大池层](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)。过滤图然后可以被展平以向分类器提供特征。

假设问题是一个多类分类任务，我们知道我们将需要一个具有 10 个节点的输出层，以便预测属于 10 个类中每一个的图像的概率分布。这也需要使用 softmax 激活功能。在特征提取器和输出层之间，我们可以添加一个密集层来解释特征，在本例中有 100 个节点。

所有层将使用 [ReLU 激活功能](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)和 he 权重初始化方案，两者都是最佳实践。

我们将对随机梯度下降优化器使用保守配置，其[学习率](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)为 0.01，动量为 0.9。[分类交叉熵](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)损失函数将被优化，适用于多类分类，我们将监控分类准确率度量，这是适当的，因为我们在 10 类中的每一类都有相同数量的例子。

下面的 *define_model()* 函数将定义并返回这个模型。

```py
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
```

### 评估模型

模型定义后，我们需要对其进行评估。

模型将使用[五重交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)进行评估。选择 *k=5* 的值是为了提供重复评估的基线，并且不会大到需要很长的运行时间。每个测试集将占训练数据集的 20%，或者大约 12，000 个例子，接近于这个问题的实际测试集的大小。

训练数据集在被分割之前被混洗，并且每次都执行样本混洗，因此我们评估的任何模型将在每个文件夹中具有相同的训练和测试数据集，从而在模型之间提供苹果对苹果的比较。

我们将为一个适度的 10 个训练时期训练基线模型，默认批量为 32 个例子。每个折叠的测试集将用于在训练运行的每个时期评估模型，以便我们以后可以创建学习曲线，并在运行结束时评估模型的表现。因此，我们将跟踪每次运行的结果历史，以及折叠的分类准确率。

下面的 *evaluate_model()* 函数实现了这些行为，将训练数据集作为参数，并返回一个准确性分数和训练历史的列表，稍后可以对其进行总结。

```py
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories
```

### 呈现结果

一旦评估了模型，我们就可以展示结果了。

有两个关键方面需要介绍:训练期间模型学习行为的诊断和模型表现的估计。这些可以使用单独的函数来实现。

首先，诊断包括创建一个线图，显示在 k 折叠交叉验证的每个折叠期间，模型在列车和测试集上的表现。这些图对于了解模型是过拟合、欠拟合还是非常适合数据集很有价值。

我们将创建一个有两个支线剧情的单一人物，一个是损失，一个是准确性。蓝色线将指示训练数据集上的模型表现，橙色线将指示等待测试数据集上的表现。下面的*summary _ diagnostics()*函数在给定收集的训练历史的情况下创建并显示该图。

```py
# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()
```

接下来，可以通过计算平均值和标准偏差来总结每次折叠期间收集的分类准确度分数。这提供了在该数据集上训练的模型的平均预期表现的估计，以及平均值的平均方差的估计。我们还将通过创建和显示一个方框和触须图来总结分数的分布。

下面的*summary _ performance()*函数为模型评估期间收集的给定分数列表实现了这一点。

```py
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()
```

### 完整示例

我们需要一个驱动测试线束的功能。

这包括调用所有的定义函数。

```py
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)
```

我们现在拥有我们需要的一切；下面列出了 MNIST 数据集上基线卷积神经网络模型的完整代码示例。

```py
# baseline cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()
```

运行该示例会打印交叉验证过程中每个折叠的分类准确率。这有助于了解模型评估正在进行。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

我们可以看到模型达到完美技能的两种情况，以及达到低于 98%准确率的一种情况。这些都是好结果。

```py
> 98.550
> 98.600
> 98.642
> 98.850
> 98.742
```

接下来，显示了一个诊断图，提供了对模型在每个折叠中的学习行为的洞察。

在这种情况下，我们可以看到该模型通常实现了良好的拟合，训练和测试学习曲线收敛。没有明显的过度或不足的迹象。

![Loss and Accuracy Learning Curves for the Baseline Model During k-Fold Cross-Validation](img/4360bacc237947f6e33995a0c66ad01c.png)

K 折交叉验证期间基线模型的损失和准确率学习曲线

接下来，计算模型表现的概要。

我们可以看到，在这种情况下，模型的估计技能约为 98.6%，这是合理的。

```py
Accuracy: mean=98.677 std=0.107, n=5
```

最后，创建一个方框和触须图来总结准确度分数的分布。

![Box and Whisker Plot of Accuracy Scores for the Baseline Model Evaluated Using k-Fold Cross-Validation](img/27f7de093a3c73d045f5d04b602986d6.png)

使用 k 倍交叉验证评估的基线模型准确度分数的方框图和须图

我们现在有了一个健壮的测试工具和一个表现良好的基线模型。

## 如何开发改进的模型

我们有很多方法可以探索基线模型的改进。

我们将关注模型配置中经常导致改进的领域，即所谓的低挂果实。第一个是学习算法的改变，第二个是模型深度的增加。

### 学习的改进

学习算法有许多方面可以探索改进。

也许最大的杠杆作用点是学习率，例如评估学习率的较小或较大值可能产生的影响，以及在培训期间改变学习率的时间表。

另一种可以快速加速模型学习并导致表现大幅提升的方法是批处理规范化。我们将评估批量标准化对基线模型的影响。

[批量归一化](https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/)可以在卷积层和全连通层之后使用。它具有改变层的输出分布的效果，特别是通过标准化输出。这具有稳定和加速学习过程的效果。

我们可以更新模型定义，以便在基线模型的卷积层和密集层的激活函数之后使用批处理规范化。带批量规格化的 *define_model()* 函数的更新版本如下。

```py
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
```

下面提供了包含此更改的完整代码列表。

```py
# cnn model with batch normalization for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()
```

再次运行该示例会报告交叉验证过程中每个折叠的模型表现。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

与交叉验证折叠中的基线相比，我们可能会看到模型表现的小幅下降。

```py
> 98.475
> 98.608
> 98.683
> 98.783
> 98.667
```

创建学习曲线的图，在这种情况下显示学习速度(相对于时代的改进)似乎与基线模型没有不同。

这些图表明，批量标准化，至少在这种情况下实现，不会带来任何好处。

![Loss and Accuracy Learning Curves for the BatchNormalization Model During k-Fold Cross-Validation](img/75857e8c73822f4e6cf4b8d74e102333.png)

K 折交叉验证中批处理模型的损失和准确率学习曲线

接下来，给出了模型的估计表现，显示了模型的平均准确率略有下降的表现:与基线模型的 98.677 相比，为 98.643。

```py
Accuracy: mean=98.643 std=0.101, n=5
```

![Box and Whisker Plot of Accuracy Scores for the BatchNormalization Model Evaluated Using k-Fold Cross-Validation](img/a2991522a4eadf02d0303403d28d4092.png)

使用 k 倍交叉验证评估的 BatchNormalization 模型的准确度分数的方框图和须图

### 模型深度增加

有许多方法可以改变模型配置，以探索对基线模型的改进。

两种常见的方法涉及改变模型的特征提取部分的[容量](https://machinelearningmastery.com/how-to-control-neural-network-model-capacity-with-nodes-and-layers/)或者改变模型的分类器部分的容量或功能。也许最大的影响点是对特征提取器的改变。

我们可以增加模型的特征提取器部分的深度，遵循[类似 VGG 的模式](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)，用相同大小的滤波器增加更多的卷积和池化层，同时增加滤波器的数量。在这种情况下，我们将添加一个双卷积层，每个层有 64 个滤波器，然后是另一个最大池层。

带有此更改的 *define_model()* 函数的更新版本如下所示。

```py
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
```

为了完整起见，下面提供了整个代码列表，包括这一更改。

```py
# deeper cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()
```

运行该示例会报告交叉验证过程中每个折叠的模型表现。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

每倍得分可能表明比基线有所改善。

```py
> 99.058
> 99.042
> 98.883
> 99.192
> 99.133
```

创建了一个学习曲线图，在这种情况下显示模型仍然很好地适合问题，没有明显的过拟合迹象。这些情节甚至表明，进一步的训练时期可能会有所帮助。

![Loss and Accuracy Learning Curves for the Deeper Model During k-Fold Cross-Validation](img/9b8e9e3ca6080806b4bb8d5d994f7421.png)

K 折交叉验证期间更深模型的损失和准确率学习曲线

接下来，给出了模型的估计表现，与基线相比，表现略有提高，从 98.677 提高到 99.062，标准偏差也略有下降。

```py
Accuracy: mean=99.062 std=0.104, n=5
```

![Box and Whisker Plot of Accuracy Scores for the Deeper Model Evaluated Using k-Fold Cross-Validation](img/372f0754b0a3a4240cbbf35fd11592af.png)

使用 k 倍交叉验证评估的更深模型的准确度分数的方框图和须图

## 如何最终确定模型并做出预测

只要我们有想法，有时间和资源来测试，模型改进的过程可能会持续很长时间。

在某些时候，必须选择并采用最终的模型配置。在这种情况下，我们将选择更深的模型作为最终模型。

首先，我们将最终确定我们的模型，但是在整个训练数据集上拟合一个模型，并将该模型保存到文件中供以后使用。然后，我们将加载模型，并在等待测试数据集上评估其表现，以了解所选模型在实践中的实际表现。最后，我们将使用保存的模型对单个图像进行预测。

### 保存最终模型

最终模型通常适合所有可用数据，例如所有训练和测试数据集的组合。

在本教程中，我们有意保留一个测试数据集，以便我们可以估计最终模型的表现，这在实践中可能是一个好主意。因此，我们将只在训练数据集上拟合我们的模型。

```py
# fit model
model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
```

一旦合适，我们可以通过调用模型上的 *save()* 函数将最终模型保存到一个 H5 文件中，并传入选择的文件名。

```py
# save model
model.save('final_model.h5')
```

注意，保存和加载 Keras 模型需要在您的工作站上安装 [h5py 库](https://www.h5py.org/)。

下面列出了在训练数据集上拟合最终深度模型并将其保存到文件中的完整示例。

```py
# save the final model to file
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# fit model
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
	# save model
	model.save('final_model.h5')

# entry point, run the test harness
run_test_harness()
```

运行此示例后，您将在当前工作目录中拥有一个名为“ *final_model.h5* ”的 1.2 兆字节文件。

### 评估最终模型

我们现在可以加载最终模型，并在等待测试数据集上对其进行评估。

如果我们有兴趣向项目涉众展示所选模型的表现，这是我们可以做的事情。

可通过 *load_model()* 功能加载模型。

下面列出了加载保存的模型并在测试数据集上对其进行评估的完整示例。

```py
# evaluate the deep model on the test dataset
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# load model
	model = load_model('final_model.h5')
	# evaluate model on test dataset
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))

# entry point, run the test harness
run_test_harness()
```

运行该示例将加载保存的模型，并在暂挂测试数据集上评估该模型。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

计算并打印测试数据集中模型的分类准确率。在这种情况下，我们可以看到该模型达到了 99.090%的准确率，或者仅仅不到 1%，这一点也不差，并且相当接近估计的 99.753%，标准偏差约为 0.5%(例如 99%的分数)。

```py
> 99.090
```

### 作出预测

我们可以使用保存的模型对新图像进行预测。

该模型假设新图像是灰度级的，它们已经对齐，使得一个图像包含一个居中的手写数字，并且图像的大小是 28×28 像素的正方形。

下面是从 MNIST 测试数据集中提取的图像。您可以将其保存在当前工作目录中，文件名为“ *sample_image.png* ”。

![Sample Handwritten Digit](img/c9c275627f70eda1420b6d853be22103.png)

示例手写数字

*   [下载样本图像(sample_image.png)](https://machinelearningmastery.com/wp-content/uploads/2019/02/sample_image.png)

我们将假设这是一个全新的、看不见的图像，以所需的方式准备，并看看我们如何使用保存的模型来预测图像所代表的整数(例如，我们期望“ *7* ”)。

首先，我们可以加载图像，强制为灰度格式，并强制大小为 28×28 像素。然后，可以调整加载图像的大小，使其具有单个通道，并表示数据集中的单个样本。 *load_image()* 函数实现了这一点，并将返回已加载的准备分类的图像。

重要的是，像素值的准备方式与在拟合最终模型时为训练数据集准备像素值的方式相同，在这种情况下，最终模型是归一化的。

```py
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
```

接下来，我们可以像上一节一样加载模型，调用 *predict()* 函数得到预测得分，然后使用 *argmax()* 得到图像所代表的数字。

```py
# predict the class
predict_value = model.predict(img)
digit = argmax(predict_value)
```

下面列出了完整的示例。

```py
# make a prediction for a new image.
from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example():
	# load the image
	img = load_image('sample_image.png')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print(digit)

# entry point, run the example
run_example()
```

运行该示例首先加载和准备图像，加载模型，然后正确预测加载的图像代表数字“ *7* ”。

```py
7
```

## 扩展ˌ扩张

本节列出了一些您可能希望探索的扩展教程的想法。

*   **调整像素缩放**。探索与基线模型相比，替代像素缩放方法如何影响模型表现，包括居中和标准化。
*   **调整学习率**。探究与基线模型相比，不同的学习率如何影响模型表现，例如 0.001 和 0.0001。
*   **调整模型深度**。探索与基线模型相比，向模型中添加更多层如何影响模型表现，例如模型的分类器部分中的另一个卷积和池层块或另一个密集层。

如果你探索这些扩展，我很想知道。
在下面的评论中发表你的发现。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 蜜蜂

*   [硬数据集 API](https://keras.io/datasets/)
*   [Keras 数据集代码](https://github.com/keras-team/keras/tree/master/keras/datasets)
*   [sklearn.model_selection。KFold 原料药](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

### 文章

*   [MNIST 数据库，维基百科。](https://en.wikipedia.org/wiki/MNIST_database)
*   [类别数据集结果，这个图像的类别是什么？](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)

## 摘要

在本教程中，您发现了如何从零开始开发用于手写数字分类的卷积神经网络。

具体来说，您了解到:

*   如何开发一个测试工具来开发一个健壮的模型评估，并为分类任务建立一个表现基线。
*   如何探索基线模型的扩展，以提高学习和模型能力。
*   如何开发最终模型，评估最终模型的表现，并使用它对新图像进行预测。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。