# 如何开发 GAN 来生成 MNIST 手写数字

> 原文：<https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/>

最后更新于 2020 年 9 月 1 日

[生成对抗网络](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)，或 GANs，是一种用于训练生成模型的架构，例如用于生成图像的深度卷积神经网络。

开发用于生成图像的 GAN 需要用于分类给定图像是真实的还是生成的鉴别器卷积神经网络模型，以及使用逆卷积层将输入转换为像素值的完整二维图像的生成器模型。

理解 GAN 如何工作以及如何在 GAN 架构中训练深度卷积神经网络模型来生成图像可能是一项挑战。对于初学者来说，一个很好的起点是在计算机视觉领域使用的标准图像数据集上练习开发和使用 GANs，例如 MNIST 手写数字数据集。使用小型且易于理解的数据集意味着可以快速开发和训练更小的模型，从而将重点放在模型架构和图像生成过程本身。

在本教程中，您将发现如何开发一个带有深度卷积网络的生成对抗网络来生成手写数字。

完成本教程后，您将知道:

*   如何定义和训练独立的鉴别器模型来学习真假图像的区别。
*   如何定义独立生成器模型和训练复合生成器和鉴别器模型。
*   如何评估 GAN 的表现并使用最终的独立生成器模型生成新图像。

**用我的新书[Python 生成对抗网络](https://machinelearningmastery.com/generative_adversarial_networks/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Develop a Generative Adversarial Network for an MNIST Handwritten Digits From Scratch in Keras](img/1f988534718c62f371060bcb175f9a6c.png)

如何为 MNIST 手写数字从零开始开发一个生成对抗网络。

## 教程概述

本教程分为七个部分；它们是:

1.  MNIST 手写数字数据集
2.  如何定义和训练鉴别器模型
3.  如何定义和使用生成器模型
4.  如何训练发电机模型
5.  如何评估 GAN 模型的表现
6.  MNIST GAN 完整范例
7.  如何使用最终生成器模型生成图像

## MNIST 手写数字数据集

[MNIST 数据集](https://en.wikipedia.org/wiki/MNIST_database)是一个首字母缩略词，代表修改后的国家标准和技术研究所数据集。

它是一个由 70，000 个 0 到 9 之间的手写单个数字的 28×28 像素小正方形灰度图像组成的数据集。

任务是将手写数字的给定图像分类为代表从 0 到 9 的整数值的 10 个类别之一。

Keras 通过 [mnist.load_dataset()函数](https://keras.io/datasets/#mnist-database-of-handwritten-digits)提供对 MNIST 数据集的访问。它返回两个元组，一个包含标准训练数据集的输入和输出元素，另一个包含标准测试数据集的输入和输出元素。

下面的示例加载数据集并总结加载数据集的形状。

**注意**:第一次加载数据集时，Keras 会自动下载图片的压缩版本，保存在*的主目录下~/。keras/数据集/* 。下载速度很快，因为压缩形式的数据集只有大约 11 兆字节。

```py
# example of loading the mnist dataset
from keras.datasets.mnist import load_data
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)
```

运行该示例将加载数据集，并打印训练的输入和输出组件的形状，以及测试图像的分割。

我们可以看到训练集中有 60K 个例子，测试集中有 10K，每个图像都是 28 乘 28 像素的正方形。

```py
Train (60000, 28, 28) (60000,)
Test (10000, 28, 28) (10000,)
```

图像是黑色背景(0 像素值)的灰度图像，手写数字是白色的(像素值接近 255)。这意味着如果图像被绘制出来，它们大部分是黑色的，中间有一个白色的数字。

我们可以使用 matplotlib 库使用 [imshow()函数](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html)绘制训练数据集中的一些图像，并通过“ *cmap* ”参数将颜色映射指定为“*灰色*，以正确显示像素值。

```py
# plot raw pixel data
pyplot.imshow(trainX[i], cmap='gray')
```

或者，当我们反转颜色并将背景绘制为白色，手写数字绘制为黑色时，图像更容易查看。

它们更容易观看，因为大部分图像现在是白色的，而感兴趣的区域是黑色的。这可以通过使用反向灰度色图来实现，如下所示:

```py
# plot raw pixel data
pyplot.imshow(trainX[i], cmap='gray_r')
```

以下示例将训练数据集中的前 25 幅图像绘制成一个 5 乘 5 的正方形。

```py
# example of loading the mnist dataset
from keras.datasets.mnist import load_data
from matplotlib import pyplot
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# plot images from the training dataset
for i in range(25):
	# define subplot
	pyplot.subplot(5, 5, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap='gray_r')
pyplot.show()
```

运行该示例将从 MNIST 训练数据集创建 25 幅图像的图，排列成 5×5 的正方形。

![Plot of the First 25 Handwritten Digits From the MNIST Dataset.](img/0ec6cc856de978a11dd8f8cea4b30e12.png)

MNIST 数据集中前 25 个手写数字的绘图。

我们将使用训练数据集中的图像作为训练生成对抗网络的基础。

具体来说，生成器模型将学习如何使用鉴别器生成 0 到 9 之间的新的似是而非的手写数字，该鉴别器将尝试区分来自 MNIST 训练数据集的真实图像和生成器模型输出的新图像。

这是一个相对简单的问题，不需要复杂的生成器或鉴别器模型，尽管它确实需要生成灰度输出图像。

## 如何定义和训练鉴别器模型

第一步是定义鉴别器模型。

该模型必须从我们的数据集获取一个样本图像作为输入，并输出一个关于样本是真的还是假的分类预测。

这是一个二分类问题:

*   **输入**:一个通道，28×28 像素大小的图像。
*   **输出**:二进制分类，样本真实(或虚假)的可能性。

鉴别器模型有两个[卷积层](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)，每个卷积层有 64 个滤波器，小内核大小为 3，比正常步长 2 大。该模型没有池化层，在输出层只有一个节点，通过 sigmoid 激活函数来预测输入样本是真的还是假的。该模型被训练成最小化适用于二进制分类的[二进制交叉熵损失函数](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)。

我们将使用一些最佳实践来定义鉴别器模型，例如使用 LeakyReLU 代替 [ReLU](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/) ，使用 [Dropout](https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/) ，以及使用 [Adam 版本的随机梯度下降](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)，学习率为 0.0002，动量为 0.5。

下面的函数*定义 _ 鉴别器()*定义鉴别器模型，并参数化输入图像的大小。

```py
# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```

我们可以使用这个函数来定义鉴别器模型并对其进行总结。

下面列出了完整的示例。

```py
# example of defining the discriminator model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define model
model = define_discriminator()
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
```

运行示例首先总结模型架构，显示每一层的输入和输出。

我们可以看到，进取的 [2×2 步起到了对输入图像](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)进行下采样的作用，先从 28×28 下降到 14×14，再下降到 7×7，然后模型再进行输出预测。

这种模式是通过设计实现的，因为我们不使用池化层，而是使用大跨度来实现类似的下采样效果。我们将在下一节的生成器模型中看到类似的模式，但方向相反。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 14, 14, 64)        640
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 14, 14, 64)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 64)          36928
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 7, 7, 64)          0
_________________________________________________________________
dropout_2 (Dropout)          (None, 7, 7, 64)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 3136)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 3137
=================================================================
Total params: 40,705
Trainable params: 40,705
Non-trainable params: 0
_________________________________________________________________
```

还创建了一个模型图，我们可以看到该模型预期两个输入，并将预测一个输出。

**注**:创建此图假设安装了 pydot 和 graphviz 库。如果这是一个问题，您可以注释掉 *plot_model* 函数的导入语句和对 *plot_model()* 函数的调用。

![Plot of the Discriminator Model in the MNIST GAN](img/d53b033947900027c7aa672cab5a9d34.png)

MNIST GAN 中鉴别器模型的绘制

我们现在可以开始用类标签为 1 的真实例子训练这个模型，并随机生成类标签为 0 的样本。

这些元素的开发将在以后有用，它有助于看到鉴别器只是一个用于二进制分类的普通神经网络模型。

首先，我们需要一个函数来加载和准备真实图像的数据集。

我们将使用 *mnist.load_data()* 函数加载 mnist 数据集，只使用训练数据集的输入部分作为真实图像。

```py
# load mnist dataset
(trainX, _), (_, _) = load_data()
```

图像是像素的 2D 阵列，卷积神经网络期望图像的 3D 阵列作为输入，其中每个图像具有一个或多个通道。

我们必须更新图像，为灰度通道增加一个维度。我们可以使用 [expand_dims() NumPy 函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html)来实现这一点，并指定通道的最终尺寸-最后图像格式。

```py
# expand to 3d, e.g. add channels dimension
X = expand_dims(trainX, axis=-1)
```

最后，我们必须[将像素值](https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/)从[0，255]中的无符号整数范围缩放到[0，1]的归一化范围。

```py
# convert from unsigned ints to floats
X = X.astype('float32')
# scale from [0,255] to [0,1]
X = X / 255.0
```

下面的 *load_real_samples()* 函数实现了这一点。

```py
# load and prepare mnist training images
def load_real_samples():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
	return X
```

该模型将分批更新，特别是收集真实样本和生成样本。在训练中，epoch 被定义为一次通过整个训练数据集。

我们可以系统地枚举训练数据集中的所有样本，这是一种很好的方法，但是通过随机梯度下降进行良好的训练需要在每个时期之前对训练数据集进行洗牌。一种更简单的方法是从训练数据集中选择图像的随机样本。

下面的 *generate_real_samples()* 函数将以训练数据集为参数，选择图像的随机子样本；它还将返回样本的类标签，特别是类标签 1，以指示真实图像。

```py
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y
```

现在，我们需要一个假图像的来源。

我们还没有一个生成器模型，所以相反，我们可以生成由随机像素值组成的图像，具体来说[随机像素值](https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/)在范围【0，1】内，就像我们缩放的真实图像一样。

下面的 *generate_fake_samples()* 函数实现了这一行为，并为 fake 生成随机像素值及其关联的类标签为 0 的图像。

```py
# generate n fake samples with class labels
def generate_fake_samples(n_samples):
	# generate uniform random numbers in [0,1]
	X = rand(28 * 28 * n_samples)
	# reshape into a batch of grayscale images
	X = X.reshape((n_samples, 28, 28, 1))
	# generate 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y
```

最后，我们需要训练鉴别器模型。

这包括重复检索真实图像样本和生成图像样本，并在固定迭代次数内更新模型。

我们现在将忽略时代的概念(例如，通过训练数据集的完整传递)，并为固定数量的批次拟合鉴别器模型。该模型将学会快速辨别真假(随机生成的)图像，因此，在学会完美辨别之前，不需要很多批次。

*train_discriminator()* 函数实现了这一点，使用了 256 幅图像的批处理大小，其中每次迭代有 128 幅是真实的，128 幅是假的。

我们为真实和虚假的例子分别更新鉴别器，以便我们可以在更新之前计算每个样本上模型的准确性。这让我们深入了解了鉴别器模型在一段时间内的表现。

```py
# train the discriminator model
def train_discriminator(model, dataset, n_iter=100, n_batch=256):
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_iter):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		# update discriminator on real samples
		_, real_acc = model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(half_batch)
		# update discriminator on fake samples
		_, fake_acc = model.train_on_batch(X_fake, y_fake)
		# summarize performance
		print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))
```

将所有这些结合在一起，下面列出了在真实和随机生成(假)图像上训练鉴别器模型实例的完整示例。

```py
# example of training the discriminator model on real and random mnist images
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# load and prepare mnist training images
def load_real_samples():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

# generate n fake samples with class labels
def generate_fake_samples(n_samples):
	# generate uniform random numbers in [0,1]
	X = rand(28 * 28 * n_samples)
	# reshape into a batch of grayscale images
	X = X.reshape((n_samples, 28, 28, 1))
	# generate 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# train the discriminator model
def train_discriminator(model, dataset, n_iter=100, n_batch=256):
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_iter):
		# get randomly selected 'real' samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		# update discriminator on real samples
		_, real_acc = model.train_on_batch(X_real, y_real)
		# generate 'fake' examples
		X_fake, y_fake = generate_fake_samples(half_batch)
		# update discriminator on fake samples
		_, fake_acc = model.train_on_batch(X_fake, y_fake)
		# summarize performance
		print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

# define the discriminator model
model = define_discriminator()
# load image data
dataset = load_real_samples()
# fit the model
train_discriminator(model, dataset)
```

运行该示例首先定义模型，加载 MNIST 数据集，然后训练鉴别器模型。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，鉴别器模型学会非常快速地分辨真实和随机生成的 MNIST 图像，大约分 50 批。

```py
...
>96 real=100% fake=100%
>97 real=100% fake=100%
>98 real=100% fake=100%
>99 real=100% fake=100%
>100 real=100% fake=100%
```

既然我们知道了如何定义和训练鉴别器模型，我们需要考虑开发生成器模型。

## 如何定义和使用生成器模型

生成器模型负责创建新的、假的但看似可信的手写数字图像。

它通过从潜在空间中取一个点作为输入并输出一个方形灰度图像来实现这一点。

潜在空间是任意定义的[高斯分布值](https://machinelearningmastery.com/statistical-data-distributions/)的向量空间，例如 100 维。它没有任何意义，但是通过从这个空间中随机抽取点，并在训练期间将它们提供给生成器模型，生成器模型将为潜在点以及潜在空间赋予意义，直到训练结束时，潜在向量空间表示输出空间 MNIST 图像的压缩表示，只有生成器知道如何变成似是而非的 MNIST 图像。

*   **输入**:潜在空间中的点，例如高斯随机数的 100 元素向量。
*   **输出**:像素值在[0，1]的 28×28 像素的二维方形灰度图像。

注意:我们不必使用 100 元素向量作为输入；这是一个整数，被广泛使用，但我希望 10、50 或 500 也可以。

开发一个生成器模型需要我们将 100 维的潜在空间中的一个向量转换成 28×28 或 784 个值的 2D 数组。

有许多方法可以实现这一点，但有一种方法在深度卷积生成对抗网络中被证明是有效的。它涉及两个主要因素。

第一个是密集层，作为第一个隐藏层，它有足够的节点来表示输出图像的低分辨率版本。具体来说，输出图像一半大小(四分之一面积)的图像将是 14×14 或 196 个节点，四分之一大小(八分之一面积)的图像将是 7×7 或 49 个节点。

我们不只是想要一个低分辨率版本的图像；我们想要输入的许多并行版本或解释。这是卷积神经网络中的[模式，其中我们有许多并行滤波器，产生多个并行激活图，称为特征图，对输入有不同的解释。相反，我们想要同样的东西:我们输出的许多并行版本具有不同的学习特性，可以在输出层折叠成最终图像。模型需要空间来发明、创造或生成。](https://machinelearningmastery.com/review-of-architectural-innovations-for-convolutional-neural-networks-for-image-classification/)

因此，第一个隐藏层“密集”需要足够的节点来存储我们输出图像的多个低分辨率版本，例如 128。

```py
# foundation for 7x7 image
model.add(Dense(128 * 7 * 7, input_dim=100))
```

然后，来自这些节点的激活可以被重新整形为类似图像的东西，以传递到卷积层，例如 128 个不同的 7×7 特征图。

```py
model.add(Reshape((7, 7, 128)))
```

下一个主要的架构创新包括将低分辨率图像上采样到图像的更高分辨率版本。

这种上采样过程有两种常见方式，有时称为反卷积。

一种方法是使用一个*upsmampling 2d*层(像一个反向池化层)，然后是一个正常的 *Conv2D* 层。另一种或许更现代的方式是将这两种操作组合成一个单一的层，称为*conv2d 转置*。我们将把后一种方法用于我们的生成器。

*conv2d 转置*层可以配置为(2×2)的步幅，这将使输入要素地图的面积增加四倍(宽度和高度尺寸增加一倍)。使用作为步长因子的内核大小(例如双倍)来避免上采样时可能观察到的棋盘图案也是一种良好的做法。

```py
# upsample to 14x14
model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
```

这可以重复，以达到我们的 28×28 输出图像。

同样，我们将使用默认斜率为 0.2 的 LeakyReLU，作为训练 GAN 模型时的最佳实践。

模型的输出层是一个 *Conv2D* ，带有一个过滤器和 7×7 的内核大小以及[“相同”填充](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)，设计用于创建单个特征图并将其尺寸保持在 28×28 像素。sigmoid 激活用于确保输出值在期望的范围[0，1]内。

下面的 *define_generator()* 函数实现了这一点，并定义了生成器模型。

**注**:发电机模型未编译，未指定损失函数或优化算法。这是因为发电机不是直接训练的。我们将在下一节中了解更多信息。

```py
# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model
```

我们可以总结模型，以帮助更好地理解输入和输出形状。

下面列出了完整的示例。

```py
# example of defining the generator model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model

# define the size of the latent space
latent_dim = 100
# define the generator model
model = define_generator(latent_dim)
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
```

运行该示例总结了模型的层及其输出形状。

我们可以看到，按照设计，第一个隐藏层有 6，272 个参数或 128 * 7 * 7，它们的激活被重新整形为 128 个 7×7 特征图。然后，通过两个*conv2d 转置*层将特征图升级到 28×28 的期望输出形状，直到输出层，在那里输出单个激活图。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 6272)              633472
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 6272)              0
_________________________________________________________________
reshape_1 (Reshape)          (None, 7, 7, 128)         0
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 14, 14, 128)       262272
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 14, 14, 128)       0
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 28, 28, 128)       262272
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 28, 28, 128)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 1)         6273
=================================================================
Total params: 1,164,289
Trainable params: 1,164,289
Non-trainable params: 0
_________________________________________________________________
```

还创建了模型的一个图，我们可以看到模型期望来自潜在空间的 100 元素点作为输入，并将生成图像作为输出。

**注**:创建此图假设安装了 pydot 和 graphviz 库。如果这是一个问题，您可以注释掉 plot_model 函数的导入语句和对 plot_model 函数的调用。

![Plot of the Generator Model in the MNIST GAN](img/6f6fc424348268418741a598dcc7418e.png)

MNIST GAN 发电机模型图

这种模式目前做不了什么。

然而，我们可以演示如何使用它来生成样本。这是一个很有帮助的演示，可以将生成器理解为另一个模型，其中一些元素在以后会很有用。

第一步是在潜在空间中生成新的点。我们可以通过调用 [randn() NumPy 函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html)来实现这一点，该函数用于生成从标准高斯中提取的[随机数数组。](https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/)

然后，随机数的数组可以被重新整形为样本，即 n 行，每行 100 个元素。下面的*生成 _ 潜在 _ 点()*函数实现了这一点，并在潜在空间中生成所需数量的点，这些点可用作生成器模型的输入。

```py
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

接下来，我们可以使用生成的点作为生成器模型的输入来生成新的样本，然后绘制样本。

我们可以更新上一节中的 *generate_fake_samples()* 函数，以生成器模型为参数，使用它来生成所需数量的样本，方法是首先调用*generate _ 潜伏 _points()* 函数来生成潜伏空间中所需数量的点，作为模型的输入。

更新后的 *generate_fake_samples()* 函数如下所示，返回生成的样本和关联的类标签。

```py
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y
```

然后，我们可以像第一部分中的真实 MNIST 示例那样，通过使用反转的灰度色图调用 imshow()函数来绘制生成的样本。

下面列出了使用未经训练的生成器模型生成新 MNIST 图像的完整示例。

```py
# example of defining and using the generator model
from numpy import zeros
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from matplotlib import pyplot

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# size of the latent space
latent_dim = 100
# define the discriminator model
model = define_generator(latent_dim)
# generate samples
n_samples = 25
X, _ = generate_fake_samples(model, latent_dim, n_samples)
# plot the generated samples
for i in range(n_samples):
	# define subplot
	pyplot.subplot(5, 5, 1 + i)
	# turn off axis labels
	pyplot.axis('off')
	# plot single image
	pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
# show the figure
pyplot.show()
```

运行该示例会生成 25 个假 MNIST 图像示例，并在一个 5 乘 5 图像的单个绘图上可视化它们。

由于模型未经训练，生成的图像完全是[0，1]中的随机像素值。

![Example of 25 MNIST Images Output by the Untrained Generator Model](img/97b7dcc0d852e1b38813713120c31ed4.png)

未经训练的生成器模型输出的 25 幅 MNIST 图像示例

既然我们知道了如何定义和使用生成器模型，下一步就是训练模型。

## 如何训练发电机模型

生成器模型中的权重根据鉴别器模型的表现进行更新。

当鉴别器擅长检测假样本时，生成器更新较多，当鉴别器模型在检测假样本时相对较差或混乱时，生成器模型更新较少。

这就定义了这两种模式之间的零和或对立关系。

使用 Keras API 实现这一点可能有很多方法，但最简单的方法可能是创建一个新的模型，将生成器和鉴别器模型结合起来。

具体而言，可以定义新的 GAN 模型，该模型堆叠生成器和鉴别器，使得生成器接收潜在空间中的随机点作为输入，并生成样本，这些样本被直接馈送到鉴别器模型中，进行分类，并且该更大模型的输出可以用于更新生成器的模型权重。

明确地说，我们不是在谈论新的第三个模型，只是一个新的逻辑模型，它使用独立生成器和鉴别器模型中已经定义的层和权重。

只有鉴别器关心的是区分真实和虚假的例子，因此鉴别器模型可以以独立的方式对每个例子进行训练，就像我们在上面的鉴别器模型部分所做的那样。

生成器模型只关心鉴别器在假例子上的表现。因此，当它是 GAN 模型的一部分时，我们将把鉴别器中的所有层标记为不可训练的，这样它们就不能被更新和过度训练在假的例子上。

当通过这个逻辑 GAN 模型训练生成器时，还有一个更重要的变化。我们希望鉴别器认为生成器输出的样本是真实的，而不是伪造的。因此，当生成器被训练为 GAN 模型的一部分时，我们将把生成的样本标记为真实的(类 1)。

***我们为什么要这么做？**T3】*

我们可以想象，鉴别器然后将生成的样本分类为不真实(类别 0)或真实概率低(0.3 或 0.5)。用于更新模型权重的反向传播过程将认为这是一个很大的误差，并将更新模型权重(即，仅生成器中的权重)来纠正这个误差，这又使得生成器更好地生成好的假样本。

让我们把这个具体化。

*   **输入**:潜在空间中的点，例如高斯随机数的 100 元素向量。
*   **输出**:二进制分类，样本真实(或虚假)的可能性。

下面的 *define_gan()* 函数将已经定义的生成器和鉴别器模型作为参数，并创建包含这两个模型的新的逻辑第三模型。鉴别器中的权重被标记为不可训练，这仅影响 GAN 模型看到的权重，而不影响独立鉴别器模型。

然后，GAN 模型使用相同的二元交叉熵损失函数作为鉴别器，并使用学习率为 0.0002、动量为 0.5 的随机梯度下降的高效 Adam 版本，这是在训练深度卷积 GAN 时推荐的。

```py
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
```

使鉴别器不可训练是 Keras API 中的一个聪明的技巧。

可训练属性会在模型编译后影响模型。鉴别器模型是用可训练层编译的，因此当通过调用 *train_on_batch()* 函数更新独立模型时，这些层中的模型权重将被更新。

然后将鉴别器模型标记为不可训练，添加到 GAN 模型中，并进行编译。在该模型中，鉴别器模型的模型权重是不可训练的，并且当通过调用 *train_on_batch()* 函数更新 GAN 模型时，不能改变。可训练属性的这种变化不会影响独立鉴别器模型的训练。

这里的 Keras API 文档中描述了这种行为:

*   [如何“冻结”Keras 层？](https://keras.io/getting-started/faq/#how-can-i-freeze-keras-layers)

下面列出了创建鉴别器、生成器和复合模型的完整示例。

```py
# demonstrate creating the three models in the gan
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# summarize gan model
gan_model.summary()
# plot gan model
plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
```

运行该示例首先会创建复合模型的摘要。

我们可以看到，该模型期望 MNIST 图像作为输入，并预测单个值作为输出。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
sequential_2 (Sequential)    (None, 28, 28, 1)         1164289
_________________________________________________________________
sequential_1 (Sequential)    (None, 1)                 40705
=================================================================
Total params: 1,204,994
Trainable params: 1,164,289
Non-trainable params: 40,705
_________________________________________________________________
```

还创建了模型的图，我们可以看到模型期望潜在空间中的 100 元素点作为输入，并将预测单个输出分类标签。

**注**:创建此图假设安装了 pydot 和 graphviz 库。如果这是一个问题，您可以注释掉 *plot_model* 函数的导入语句和对 *plot_model()* 函数的调用。

![Plot of the Composite Generator and Discriminator Model in the MNIST GAN](img/53244a279886e4f47b19858a54c107f2.png)

MNIST GAN 中复合发生器和鉴别器模型的绘图

训练复合模型包括通过上一节中的*generate _ 潜伏 _points()* 函数在潜伏空间中生成一批有价值的点，以及 *class=1* 标签和调用 *train_on_batch()* 函数。

下面的 *train_gan()* 函数演示了这一点，尽管它非常简单，因为每个时期只有生成器会更新，给鉴别器留下默认的模型权重。

```py
# train the composite model
def train_gan(gan_model, latent_dim, n_epochs=100, n_batch=256):
	# manually enumerate epochs
	for i in range(n_epochs):
		# prepare points in latent space as input for the generator
		x_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		gan_model.train_on_batch(x_gan, y_gan)
```

相反，我们需要的是首先用真样本和假样本更新鉴别器模型，然后通过复合模型更新生成器。

这需要组合以上鉴别器部分定义的 *train_discriminator()* 函数和以上定义的 *train_gan()* 函数中的元素。它还要求我们列举一个时代内的时代和批次。

下面列出了更新鉴别器模型和生成器(通过复合模型)的完整训练函数。

在这个模型训练函数中需要注意一些事情。

首先，一个时期内的批次数量由批次大小划分到训练数据集中的次数来定义。我们的数据集大小为 60K 个样本，因此向下舍入后，每个时期有 234 个批次。

通过 [vstack() NumPy 函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html)将半批假实例和真实实例合并成一批，每批更新一次鉴别器模型。您可以分别用每半批更新鉴别器(对于更复杂的数据集，建议这样做)，但是从长远来看，将样本合并成一批会更快，尤其是在 GPU 硬件上进行训练时。

最后，我们报告每批的损失。保持对批次损失的关注至关重要。其原因是鉴别器丢失的崩溃表明生成器模型已经开始生成鉴别器可以轻松鉴别的垃圾例子。

监控鉴别器损耗，并期望它在该数据集上每批大约 0.5 到 0.8。发电机损耗不太重要，可能会在 0.5 和 2 之间徘徊或更高。一个聪明的程序员甚至可能试图检测鉴别器的崩溃丢失，暂停，然后重新开始训练过程。

```py
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
```

我们几乎拥有为 MNIST 手写数字数据集开发 GAN 所需的一切。

剩下的一个方面是对模型的评估。

## 如何评估 GAN 模型的表现

通常，没有客观的方法来评估 GAN 模型的表现。

我们无法计算生成图像的客观误差分数。在 MNIST 图像的情况下，这可能是可能的，因为图像受到很好的约束，但总的来说，这是不可能的。

相反，图像必须由人工操作员主观评估质量。这意味着，如果不查看生成的图像示例，我们就无法知道何时停止训练。反过来，训练过程的对抗性意味着生成器在每一批之后都在变化，这意味着一旦能够生成足够好的图像，图像的主观质量可能会随着后续的更新而开始变化、提高甚至下降。

有三种方法可以处理这种复杂的训练情况。

1.  定期评估真假图像鉴别器的分类准确率。
2.  定期生成许多图像，并保存到文件中进行主观审查。
3.  定期保存发电机模型。

对于给定的训练时期，可以同时执行所有这三个动作，例如每五个或十个训练时期。结果将是一个保存的生成器模型，对于该模型，我们有一种主观评估其输出质量的方法，并且客观地知道在保存模型时鉴别器被愚弄的程度。

在许多时期，例如数百或数千个时期，训练 GAN 将产生模型的许多快照，这些快照可以被检查，并且可以从中挑选特定的输出和模型供以后使用。

首先，我们可以定义一个名为*summary _ performance()*的函数，该函数将总结鉴别器模型的表现。它通过检索真实 MNIST 图像的样本，以及用生成器模型生成相同数量的假 MNIST 图像，然后在每个样本上评估鉴别器模型的分类准确率并报告这些分数来实现这一点。

```py
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
```

该功能可以从*列车()*功能中基于当前历元号调用，如每 10 个历元调用一次。

```py
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
	...
	# evaluate the model performance, sometimes
	if (i+1) % 10 == 0:
		summarize_performance(i, g_model, d_model, dataset, latent_dim)
```

接下来，我们可以更新*summary _ performance()*函数，以保存模型并创建和保存绘图生成的示例。

通过调用发电机模型上的 *save()* 函数，并根据训练时期号提供唯一的文件名，可以保存发电机模型。

```py
...
# save the generator model tile file
filename = 'generator_model_%03d.h5' % (epoch + 1)
g_model.save(filename)
```

我们可以开发一个函数来创建生成样本的图。

当我们在 100 幅生成的 MNIST 图像上评估鉴别器时，我们可以将所有 100 幅图像绘制成 10×10 的网格。下面的 *save_plot()* 函数实现了这一点，再次使用基于纪元号的唯一文件名保存结果图。

```py
# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()
```

添加了这些内容的更新后的*summary _ performance()*功能如下所示。

```py
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)
```

## MNIST GAN 完整范例

我们现在拥有了在 MNIST 手写数字数据集上训练和评估 GAN 所需的一切。

下面列出了完整的示例。

注意:这个例子可以在一个中央处理器上运行，但可能需要几个小时。该示例可以在图形处理器上运行，例如亚马逊 EC2 p3 实例，并将在几分钟内完成。

有关设置 AWS EC2 实例以运行此代码的帮助，请参见教程:

*   [如何设置亚马逊 AWS EC2 GPUs 训练 Keras 深度学习模型(分步)](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)

```py
# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot

# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load and prepare mnist training images
def load_real_samples():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		# evaluate the model performance, sometimes
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)

# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
```

所选择的配置导致生成模型和判别模型的稳定训练。

每批报告一次模型表现，包括区分模型( *d* )和生成模型( *g* )的损失。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，损失在整个训练过程中保持稳定。

```py
>1, 1/234, d=0.711, g=0.678
>1, 2/234, d=0.703, g=0.698
>1, 3/234, d=0.694, g=0.717
>1, 4/234, d=0.684, g=0.740
>1, 5/234, d=0.679, g=0.757
>1, 6/234, d=0.668, g=0.777
...
>100, 230/234, d=0.690, g=0.710
>100, 231/234, d=0.692, g=0.705
>100, 232/234, d=0.698, g=0.701
>100, 233/234, d=0.697, g=0.688
>100, 234/234, d=0.693, g=0.698
```

每 20 个时期对生成器进行一次评估，结果是 10 次评估、10 幅生成的图像图和 10 个保存的模型。

在这种情况下，我们可以看到准确率随着训练而波动。当查看鉴别器模型的准确度分数与生成的图像一致时，我们可以看到假例子的准确度与图像的主观质量没有很好的关联，但是真实例子的准确度可能有关联。

这是一个粗略的、可能不可靠的 GAN 表现指标，还有损耗。

```py
>Accuracy real: 51%, fake: 78%
>Accuracy real: 30%, fake: 95%
>Accuracy real: 75%, fake: 59%
>Accuracy real: 98%, fake: 11%
>Accuracy real: 27%, fake: 92%
>Accuracy real: 21%, fake: 92%
>Accuracy real: 29%, fake: 96%
>Accuracy real: 4%, fake: 99%
>Accuracy real: 18%, fake: 97%
>Accuracy real: 28%, fake: 89%
```

超出某个点的更多训练并不意味着生成的图像质量更好。

在这种情况下，10 个时期后的结果质量较低，尽管我们可以看到生成器已经学会在背景上生成白色居中的图形(回想一下，我们已经反转了图中的灰度)。

![Plot of 100 GAN Generated MNIST Figures After 10 Epochs](img/dc5299e186ddf116fcbbf44190fa9e2c.png)

10 个时代后 100 个 GAN 生成的 MNIST 图形图

在 20 或 30 个以上的时代之后，模型开始生成非常可信的 MNIST 数字，这表明所选模型配置可能不需要 100 个时代。

![Plot of 100 GAN Generated MNIST Figures After 40 Epochs](img/d25d929fa04e000f7ff5151096729740.png)

40 年代后 100 个 GAN 生成的 MNIST 图形图

100 个纪元后生成的图像差别不大，但我相信我可以在曲线中检测到较少的块状。

![Plot of 100 GAN Generated MNIST Figures After 100 Epochs](img/a6d45c0cf931764db2d9dca1859a2ddb.png)

100 个 GAN 生成的 MNIST 图形在 100 个时代之后的绘图

## 如何使用最终生成器模型生成图像

一旦选择了最终的生成器模型，它就可以独立地用于您的应用程序。

这包括首先从文件中加载模型，然后使用它生成图像。每个图像的生成需要潜在空间中的一个点作为输入。

下面列出了加载保存的模型并生成图像的完整示例。在这种情况下，我们将使用在 100 个训练时期之后保存的模型，但是在 40 或 50 个时期之后保存的模型也同样有效。

```py
# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()

# load model
model = load_model('generator_model_100.h5')
# generate images
latent_points = generate_latent_points(100, 25)
# generate images
X = model.predict(latent_points)
# plot the result
save_plot(X, 5)
```

运行该示例首先加载模型，对潜在空间中的 25 个随机点进行采样，生成 25 幅图像，然后将结果绘制为一幅图像。

我们可以看到大多数图像都是可信的，或者是手写数字的可信片段。

![Example of 25 GAN Generated MNIST Handwritten Images](img/e695b9562587b6e7f76ad2f428420d83.png)

25 幅 GAN 生成的 MNIST 手写图像示例

潜在空间现在定义了 MNIST 手写数字的压缩表示。

你可以尝试在这个空间中生成不同的点，看看它们生成什么类型的数字。

下面的示例使用所有 0.0 值的向量生成一个手写数字。

```py
# example of generating an image for a specific point in the latent space
from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
# load model
model = load_model('generator_model_100.h5')
# all 0s
vector = asarray([[0.0 for _ in range(100)]])
# generate image
X = model.predict(vector)
# plot the result
pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
pyplot.show()
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，一个全零的向量会得到一个手写的 9 或者 8。然后，您可以尝试在空间中导航，看看是否可以生成一系列相似但不同的手写数字。

![Example of a GAN Generated MNIST Handwritten Digit for a Vector of Zeros](img/d78de9e021ec19ad85366b8ed68fe0d2.png)

零向量的 GAN 生成的 MNIST 手写数字示例

## 扩展ˌ扩张

本节列出了一些您可能希望探索的扩展教程的想法。

*   **TanH 激活和缩放**。更新示例以使用生成器中的 tanh 激活函数，并将所有像素值缩放至范围[-1，1]。
*   **改变潜伏空间**。更新示例以使用更大或更小的潜在空间，并比较结果质量和训练速度。
*   **批量归一化**。更新鉴别器和/或生成器，以利用批处理标准化，推荐用于 DCGAN 模型。
*   **标签平滑**。更新示例以在训练鉴别器时使用单侧标签平滑，特别是将真实示例的目标标签从 1.0 更改为 0.9，并查看对图像质量和训练速度的影响。
*   **型号配置**。更新模型配置，以使用更深或更浅的鉴别器和/或生成器模型，也许可以在生成器中实验 UpSampling2D 层。

如果你探索这些扩展，我很想知道。
在下面的评论中发表你的发现。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   第二十章。深度生成模型，[深度学习](https://amzn.to/2YuwVjL)，2016。
*   第八章。生成式深度学习，[Python 深度学习](https://amzn.to/2U2bHuP)，2017。

### 报纸

*   [生成对抗网络](https://arxiv.org/abs/1406.2661)，2014。
*   [教程:生成对抗网络，NIPS](https://arxiv.org/abs/1701.00160) ，2016。
*   [深度卷积生成对抗网络的无监督表示学习](https://arxiv.org/abs/1511.06434)，2015

### 应用程序接口

*   [硬数据集 API](https://keras.io/datasets/)
*   [Keras 顺序模型 API](https://keras.io/models/sequential/)
*   [Keras 卷积层应用编程接口](https://keras.io/layers/convolutional/)
*   [如何“冻结”Keras 层？](https://keras.io/getting-started/faq/#how-can-i-freeze-keras-layers)
*   [MatplotLib API](https://matplotlib.org/api/)
*   [NumPy 随机采样(numpy.random) API](https://docs.scipy.org/doc/numpy/reference/routines.random.html)
*   [NumPy 数组操作例程](https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)

### 邮件

*   [如何在 Python 中生成随机数](https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/)

### 文章

*   [MNIST 数据集，维基百科](https://en.wikipedia.org/wiki/MNIST_database)。
*   [GitHub](https://github.com/kroosen/GAN-in-keras-on-mnist)千里甘项目。
*   [Keras-GAN 项目，GitHub](https://github.com/eriklindernoren/Keras-GAN) 。
*   [喀尔巴阡山-MNIST-GAN 项目，GitHub](https://github.com/Zackory/Keras-MNIST-GAN) 。

## 摘要

在本教程中，您发现了如何开发一个带有深度卷积网络的生成对抗网络来生成手写数字。

具体来说，您了解到:

*   如何定义和训练独立的鉴别器模型来学习真假图像的区别。
*   如何定义独立生成器模型和训练复合生成器和鉴别器模型。
*   如何评估 GAN 的表现并使用最终的独立生成器模型生成新图像。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。