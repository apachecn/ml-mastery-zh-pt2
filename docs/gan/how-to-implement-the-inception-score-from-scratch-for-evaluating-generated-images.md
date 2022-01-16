# 如何实现评估 GANs 的初始得分

> 原文：<https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/>

最后更新于 2019 年 10 月 11 日

[生成对抗网络](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)，简称 GANs，是一种深度学习神经网络架构，用于训练生成合成图像的生成器模型。

生成模型的一个问题是没有客观的方法来评估生成图像的质量。

因此，通常在模型训练过程中周期性地生成和保存图像，并使用对所生成图像的主观人类评估，以便评估所生成图像的质量并选择最终的生成器模型。

已经进行了许多尝试来建立生成的图像质量的客观度量。一个早期的、被广泛采用的生成图像客观评估方法的例子是初始得分。

在本教程中，您将发现评估生成图像质量的初始得分。

完成本教程后，您将知道:

*   如何计算初始得分以及它所衡量的东西背后的直觉。
*   如何用 NumPy 和 Keras 深度学习库在 Python 中实现初始得分。
*   如何计算小型图像的初始得分，例如 CIFAR-10 数据集中的图像。

**用我的新书[Python 生成对抗网络](https://machinelearningmastery.com/generative_adversarial_networks/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **更新 2019 年 10 月**:更新了均等分配示例的初始得分小 bug。

![How to Implement the Inception Score (IS) From Scratch for Evaluating Generated Images](img/b023678cee5096c58161e6368455a93c.png)

如何从零开始实现初始评分以评估生成的图像
照片由[阿尔弗雷多·阿法托](https://www.flickr.com/photos/alaffat/34963676170/)拍摄，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  《Inception》的评分是多少？
2.  如何计算初始得分
3.  如何用 NumPy 实现初始评分
4.  如何用 Keras 实现初始评分
5.  《Inception》评分的问题

## 《Inception》的评分是多少？

初始得分，简称 IS，是一个客观的度量标准，用于评估生成图像的质量，特别是由生成对抗网络模型输出的合成图像。

初始得分是由蒂姆·萨利曼(Tim Salimans)等人在 2016 年发表的题为“T2 训练 GANs 的改进技术”的论文中提出的

在论文中，作者使用了一个众包平台( [Amazon Mechanical Turk](https://www.mturk.com/) )来评估大量 GAN 生成的图像。他们开发了初始得分，试图消除人类对图像的主观评估。

作者发现他们的分数与主观评估有很好的相关性。

> 作为人类注释器的替代，我们提出了一种自动评估样本的方法，我们发现该方法与人类评估有很好的相关性。

——[训练 GANs 的改进技术](https://arxiv.org/abs/1606.03498)，2016 年。

初始得分包括使用用于图像分类的预先训练的深度学习神经网络模型来对生成的图像进行分类。具体来说，[克里斯蒂安·塞格迪(Christian Szegedy)](https://ai.google/research/people/ChristianSzegedy)等人在 2015 年发表的题为“[重新思考计算机视觉的初始架构](https://arxiv.org/abs/1512.00567)的论文中描述了[初始 v3 模型](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)对初始模型的依赖赋予了初始得分这个名字。

使用该模型对大量生成的图像进行分类。具体地，预测图像属于每个类别的概率。这些预测然后被总结成初始得分。

该评分旨在捕捉生成的图像集合的两个属性:

*   **图像质量**。图像看起来像特定的对象吗？
*   **图像多样性**。是否生成了广泛的对象？

初始得分的最低值为 1.0，最高值为分类模型支持的类别数；在这种情况下，Inception v3 模型支持 [ILSVRC 2012 数据集](https://machinelearningmastery.com/introduction-to-the-imagenet-large-scale-visual-recognition-challenge-ilsvrc/)的 1000 个类，因此，该数据集上的最高 Inception 分数为 1000。

[CIFAR-10 数据集](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/)是分为 10 类对象的 50，000 幅图像的集合。介绍初始阶段的原始论文在真实的 CIFAR-10 训练数据集上计算了分数，获得了 11.24 +/- 0.12 的结果。

使用他们论文中介绍的 GAN 模型，他们在为该数据集生成合成图像时获得了 8.09+/-0.07 的初始得分。

## 如何计算初始得分

初始得分是通过首先使用预先训练的初始 v3 模型来预测每个生成图像的类别概率来计算的。

这些是条件概率，例如以生成的图像为条件的类别标签。与所有其他类别相比，被强分类为一个类别的图像表示高质量。因此，集合中所有生成图像的条件概率应该具有[低熵](https://en.wikipedia.org/wiki/Entropy_(information_theory))。

> 包含有意义对象的图像应该具有低熵的条件标签分布 p(y|x)。

——[训练 GANs 的改进技术](https://arxiv.org/abs/1606.03498)，2016 年。

熵的计算方法是每个观测概率的负和乘以概率的对数。这里的直觉是，大概率比小概率具有更少的信息。

*   熵=-和(p_i * log(p_i))

条件概率抓住了我们对图像质量的兴趣。

为了捕捉我们对各种图像的兴趣，我们使用了边缘概率。这是所有生成图像的概率分布。因此，我们更希望边缘概率分布的积分具有高熵。

> 此外，我们期望该模型生成各种图像，因此边缘积分 p(y|x = G(z))dz 应该具有高熵。

——[训练 GANs 的改进技术](https://arxiv.org/abs/1606.03498)，2016 年。

这些元素通过计算条件概率分布和边缘概率分布之间的[库尔巴克-莱布勒散度](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)或 KL 散度(相对熵)来组合。

计算两个分布之间的散度是使用“||”运算符编写的，因此我们可以说我们对条件分布的 C 和边际分布的 M 之间的 KL 散度感兴趣，或者:

*   KL (C || M)

具体来说，我们对所有生成图像的 KL 散度的平均值感兴趣。

> 结合这两个需求，我们提出的度量是:exp(Ex KL(p(y|x)||p(y))。

——[训练 GANs 的改进技术](https://arxiv.org/abs/1606.03498)，2016 年。

我们不需要翻译初始得分的计算。值得庆幸的是，论文的作者还提供了 GitHub 上的[源代码，其中包含了一个](https://github.com/openai/improved-gan)[实现的初始得分](https://github.com/openai/improved-gan/blob/master/inception_score/model.py)。

分数的计算假设一系列对象的大量图像，例如 50，000。

将图像分成 10 组，例如每组 5，000 幅图像，并对每组图像计算初始得分，然后报告分数的平均值和标准偏差。

对一组图像的初始得分的计算包括首先使用初始 v3 模型来计算每个图像的条件概率(p(y|x))。然后，边缘概率被计算为组中图像的条件概率的平均值(p(y))。

然后为每个图像计算 KL 散度，作为条件概率乘以条件概率的对数减去边缘概率的对数。

*   KL 散度= p(y | x)*(log(p(y | x))–log(p(y)))

然后在所有图像上对 KL 散度求和，并在所有类别上求平均值，计算结果的指数以给出最终分数。

这定义了在大多数使用分数的论文中报告时使用的官方初始得分实现，尽管在如何计算分数上确实存在差异。

## 如何用 NumPy 实现初始评分

用 NumPy 数组在 Python 中实现初始得分的计算非常简单。

首先，让我们定义一个函数，它将收集条件概率并计算初始得分。

下面列出的*calculate _ initiation _ score()*函数执行该过程。

一个小的变化是在计算对数概率时引入了一个ε(一个接近于零的微小数字)，以避免在试图计算零概率的对数时爆炸。这在实践中可能是不需要的(例如，对于真实生成的图像)，但是在这里是有用的，并且是处理对数概率时的良好实践。

```py
# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
	# calculate p(y)
	p_y = expand_dims(p_yx.mean(axis=0), 0)
	# kl divergence for each image
	kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
	# sum over classes
	sum_kl_d = kl_d.sum(axis=1)
	# average over images
	avg_kl_d = mean(sum_kl_d)
	# undo the logs
	is_score = exp(avg_kl_d)
	return is_score
```

然后我们可以测试这个函数来计算一些人为条件概率的初始得分。

我们可以想象三类图像的情况，以及对于三个图像的每一类的完美自信预测。

```py
# conditional probabilities for high quality images
p_yx = asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
```

我们希望这个案例的初始得分是 3.0(或者非常接近)。这是因为我们对于每个图像类都有相同数量的图像(三个类中的每一个都有一个图像)，并且每个条件概率都是最有把握的。

下面列出了计算这些概率的初始得分的完整示例。

```py
# calculate inception score in numpy
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp

# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
	# calculate p(y)
	p_y = expand_dims(p_yx.mean(axis=0), 0)
	# kl divergence for each image
	kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
	# sum over classes
	sum_kl_d = kl_d.sum(axis=1)
	# average over images
	avg_kl_d = mean(sum_kl_d)
	# undo the logs
	is_score = exp(avg_kl_d)
	return is_score

# conditional probabilities for high quality images
p_yx = asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
score = calculate_inception_score(p_yx)
print(score)
```

运行该示例给出的预期得分为 3.0(或非常接近的数字)。

```py
2.999999999999999
```

我们也可以试试最坏的情况。

这就是为什么我们对于每个类仍然有相同数量的图像(三个类中的每一个都有一个)，但是对象是未知的，这给出了每个类的统一预测概率分布。

```py
# conditional probabilities for low quality images
p_yx = asarray([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
score = calculate_inception_score(p_yx)
print(score)
```

在这种情况下，我们期望初始得分是最差的，条件分布和边际分布之间没有差异，例如初始得分为 1.0。

将这些联系在一起，完整的示例如下所示。

```py
# calculate inception score in numpy
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp

# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
	# calculate p(y)
	p_y = expand_dims(p_yx.mean(axis=0), 0)
	# kl divergence for each image
	kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
	# sum over classes
	sum_kl_d = kl_d.sum(axis=1)
	# average over images
	avg_kl_d = mean(sum_kl_d)
	# undo the logs
	is_score = exp(avg_kl_d)
	return is_score

# conditional probabilities for low quality images
p_yx = asarray([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
score = calculate_inception_score(p_yx)
print(score)
```

运行该示例报告预期的初始得分为 1.0。

```py
1.0
```

您可能想尝试计算初始得分，并测试其他病理情况。

## 如何用 Keras 实现初始评分

现在我们知道如何计算初始得分并在 Python 中实现它，我们可以在 Keras 中开发一个实现。

这包括使用真实的 Inception v3 模型对图像进行分类，并在图像集合的多个分割中平均计算得分。

首先，我们可以直接在 Keras 中加载 Inception v3 模型。

```py
...
# load inception v3 model
model = InceptionV3()
```

该模型期望图像是彩色的，并且具有 299×299 像素的形状。

此外，像素值必须以与训练数据图像相同的方式进行缩放，然后才能进行分类。

这可以通过将像素值从整数转换为浮点值，然后为图像调用*prepare _ input()*函数来实现。

```py
...
# convert from uint8 to float32
processed = images.astype('float32')
# pre-process raw images for inception v3 model
processed = preprocess_input(processed)
```

然后，可以为图像预测 1000 个图像类别中的每一个的条件概率。

```py
...
# predict class probabilities for images
yhat = model.predict(images)
```

然后，可以像我们在上一节中所做的那样，直接在概率的 NumPy 数组上计算初始得分。

在此之前，我们必须将条件概率分成组，由 *n_split* 参数控制，并设置为原始论文中使用的默认值 10。

```py
...
n_part = floor(images.shape[0] / n_split)
```

然后，我们可以在 *n_part* 图像或预测的块中枚举条件概率，并计算初始得分。

```py
...
# retrieve p(y|x)
ix_start, ix_end = i * n_part, (i+1) * n_part
p_yx = yhat[ix_start:ix_end]
```

在计算了条件概率的每个分割的分数之后，我们可以计算并返回平均和标准偏差初始得分。

```py
...
# average across images
is_avg, is_std = mean(scores), std(scores)
```

将所有这些联系在一起，下面的*calculate _ inception _ score()*函数获取一组预期大小和像素值在[0，255]的图像，并使用 Keras 中的 inception v3 模型计算平均和标准差的 inception 分数。

```py
# assumes images have the shape 299x299x3, pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
	# load inception v3 model
	model = InceptionV3()
	# convert from uint8 to float32
	processed = images.astype('float32')
	# pre-process raw images for inception v3 model
	processed = preprocess_input(processed)
	# predict class probabilities for images
	yhat = model.predict(processed)
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve p(y|x)
		ix_start, ix_end = i * n_part, i * n_part + n_part
		p_yx = yhat[ix_start:ix_end]
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std
```

我们可以用 50 幅所有像素值为 1.0 的人工图像来测试这个函数。

```py
...
# pretend to load images
images = ones((50, 299, 299, 3))
print('loaded', images.shape)
```

这将计算每组五幅图像的分数，低质量将表明平均初始得分为 1.0。

下面列出了完整的示例。

```py
# calculate inception score with Keras
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

# assumes images have the shape 299x299x3, pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
	# load inception v3 model
	model = InceptionV3()
	# convert from uint8 to float32
	processed = images.astype('float32')
	# pre-process raw images for inception v3 model
	processed = preprocess_input(processed)
	# predict class probabilities for images
	yhat = model.predict(processed)
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve p(y|x)
		ix_start, ix_end = i * n_part, i * n_part + n_part
		p_yx = yhat[ix_start:ix_end]
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std

# pretend to load images
images = ones((50, 299, 299, 3))
print('loaded', images.shape)
# calculate inception score
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)
```

运行该示例首先定义 50 个假图像，然后计算每个批次的初始得分，并报告预期的初始得分为 1.0，标准偏差为 0.0。

**注**:第一次使用 InceptionV3 模型时，Keras 会下载模型权重保存到 *~/。工作站上的 keras/models/* 目录。权重约为 100 兆字节，根据您的互联网连接速度，下载可能需要一些时间。

```py
loaded (50, 299, 299, 3)
score 1.0 0.0
```

我们可以在一些真实图像上测试初始得分的计算。

Keras 应用编程接口提供对 [CIFAR-10 数据集](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/)的访问。

这些是小尺寸 32×32 像素的彩色照片。首先，我们可以将图像分成组，然后将图像上采样到预期的大小 299×299，对像素值进行预处理，预测类概率，然后计算初始得分。

如果您打算在自己生成的图像上计算初始得分，这将是一个有用的例子，因为您可能必须将图像缩放到初始 v3 模型的预期大小，或者更改模型来为您执行上采样。

首先，图像可以被加载和混洗，以确保每个分割覆盖不同的类集。

```py
...
# load cifar10 images
(images, _), (_, _) = cifar10.load_data()
# shuffle images
shuffle(images)
```

接下来，我们需要一种缩放图像的方法。

我们将使用 [scikit-image 库](https://scikit-image.org/)将像素值的 NumPy 数组调整到所需的大小。下面的 *scale_images()* 功能实现了这一点。

```py
# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
```

注意，如果尚未安装 scikit-image 库，您可能需要安装它。这可以通过以下方式实现:

```py
sudo pip install scikit-image
```

然后，我们可以列举拆分的数量，选择图像的子集，缩放它们，预处理它们，并使用该模型来预测条件类概率。

```py
...
# retrieve images
ix_start, ix_end = i * n_part, (i+1) * n_part
subset = images[ix_start:ix_end]
# convert from uint8 to float32
subset = subset.astype('float32')
# scale images to the required size
subset = scale_images(subset, (299,299,3))
# pre-process images, scale to [-1,1]
subset = preprocess_input(subset)
# predict p(y|x)
p_yx = model.predict(subset)
```

初始得分的其余计算是相同的。

将所有这些结合起来，下面列出了在真实的 CIFAR-10 训练数据集上计算初始得分的完整示例。

基于最初的初始得分论文中报告的类似计算，我们预计该数据集上报告的分数大约为 11。有趣的是，在使用渐进式增长 GAN 编写时，生成图像的 CIFAR-10 的[最佳初始得分约为 8.8。](https://paperswithcode.com/sota/image-generation-on-cifar-10)

```py
# calculate inception score for cifar-10 in Keras
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
	# load inception v3 model
	model = InceptionV3()
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve images
		ix_start, ix_end = i * n_part, (i+1) * n_part
		subset = images[ix_start:ix_end]
		# convert from uint8 to float32
		subset = subset.astype('float32')
		# scale images to the required size
		subset = scale_images(subset, (299,299,3))
		# pre-process images, scale to [-1,1]
		subset = preprocess_input(subset)
		# predict p(y|x)
		p_yx = model.predict(subset)
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std

# load cifar10 images
(images, _), (_, _) = cifar10.load_data()
# shuffle images
shuffle(images)
print('loaded', images.shape)
# calculate inception score
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)
```

运行该示例将加载数据集，准备模型，并计算 CIFAR-10 训练数据集的初始得分。

我们可以看到得分是 11.3，接近预期得分 11.24。

**注意**:第一次使用 CIFAR-10 数据集时，Keras 会下载压缩格式的图片，存储在 *~/。keras/datasets/* 目录。下载量约为 161 兆字节，根据您的互联网连接速度，可能需要几分钟。

```py
loaded (50000, 32, 32, 3)
score 11.317895 0.14821531
```

## 《Inception》评分的问题

初始得分是有效的，但并不完美。

通常，初始得分适用于模型已知的用于计算条件类概率的对象的生成图像。

在这种情况下，因为使用了初始 v3 模型，这意味着它最适合 [ILSVRC 2012 数据集](http://image-net.org/challenges/LSVRC/2012/)中使用的 1000 种对象类型。这是一个很大的类，但不是我们感兴趣的所有对象。

您可以在这里看到完整的课程列表:

*   [ILSVRC 2012 数据集的 1000 个对象类](http://image-net.org/challenges/LSVRC/2012/browse-synsets)。

它还要求图像是正方形的，并且具有大约 300×300 像素的相对较小的尺寸，包括将生成的图像缩放到该尺寸所需的任何缩放。

一个好的分数还需要生成的图像在模型支持的可能对象中有一个好的分布，并且每个类有接近偶数个例子。对于许多不能控制生成对象类型的 GAN 模型来说，这是很难控制的。

[Shane Barratt](http://web.stanford.edu/~sbarratt/) 和 Rishi Sharma 在他们 2018 年发表的题为“[关于《Inception》评分的注释](https://arxiv.org/abs/1801.01973)的论文中，仔细查看了《Inception》评分，并列出了一些技术问题和边缘案例如果你想潜得更深，这是一个很好的参考。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [训练 GANs 的改进技术](https://arxiv.org/abs/1606.03498)，2016。
*   [关于《Inception》评分的一个注记](https://arxiv.org/abs/1801.01973)，2018。
*   [重新思考计算机视觉的初始架构](https://arxiv.org/abs/1512.00567)，2015 年。

### 项目

*   [论文“训练 GANs 的改进技术”的代码](https://github.com/openai/improved-gan)
*   【2012 年大规模视觉识别挑战赛(ILSVRC2012)

### 应用程序接口

*   [Keras Inception v3 模型](https://keras.io/applications/#inceptionv3)
*   [sci kit-图像库](https://scikit-image.org/)

### 文章

*   [在 CIFAR-10 上生成图像](https://paperswithcode.com/sota/image-generation-on-cifar-10)
*   [初始得分计算](https://github.com/openai/improved-gan/issues/29)，2017 年。
*   [初始得分的简单解释](https://medium.com/octavian-ai/a-simple-explanation-of-the-inception-score-372dff6a8c7a)
*   [Inception 评分——评估你 GAN](https://sudomake.ai/inception-score-explained/) 的真实感，2018。
*   [kul LBA-leilbler 分歧，维基百科。](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
*   [熵(信息论)，维基百科。](https://en.wikipedia.org/wiki/Entropy_(information_theory))

## 摘要

在本教程中，您发现了评估生成图像质量的初始得分。

具体来说，您了解到:

*   如何计算初始得分以及它所衡量的东西背后的直觉。
*   如何用 NumPy 和 Keras 深度学习库在 Python 中实现初始得分。
*   如何计算小型图像的初始得分，例如 CIFAR-10 数据集中的图像。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。