# 如何获得更好的深度学习效果（7 天迷你课程）

> 原文：<https://machinelearningmastery.com/better-deep-learning-neural-networks-crash-course/>

最后更新于 2020 年 1 月 8 日

### 更好的深度学习神经网络速成班。

#### 在 7 天内从深度学习模式中获得更好的表现。

配置神经网络模型通常被称为“黑暗艺术”

这是因为没有针对给定问题配置网络的硬性规则。我们无法分析计算给定数据集的最佳模型类型或模型配置。

幸运的是，在配置和训练神经网络时，有一些已知的技术可以解决特定的问题，这些技术在现代深度学习库中是可用的，比如 Keras。

在本速成课程中，您将发现如何在七天内自信地从深度学习模型中获得更好的表现。

这是一个又大又重要的岗位。你可能想把它做成书签。

**用我的新书[更好的深度学习](https://machinelearningmastery.com/better-deep-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 1 月更新**:更新了 Keras 2.3 和 TensorFlow 2.0 的 API。

![How to Get Better Deep Learning Performance (7 Day Mini-Course)](img/e8c2eebe227ee385c587b821a178b6d2.png)

如何获得更好的深度学习表现(7 天迷你课程)
图片由[达米安·加达尔](https://www.flickr.com/photos/23024164@N06/6827097549/)提供，版权所有。

## 这个速成班是给谁的？

在我们开始之前，让我们确保你在正确的地方。

下面的列表提供了一些关于本课程是为谁设计的一般指南。

你需要知道:

*   绕过基本的 Python 和 NumPy。
*   深度学习的基础知识。

你不需要知道:

*   如何成为数学奇才！
*   如何成为深度学习专家！

这个速成课程将把你从一个知道一点深度学习的开发人员带到一个可以在深度学习项目中获得更好表现的开发人员。

注意:本速成课程假设您有一个工作正常的 Python 2 或 3 SciPy 环境，其中至少安装了 NumPy 和 Keras 2。如果您需要环境方面的帮助，可以遵循这里的逐步教程:

*   [如何用 Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 建立机器学习和深度学习的 Python 环境

## 速成班概述

这门速成课分为七节课。

您可以每天完成一节课(推荐)或一天内完成所有课程(硬核)。这真的取决于你有多少时间和你的热情程度。

以下七节课将让您自信地提高深度学习模式的绩效:

*   **第 01 课**:更好的深度学习框架
*   **第 02 课**:批量
*   **第 03 课**:学习进度计划表
*   **第 04 课**:批次正常化
*   **第 05 课**:权重正则化
*   **第 06 课**:增加噪音
*   **第 07 课**:提前停止

每节课可能需要你 60 秒或 30 分钟。慢慢来，按照自己的节奏完成课程。提问，甚至在下面的评论中发布结果。

这些课程期望你去发现如何做事。我会给你提示，但每节课的一部分要点就是逼你学会去哪里找帮助(提示，我所有的答案都直接在这个博客上；使用搜索框)。

我确实以相关帖子链接的形式提供了更多帮助，因为我想让你建立一些信心和惰性。

在评论中发布您的结果；我会为你加油的！

坚持住。不要放弃。

**注**:这只是速成班。要了解更多细节和充实的教程，请参阅我的书，主题为“T2 更好的深度学习”

## 第 01 课:更好的深度学习框架

在本课中，您将发现一个框架，可以用来系统地提高深度学习模型的表现。

现代深度学习库(如 Keras)允许您在几分钟内用几行代码定义并开始拟合各种神经网络模型。

然而，配置神经网络以在新的预测建模问题上获得良好的表现仍然是一个挑战。

就深度学习神经网络模型的不良表现而言，有三种类型的问题很容易诊断；它们是:

*   **学习上的问题**。学习中的问题表现在模型不能有效地学习训练数据集，或者在学习训练数据集时表现出缓慢的进度或糟糕的表现。
*   **泛化的问题**。泛化的问题表现在一个模型中，该模型过度扩展了训练数据集，并且在保持数据集上表现不佳。
*   **预测问题**。预测的问题表现为随机训练算法对最终模型有很大的影响，导致行为和表现的巨大差异。

提议的细分中三个领域之间的顺序关系允许深度学习模型表现的问题首先被隔离，然后用特定的技术或方法作为目标。

我们可以将有助于解决这些问题的技术总结如下:

*   **更好的学习**。响应于训练数据集改进或加速神经网络模型权重自适应的技术。
*   **更好的概括**。提高神经网络模型在保持数据集上的表现的技术。
*   **更好的预测**。降低最终模型表现差异的技术。

您可以使用这个框架来首先诊断您的问题类型，然后确定一种技术来评估以尝试解决您的问题。

### 你的任务

在本课中，您必须列出属于框架三个领域的两种技术或关注领域。

有问题吗？请注意，作为本小型课程的一部分，我们将从三个领域中的两个领域寻找一些示例。

在下面的评论中发表你的答案。我很想看看你的发现。

### 然后

在下一课中，您将发现如何控制批量学习的速度。

## 第 02 课:批量

在本课中，您将发现[批次大小](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)在训练神经网络时的重要性。

使用梯度下降来训练神经网络，其中基于训练数据集的子集来计算用于更新权重的误差估计。

误差梯度估计中使用的训练数据集中的示例数称为批量，是影响学习算法动态性的重要超参数。

批量的选择控制算法学习的速度，例如:

*   **批量梯度下降**。批量大小设置为训练数据集中的示例数，误差估计更准确，但权重更新之间的时间更长。
*   **随机梯度下降**。批次大小设置为 1，误差估计值有噪声，但权重经常更新。
*   **迷你批次梯度下降**。批处理大小设置为大于 1 且小于训练示例数的值，在批处理和随机梯度下降之间进行权衡。

Keras 允许您通过 *fit()* 函数的 *batch_size* 参数来配置批次大小，例如:

```py
# fit model
history = model.fit(trainX, trainy, epochs=1000, batch_size=len(trainX))
```

下面的例子演示了一个多层感知器，它在二进制分类问题上具有批量梯度下降。

```py
# example of batch gradient descent
from sklearn.datasets import make_circles
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot
# generate dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1000, batch_size=len(trainX), verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

### 你的任务

对于本课，您必须对每种梯度下降类型(批处理、小批量和随机)运行代码示例，并描述它在训练期间对[学习曲线](https://machinelearningmastery.com/how-to-control-neural-network-model-capacity-with-nodes-and-layers/)的影响。

在下面的评论中发表你的答案。我很想看看你的发现。

### 然后

在下一课中，您将了解如何在培训过程中根据学习进度计划对模型进行微调

## 第 03 课:学习进度计划

在本课中，您将了解如何配置自适应学习率计划，以便在培训运行期间微调模型。

在这个搜索过程的每一步中，模型的变化量，或者说步长，被称为“*学习率*”，它可能提供了最重要的超参数来调整你的神经网络，以便在你的问题上取得良好的表现。

配置一个固定的学习率是非常具有挑战性的，需要仔细的实验。使用固定学习率的替代方法是在整个训练过程中改变学习率。

Keras 提供了*reduce lronplateaau*学习率计划，当检测到模型表现平稳时，例如，给定数量的训练时期没有变化，该计划将调整学习率。例如:

```py
# define learning rate schedule
rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_delta=1E-7, verbose=1)
```

该回调旨在降低模型停止改进后的学习率，希望在训练过程中微调模型权重。

下面的示例演示了一个多层感知器，该感知器在二进制分类问题上有一个学习率计划，如果在 5 个训练时期内没有检测到验证损失的变化，学习率将降低一个数量级。

```py
# example of a learning rate schedule
from sklearn.datasets import make_circles
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot
# generate dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# define learning rate schedule
rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_delta=1E-7, verbose=1)
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=300, verbose=0, callbacks=[rlrp])
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

### 你的任务

在本课中，您必须在有和没有学习率计划的情况下运行代码示例，并描述学习率计划在培训期间对学习曲线的影响。

在下面的评论中发表你的答案。我很想看看你的发现。

### 然后

在下一课中，您将发现如何通过批处理规范化来加速培训过程

## 第 04 课:批量标准化

在本课中，您将发现如何使用批处理规范化来加速深度学习神经网络的训练过程。

批处理规范化，简称 batchnorm，是一种帮助协调模型中多个层的更新的技术。

引入批量归一化的论文作者将训练期间输入分布的变化称为“*内部协变量移位*”。批处理规范化旨在通过缩放前一层的输出来对抗内部协变量偏移，特别是通过标准化每个小批处理的每个输入变量的激活，例如前一层节点的激活。

Keras 通过单独的*批处理规范化*层支持批处理规范化，该层可以添加到模型的隐藏层之间。例如:

```py
model.add(BatchNormalization())
```

下面的例子演示了一个多层感知器模型，它对二进制分类问题进行了批量规范化。

```py
# example of batch normalization
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from matplotlib import pyplot
# generate dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=300, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

### 你的任务

对于本课，您必须运行带有和不带有批处理规范化的代码示例，并描述批处理规范化在训练期间对学习曲线的影响。

在下面的评论中发表你的答案。我很想看看你的发现。

### 然后

在下一课中，您将发现如何使用权重正则化来减少过拟合。

## 第 05 课:权重正则化

在本课中，您将发现如何使用权重正则化来减少深度学习神经网络的过拟合。

权重大的模型比权重小的模型更复杂。这是一个网络的迹象，可能过于专门用于训练数据。

学习算法可以更新，以鼓励网络使用小权重。

一种方法是改变网络优化中使用的损耗计算，同时考虑权重的大小。这被称为权重正则化或权重衰减。

Keras 通过层上的*核正则化器*参数支持权重正则化，可以配置为使用 [L1 或 L2 向量范数](https://machinelearningmastery.com/vector-norms-machine-learning/)，例如:

```py
model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l2(0.01)))
```

下面的例子演示了一个多层感知器模型，它在二分类问题上具有权重衰减。

```py
# example of weight decay
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from matplotlib import pyplot
# generate dataset
X, y = make_circles(n_samples=100, noise=0.1, random_state=1)
# split into train and test
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

### 你的任务

对于本课，您必须运行带有和不带权重正则化的代码示例，并描述它在训练期间对学习曲线的影响。

在下面的评论中发表你的答案。我很想看看你的发现。

### 然后

在下一课中，您将发现如何通过向模型添加噪声来减少过拟合

## 第 06 课:添加噪音

在本课中，您将发现在训练过程中向神经网络添加噪声可以提高网络的鲁棒性，从而获得更好的泛化能力和更快的学习速度。

使用小数据集训练神经网络会导致网络记住所有训练示例，进而导致在保持数据集上表现不佳。

使输入空间更平滑、更容易学习的一种方法是在训练过程中给输入添加噪声。

在神经网络模型的训练期间添加噪声具有正则化效果，并且反过来提高了模型的鲁棒性。

噪声可以通过*高斯噪声*层添加到你的模型中。例如:

```py
model.add(GaussianNoise(0.1))
```

噪声可以添加到输入层或隐藏层之间的模型中。

下面的例子演示了一个多层感知器模型，在二进制分类问题上隐藏层之间增加了噪声。

```py
# example of adding noise
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GaussianNoise
from matplotlib import pyplot
# generate dataset
X, y = make_circles(n_samples=100, noise=0.1, random_state=1)
# split into train and test
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu'))
model.add(GaussianNoise(0.1))
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

### 你的任务

对于本课，您必须运行添加和不添加噪声的代码示例，并描述它在训练过程中对学习曲线的影响。

在下面的评论中发表你的答案。我很想看看你的发现。

### 然后

在下一课中，您将发现如何使用提前停止来减少过拟合。

## 第 07 课:提前停止

在本课中，您将发现在神经网络过度训练数据集之前尽早停止神经网络的训练可以减少过度训练并提高深度神经网络的泛化能力。

训练神经网络的一个主要挑战是训练它们需要多长时间。

太少的训练将意味着模型将会使火车和测试设备下不来。过多的训练将意味着模型会过度训练训练数据集，并且在测试集上表现不佳。

一种折中的方法是在训练数据集上进行训练，但在验证数据集的表现开始下降时停止训练。这种简单、有效且广泛使用的神经网络训练方法被称为提前停止。

Keras 支持通过*提前停止*回调来提前停止，该回调允许您指定训练期间要监控的指标。

```py
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
```

下面的示例演示了一个多层感知器，它在二进制分类问题上提前停止，当验证损失在 200 个训练时期内没有改善时，它将停止。

```py
# example of early stopping
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
# generate dataset
X, y = make_circles(n_samples=100, noise=0.1, random_state=1)
# split into train and test
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es])
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
```

### 你的任务

对于本课，您必须在有和没有提前停止的情况下运行代码示例，并描述它在培训期间对学习曲线的影响。

在下面的评论中发表你的答案。我很想看看你的发现。

### 然后

这是你的最后一课。

## 末日！
( *看你走了多远！*)

你成功了。干得好！

花一点时间，回头看看你已经走了多远。

你发现了:

*   一个框架，您可以使用它来系统地诊断和改进深度学习模型的表现。
*   批量可以用来控制估计误差的准确率和训练过程中的学习速度。
*   学习率计划可用于在训练期间微调模型权重。
*   批量归一化可用于显著加速神经网络模型的训练过程。
*   权重正则化将基于权重的大小惩罚模型，并减少过拟合。
*   添加噪声将使模型对输入差异更加稳健，并减少过拟合
*   提前停止将在正确的时间停止训练过程，减少过度训练。

这只是您深度学习绩效提升之旅的开始。不断练习和发展你的技能。

进行下一步，查看[我的关于通过深度学习获得更好表现的书](https://machinelearningmastery.com/better-deep-learning/)。

## 摘要

你觉得迷你课程怎么样？
你喜欢这个速成班吗？

你有什么问题吗？有什么症结吗？
让我知道。请在下面留言。