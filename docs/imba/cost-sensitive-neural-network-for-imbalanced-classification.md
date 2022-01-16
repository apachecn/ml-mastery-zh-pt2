# 如何为不平衡分类开发成本敏感的神经网络

> 原文：<https://machinelearningmastery.com/cost-sensitive-neural-network-for-imbalanced-classification/>

最后更新于 2020 年 8 月 21 日

深度学习神经网络是一类灵活的机器学习算法，在广泛的问题上表现良好。

使用误差反向传播算法训练神经网络，该算法包括计算模型在训练数据集上产生的误差，并根据这些误差按比例更新模型权重。这种训练方法的局限性在于，每个类的例子都被同等对待，对于不平衡的数据集，这意味着一个类比另一个类更适合模型。

反向传播算法可以被更新，以与类别的重要性成比例地加权错误分类误差，称为加权神经网络或成本敏感神经网络。这使得模型在类分布严重偏斜的数据集中，更关注少数类的例子，而不是多数类的例子。

在本教程中，您将发现用于不平衡分类的加权神经网络。

完成本教程后，您将知道:

*   标准神经网络算法如何不支持不平衡分类？
*   如何修改神经网络训练算法，以便根据类别重要性按比例加权错误分类错误。
*   如何为神经网络配置类权重并评估对模型表现的影响？

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Develop a Cost-Sensitive Neural Network for Imbalanced Classification](img/94138d312002193bd2e590857428f746.png)

如何为不平衡分类开发一个成本敏感的神经网络。新西兰，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  不平衡类别数据集
2.  Keras 神经网络模型
3.  不平衡分类的深度学习
4.  带 Keras 的加权神经网络

## 不平衡类别数据集

在我们深入研究不平衡分类的神经网络修改之前，让我们首先定义一个不平衡类别数据集。

我们可以使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)定义一个合成的不平衡两类类别数据集。我们将生成 10，000 个少数与多数类比例大约为 1:100 的示例。

```py
...
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
```

生成后，我们可以总结类分布，以确认数据集是按照我们的预期创建的。

```py
...
# summarize class distribution
counter = Counter(y)
print(counter)
```

最后，我们可以创建示例的散点图，并按类别标签对它们进行着色，以帮助理解从该数据集中对示例进行分类的挑战。

```py
...
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

将这些联系在一起，下面列出了生成合成数据集和绘制示例的完整示例。

```py
# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
# summarize class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

运行该示例首先创建数据集并总结类分布。

我们可以看到，数据集具有大约 1:100 的类分布，多数类中的示例不到 10，000 个，少数类中的示例不到 100 个。

```py
Counter({0: 9900, 1: 100})
```

接下来，创建数据集的散点图，显示多数类的大量示例(蓝色)和少数类的少量示例(橙色)，并有一些适度的类重叠。

![Scatter Plot of Binary Classification Dataset with 1 to 100 Class Imbalance](img/9335c399e118b2dfd1ac9b288ae41770.png)

1 到 100 类不平衡的二进制类别数据集的散点图

## Keras 神经网络模型

接下来，我们可以在数据集上拟合标准神经网络模型。

首先，我们可以定义一个函数来创建合成数据集，并将其分成单独的训练和测试数据集，每个数据集有 5000 个示例。

```py
# prepare train and test dataset
def prepare_data():
	# generate 2d classification dataset
	X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
	# split into train and test
	n_train = 5000
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy
```

可以使用 [Keras 深度学习库](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)定义多层感知器神经网络。我们将定义一个神经网络，它需要两个输入变量，有一个包含 10 个节点的隐藏层，然后是一个预测类标签的输出层。

我们将在隐藏层使用流行的 [ReLU 激活](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)函数，在输出层使用 sigmoid 激活函数，以确保预测的概率在[0，1]范围内。该模型将使用具有默认学习率的随机梯度下降进行拟合，并根据交叉熵损失进行优化。

网络结构和超参数没有针对问题进行优化；相反，当训练算法后来被修改以处理倾斜的类分布时，网络提供了比较的基础。

下面的 *define_model()* 函数定义并返回模型，以网络的输入变量个数为自变量。

```py
# define the neural network model
def define_model(n_input):
	# define model
	model = Sequential()
	# define first hidden layer and visible layer
	model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
	# define output layer
	model.add(Dense(1, activation='sigmoid'))
	# define loss and optimizer
	model.compile(loss='binary_crossentropy', optimizer='sgd')
	return model
```

一旦定义了模型，它就可以适合训练数据集。

我们将为 100 个训练阶段的模型设定默认的批次大小。

```py
...
# fit model
model.fit(trainX, trainy, epochs=100, verbose=0)
```

一旦拟合，我们可以使用该模型对测试数据集进行预测，然后使用 [ROC AUC](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/) 分数评估预测。

```py
...
# make predictions on the test dataset
yhat = model.predict(testX)
# evaluate the ROC AUC of the predictions
score = roc_auc_score(testy, yhat)
print('ROC AUC: %.3f' % score)
```

将这些联系在一起，下面列出了在不平衡类别数据集上拟合标准神经网络模型的完整示例。

```py
# standard neural network on an imbalanced classification dataset
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from keras.layers import Dense
from keras.models import Sequential

# prepare train and test dataset
def prepare_data():
	# generate 2d classification dataset
	X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
	# split into train and test
	n_train = 5000
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy

# define the neural network model
def define_model(n_input):
	# define model
	model = Sequential()
	# define first hidden layer and visible layer
	model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
	# define output layer
	model.add(Dense(1, activation='sigmoid'))
	# define loss and optimizer
	model.compile(loss='binary_crossentropy', optimizer='sgd')
	return model

# prepare dataset
trainX, trainy, testX, testy = prepare_data()
# define the model
n_input = trainX.shape[1]
model = define_model(n_input)
# fit model
model.fit(trainX, trainy, epochs=100, verbose=0)
# make predictions on the test dataset
yhat = model.predict(testX)
# evaluate the ROC AUC of the predictions
score = roc_auc_score(testy, yhat)
print('ROC AUC: %.3f' % score)
```

运行该示例评估不平衡数据集上的神经网络模型，并报告 ROC AUC。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，模型实现了大约 0.949 的 ROC AUC。这表明，与 ROC AUC 为 0.5 的朴素分类器相比，该模型具有一些技巧。

```py
ROC AUC: 0.949
```

这为标准神经网络训练算法的任何修改的比较提供了基线。

## 不平衡分类的深度学习

神经网络模型通常使用误差算法的反向传播来训练。

这包括使用模型的当前状态为训练集示例进行预测，计算预测的误差，然后使用误差更新模型权重，并将误差的信用分配给从输出层向后到输入层的不同节点和层。

给定对错误分类错误的平衡关注，大多数标准神经网络算法不太适合具有严重偏斜分类分布的数据集。

> 现有的深度学习算法大多没有考虑数据不平衡问题。因此，这些算法可以在平衡数据集上很好地执行，而在不平衡数据集上它们的表现不能得到保证。

——[在不平衡数据集上训练深度神经网络](https://ieeexplore.ieee.org/abstract/document/7727770)，2016。

可以修改这个训练过程，使得一些例子比其他例子有更多或更少的误差。

> 也可以通过改变被最小化的误差函数来考虑错误分类成本。反向传播学习过程应该使错误分类成本最小化，而不是最小化平方误差。

——[神经网络成本敏感学习](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.13.8285)，1998。

实现这一点的最简单的方法是基于示例的类别对其使用固定的错误分数权重，其中对于更重要类别中的示例，预测错误增加，而对于不太重要类别中的示例，预测错误减少或保持不变。

> ……成本敏感的学习方法基于与错误分类样本相关的成本考虑来解决数据不平衡问题。特别是，它为样本的错误分类分配了不同的成本值。

——[在不平衡数据集上训练深度神经网络](https://ieeexplore.ieee.org/abstract/document/7727770)，2016。

可以对少数类中的那些示例应用较大的误差权重，因为它们在不平衡分类问题中通常比多数类中的示例更重要。

*   **大权重**:分配给小众类的例子。
*   **小权重**:分配给多数类的例子。

对神经网络训练算法的这种修改被称为加权神经网络或成本敏感神经网络。

通常，在定义成本或用于成本敏感学习的“T0”权重时，需要特别注意。然而，对于只关注错误分类的不平衡分类，加权可以使用在训练数据集中观察到的类别分布的倒数。

## 带 Keras 的加权神经网络

Keras Python 深度学习库提供支持类加权。

用于训练 Keras 神经网络模型的 [fit()函数](https://keras.io/models/sequential/)采用了一个名为 *class_weight* 的参数。此参数允许您定义一个字典，将类整数值映射到应用于每个类的重要性。

此函数用于训练每种不同类型的神经网络，包括多层感知器、卷积神经网络和递归神经网络，因此类别加权功能适用于所有这些网络类型。

例如，每个类别 0 和 1 的 1 比 1 权重可以定义如下:

```py
...
# fit model
weights = {0:1, 1:1}
history = model.fit(trainX, trainy, class_weight=weights, ...)
```

类别称重可以多种方式定义；例如:

*   **领域专长**，通过与主题专家交谈确定。
*   **调谐**，由超参数搜索如网格搜索确定。
*   **启发式**，使用一般最佳实践指定。

使用类别权重的最佳实践是使用训练数据集中类别分布的倒数。

例如，测试数据集的类分布是少数类与多数类的比例为 1:100。该比率的倒数可以与多数类的 1 和少数类的 100 一起使用，例如:

```py
...
# fit model
weights = {0:1, 1:100}
history = model.fit(trainX, trainy, class_weight=weights, ...)
```

代表相同比率的分数没有相同的效果。例如，对多数类和少数类分别使用 0.01 和 0.99 可能会导致比使用 1 和 100 更差的表现(在这种情况下确实如此)。

```py
...
# fit model
weights = {0:0.01, 1:0.99}
history = model.fit(trainX, trainy, class_weight=weights, ...)
```

原因是从多数阶级和少数阶级提取的例子的误差减少了。此外，多数类的误差减少被显著地缩小到非常小的数字，这些数字可能对模型权重有有限的影响或者只有非常小的影响。

因此，建议使用这样的整数来表示类别权重，例如 1 表示无变化，100 表示类别 1 的错误分类错误，其影响或损失是类别 0 的错误分类错误的 100 倍。

我们可以使用上一节中定义的相同评估过程，使用类权重来评估神经网络算法。

我们期望神经网络的类加权版本比没有任何类加权的训练算法版本表现得更好。

下面列出了完整的示例。

```py
# class weighted neural network on an imbalanced classification dataset
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from keras.layers import Dense
from keras.models import Sequential

# prepare train and test dataset
def prepare_data():
	# generate 2d classification dataset
	X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
	# split into train and test
	n_train = 5000
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy

# define the neural network model
def define_model(n_input):
	# define model
	model = Sequential()
	# define first hidden layer and visible layer
	model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
	# define output layer
	model.add(Dense(1, activation='sigmoid'))
	# define loss and optimizer
	model.compile(loss='binary_crossentropy', optimizer='sgd')
	return model

# prepare dataset
trainX, trainy, testX, testy = prepare_data()
# get the model
n_input = trainX.shape[1]
model = define_model(n_input)
# fit model
weights = {0:1, 1:100}
history = model.fit(trainX, trainy, class_weight=weights, epochs=100, verbose=0)
# evaluate model
yhat = model.predict(testX)
score = roc_auc_score(testy, yhat)
print('ROC AUC: %.3f' % score)
```

运行该示例准备合成的不平衡类别数据集，然后评估神经网络训练算法的类加权版本。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

报告了 ROC AUC 分数，在这种情况下显示出比训练算法的未加权版本更好的分数，或者与大约 0.949 相比大约 0.973。

```py
ROC AUC: 0.973
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [神经网络的成本敏感学习](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.13.8285)，1998。
*   [在不平衡数据集上训练深度神经网络](https://ieeexplore.ieee.org/abstract/document/7727770)，2016。

### 书

*   [从不平衡数据集中学习](https://amzn.to/307Xlva)，2018。
*   [不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

### 蜜蜂

*   [sklearn . datasets . make _ classification API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
*   [硬模型 API](https://keras.io/models/sequential/) 。

## 摘要

在本教程中，您发现了用于不平衡分类的加权神经网络。

具体来说，您了解到:

*   标准神经网络算法如何不支持不平衡分类？
*   如何修改神经网络训练算法，以便根据类别重要性按比例加权错误分类错误。
*   如何为神经网络配置类权重并评估对模型表现的影响？

你有什么问题吗？
在下面的评论中提问，我会尽力回答。