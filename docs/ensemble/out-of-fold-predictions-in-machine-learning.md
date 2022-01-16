# 如何在机器学习中使用折外预测

> 原文：<https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/>

最后更新于 2021 年 4 月 27 日

机器学习算法通常使用重采样技术进行评估，例如 k 倍交叉验证。

在 k 倍交叉验证过程中，对测试集进行预测，测试集由不用于训练模型的数据组成。这些预测被称为**超差预测**，一种样本外预测。

在未来对新数据进行预测时，所谓的模型的泛化表现，以及集成模型的开发中，超折叠预测在机器学习中都起着重要的作用。

在本教程中，您将发现一个关于机器学习中异常预测的温和介绍。

完成本教程后，您将知道:

*   超差预测是对未用于训练模型的数据进行的一种超差预测。
*   在对看不见的数据进行预测时，超差预测最常用于估计模型的表现。
*   叠外预测可以用来构建一个集合模型，称为叠化概化或叠化集合。

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 1 月更新**:针对 Sklearn v0.22 API 的变化进行了更新。

![How to Use Out-of-Fold Predictions in Machine Learning](img/a1d63fc885408f6c33dd5db32e9e3454.png)

如何在机器学习中使用超折叠预测
图片由[盖尔·瓦罗库](https://flickr.com/photos/gaelvaroquaux/39654884111/)提供，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  什么是异常预测？
2.  用于评估的超范围预测
3.  集成的异常预测

## 什么是异常预测？

使用重采样技术(如 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/))来评估数据集上机器学习算法的表现是很常见的。

K 折交叉验证过程包括将训练数据集分成 *k* 组，然后在测试集上使用每组 *k* 示例，而剩余示例用作训练集。

这意味着 *k* 不同的模型被训练和评估。模型的表现是使用所有 k 倍模型的预测来估计的。

该程序可总结如下:

*   1.随机打乱数据集。
*   2.将数据集分成 k 个组。
*   3.对于每个唯一的组:
    *   a.将该组作为保持或测试数据集。
    *   b.将剩余的组作为训练数据集。
    *   c.在训练集上安装一个模型，并在测试集上对其进行评估。
    *   d.保留评估分数并放弃模型。
*   4.使用模型评估分数样本总结模型技巧。

重要的是，数据样本中的每个观察值被分配到一个单独的组，并在整个过程中保持在该组中。这意味着每个样本都有机会在保持设置中使用 1 次，并用于训练模型 k-1 次。

有关 k-fold 交叉验证主题的更多信息，请参见教程:

*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)

折外预测是模型在 k 折叠交叉验证过程中的预测。

也就是说，超倍预测是在重采样过程中对保持数据集进行的预测。如果执行正确，训练数据集中的每个示例都有一个预测。

有时，折外用缩写 OOF 概括。

*   **折外预测**:模型在保持示例的 k 折叠交叉验证过程中做出的预测。

超差预测的概念与**超差预测**的概念直接相关，因为这两种情况下的预测都是在模型训练期间没有使用的例子上进行的，当用于对新数据进行预测时，可以用来估计模型的表现。

因此，超越预测是一种样本外预测，尽管是在使用 k 倍交叉验证评估的模型环境中描述的。

*   **样本外预测**:模型对模型训练期间未使用的数据进行的预测。

样本外预测也可以称为保持预测。

超折叠预测有两个主要用途；它们是:

*   根据看不见的数据评估模型的表现。
*   适合整体模型。

让我们仔细看看这两个案例。

## 用于评估的超范围预测

无序预测最常见的用途是估计模型的表现。

也就是说，对未用于训练模型的数据的预测可以使用诸如误差或准确度的评分标准来进行和评估。当用于对新数据进行预测时，例如当模型将在实践中用于进行预测时，该度量提供了模型表现的估计。

一般来说，对不用于训练模型的数据所做的预测提供了模型如何推广到新情况的洞察力。因此，评估这些预测的分数被称为机器学习模型的广义表现。

这些预测可以使用两种主要方法来估计模型的表现。

第一种方法是根据每次折叠期间所做的预测对模型进行评分，然后计算这些评分的平均值。例如，如果我们正在评估一个分类模型，那么可以对每组超出范围的预测计算分类准确率，然后可以报告平均准确率。

*   **方法 1** :将绩效评估为每组异常预测的平均得分。

第二种方法是考虑每个例子在每个测试集中只出现一次。也就是说，训练数据集中的每个示例在 k 倍交叉验证过程中都有一个预测。因此，我们可以收集所有预测，并将它们与预期结果进行比较，并直接在整个训练数据集中计算分数。

*   **方法 2:** 使用所有异常预测的总和来评估表现。

这两种方法都是合理的，每种方法得出的分数应该大致相等。

从每组样本外预测中计算平均值可能是最常见的方法，因为估计的方差也可以计算为标准偏差或标准误差。

> 对重新采样的表现估计值进行了总结(通常带有平均值和标准误差)…

—第 70 页，[应用预测建模](https://amzn.to/32M80ta)，2013 年。

我们可以用一个小的工作实例来说明这两种方法之间的差异，这两种方法使用不同的预测来评估模型。

我们将使用 [make_blobs() Sklearn](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) 函数创建一个包含 1000 个示例、两个类和 100 个输入特征的测试二进制分类问题。

下面的示例准备了一个数据示例，并总结了数据集的输入和输出元素的形状。

```py
# example of creating a test dataset
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# summarize the shape of the arrays
print(X.shape, y.shape)
```

运行该示例会打印输入数据的形状，显示 1000 行数据、100 列或输入要素以及相应的分类标签。

```py
(1000, 100) (1000,)
```

接下来，我们可以使用 *k* 折叠交叉验证来评估一个[kneighgborksclassifier](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)模型。

我们将使用 *k* =10 作为 [KFold](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.KFold.html) 对象，合理的默认值，在每个训练数据集上拟合一个模型，并在每个保持折叠上评估它。

准确性分数将存储在每个模型评估的列表中，并将报告这些分数的平均值和标准偏差。

下面列出了完整的示例。

```py
# evaluate model by averaging performance across each fold
from numpy import mean
from numpy import std
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# k-fold cross validation
scores = list()
kfold = KFold(n_splits=10, shuffle=True)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	model = KNeighborsClassifier()
	model.fit(train_X, train_y)
	# evaluate model
	yhat = model.predict(test_X)
	acc = accuracy_score(test_y, yhat)
	# store score
	scores.append(acc)
	print('> ', acc)
# summarize model performance
mean_s, std_s = mean(scores), std(scores)
print('Mean: %.3f, Standard Deviation: %.3f' % (mean_s, std_s))
```

运行该示例会报告每次迭代中保持折叠的模型分类准确率。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行结束时，报告准确度分数的平均值和标准偏差。

```py
>  0.95
>  0.92
>  0.95
>  0.95
>  0.91
>  0.97
>  0.96
>  0.96
>  0.98
>  0.91
Mean: 0.946, Standard Deviation: 0.023
```

我们可以用另一种方法来对比这一点，这种方法将所有预测作为一个整体来评估。

不是在每个保持折叠上评估模型，而是进行预测并存储在列表中。然后，在运行结束时，将预测值与每个保持测试集的期望值进行比较，并报告单个准确度分数。

下面列出了完整的示例。

```py
# evaluate model by calculating the score across all predictions
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# k-fold cross validation
data_y, data_yhat = list(), list()
kfold = KFold(n_splits=10, shuffle=True)
# enumerate splits
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# fit model
	model = KNeighborsClassifier()
	model.fit(train_X, train_y)
	# make predictions
	yhat = model.predict(test_X)
	# store
	data_y.extend(test_y)
	data_yhat.extend(yhat)
# evaluate the model
acc = accuracy_score(data_y, data_yhat)
print('Accuracy: %.3f' % (acc))
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例会收集每个保持数据集的所有预期值和预测值，并在运行结束时报告一个准确度分数。

```py
Accuracy: 0.930
```

同样，这两种方法是可比较的，这可能是你在自己的预测建模问题上使用的方法的品味问题。

## 集成的异常预测

超折叠预测的另一个常见用途是在集合模式的开发中使用它们。

集成是一种机器学习模型，它结合了在同一训练数据集上准备的两个或多个模型的预测。

这是在进行机器学习竞赛时非常常用的程序。

当不用于训练模型时，折外预测总体上提供了关于模型如何在训练数据集中的每个示例上执行的信息。这些信息可以用来训练模型来修正或改进这些预测。

首先，对每个感兴趣的基础模型执行 *k* 折叠交叉验证程序，并收集所有的折外预测。重要的是，对每个模型执行训练数据到 *k* 折叠的相同分割。现在，我们为每个模型都有一组聚合的样本外预测，例如训练数据集中每个示例的预测。

*   **基础模型**:使用 *k* 评估的模型在训练数据集上进行多重交叉验证，并且保留所有不符合的预测。

接下来，第二个更高阶的模型，称为元模型，在其他模型的预测上进行训练。当进行预测时，该元模型可以也可以不将每个示例的输入数据作为输入。这个模型的工作是学习如何最好地结合和纠正其他模型使用他们的超折叠预测所做的预测。

*   **元模型**:将一个或多个模型做出的超差预测作为输入，并展示如何最好地组合和校正预测的模型。

例如，我们可能有一个两类分类预测建模问题，并训练决策树和 k-最近邻模型作为基础模型。每个模型通过超范围预测为训练数据集中的每个示例预测 0 或 1。这些预测以及输入数据可以形成元模型的新输入。

*   **元模型输入**:给定样本的输入部分与每个基础模型所做的预测相连接。
*   **元模型输出**:给定样本的输出部分。

*为什么要用出格的预测来训练元模型？*

我们可以在整个训练数据集上训练每个基础模型，然后对训练数据集中的每个示例进行预测，并将预测用作元模型的输入。问题是预测将是乐观的，因为样本用于每个基础模型的训练。这种乐观的偏差意味着预测会比正常情况下更好，元模型可能不会从基础模型中学习到组合和校正预测所需的东西。

通过使用来自基础模型的折外预测来训练元模型，元模型可以在对看不见的数据进行操作时看到并利用每个基础模型的预期行为，正如在实践中使用集成对新数据进行预测时的情况一样。

最后，在整个训练数据集上训练每个基础模型，并且这些最终模型和元模型可以用于对新数据进行预测。这种集成的表现可以在单独的保持测试数据集上进行评估，该数据集在训练期间不使用。

该程序可总结如下:

*   1.对于每个基础模型:
    *   a.使用 k 倍交叉验证并收集不一致的预测。
    *   b.根据所有模型的不一致预测训练元模型。
    *   c.在整个训练数据集上训练每个基础模型。

这个过程被称为堆叠一般化，简称堆叠。因为通常使用线性加权和作为元模型，这个过程有时被称为 ***混合*** 。

有关堆叠主题的更多信息，请参见教程:

*   [如何用 Keras 开发 Python 深度学习神经网络的堆叠集成](https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/)
*   [如何用 Python 从零开始实现堆叠泛化(堆叠)](https://machinelearningmastery.com/implementing-stacking-scratch-python/)

我们可以使用上一节中使用的相同数据集，通过一个工作示例来具体实现这个过程。

首先，我们将数据分成训练和验证数据集。训练数据集将用于拟合子模型和元模型，验证数据集将从训练中保留下来，并在最后用于评估元模型和子模型。

```py
...
# split
X, X_val, y, y_val = train_test_split(X, y, test_size=0.33)
```

在这个例子中，我们将使用 k 倍交叉验证来拟合一个[决策树分类器](https://Sklearn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)和 [KNeighborsClassifier](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) 模型每个交叉验证折叠，并使用拟合模型来进行超折叠预测。

这些模型将预测概率而不是类别标签，试图为元模型提供更有用的输入特征。这是一个很好的做法。

我们还将跟踪折叠数据的输入数据(100 个特征)和输出数据(预期标签)。

```py
...
# collect out of sample predictions
data_x, data_y, knn_yhat, cart_yhat = list(), list(), list(), list()
kfold = KFold(n_splits=10, shuffle=True)
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	data_x.extend(test_X)
	data_y.extend(test_y)
	# fit and make predictions with cart
	model1 = DecisionTreeClassifier()
	model1.fit(train_X, train_y)
	yhat1 = model1.predict_proba(test_X)[:, 0]
	cart_yhat.extend(yhat1)
	# fit and make predictions with cart
	model2 = KNeighborsClassifier()
	model2.fit(train_X, train_y)
	yhat2 = model2.predict_proba(test_X)[:, 0]
	knn_yhat.extend(yhat2)
```

在运行结束时，我们可以为元分类器构建一个数据集，该数据集由输入数据的 100 个输入特征和来自 kNN 和决策树模型的两列预测概率组成。

下面的 *create_meta_dataset()* 函数实现了这一点，将折外的数据和跨折叠的预测作为输入，并为元模型构建输入数据集。

```py
# create a meta dataset
def create_meta_dataset(data_x, yhat1, yhat2):
	# convert to columns
	yhat1 = array(yhat1).reshape((len(yhat1), 1))
	yhat2 = array(yhat2).reshape((len(yhat2), 1))
	# stack as separate columns
	meta_X = hstack((data_x, yhat1, yhat2))
	return meta_X
```

然后我们可以调用这个函数为元模型准备数据。

```py
...
# construct meta dataset
meta_X = create_meta_dataset(data_x, knn_yhat, cart_yhat)
```

然后，我们可以在整个训练数据集中拟合每个子模型，以便在验证数据集中进行预测。

```py
...
# fit final submodels
model1 = DecisionTreeClassifier()
model1.fit(X, y)
model2 = KNeighborsClassifier()
model2.fit(X, y)
```

然后，我们可以在准备好的数据集上拟合元模型，在这种情况下，是[物流配送](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)模型。

```py
...
# construct meta classifier
meta_model = LogisticRegression(solver='liblinear')
meta_model.fit(meta_X, data_y)
```

最后，我们可以使用元模型对保持数据集进行预测。

这要求数据首先通过子模型，即用于构建元模型数据集的输出，然后元模型用于进行预测。我们将把所有这些装袋成一个名为 *stack_prediction()* 的函数，该函数获取将要进行预测的模型和数据。

```py
# make predictions with stacked model
def stack_prediction(model1, model2, meta_model, X):
	# make predictions
	yhat1 = model1.predict_proba(X)[:, 0]
	yhat2 = model2.predict_proba(X)[:, 0]
	# create input dataset
	meta_X = create_meta_dataset(X, yhat1, yhat2)
	# predict
	return meta_model.predict(meta_X)
```

然后，我们可以评估保持数据集上的子模型以供参考，然后使用元模型对保持数据集进行预测并进行评估。

我们预计，元模型在保持数据集上的表现将与任何单个子模型一样好或更好。如果不是这种情况，可以使用替代子模型或元模型来解决问题。

```py
...
# evaluate sub models on hold out dataset
acc1 = accuracy_score(y_val, model1.predict(X_val))
acc2 = accuracy_score(y_val, model2.predict(X_val))
print('Model1 Accuracy: %.3f, Model2 Accuracy: %.3f' % (acc1, acc2))
# evaluate meta model on hold out dataset
yhat = stack_prediction(model1, model2, meta_model, X_val)
acc = accuracy_score(y_val, yhat)
print('Meta Model Accuracy: %.3f' % (acc))
```

将这些结合在一起，完整的示例如下所示。

```py
# example of a stacked model for binary classification
from numpy import hstack
from numpy import array
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# create a meta dataset
def create_meta_dataset(data_x, yhat1, yhat2):
	# convert to columns
	yhat1 = array(yhat1).reshape((len(yhat1), 1))
	yhat2 = array(yhat2).reshape((len(yhat2), 1))
	# stack as separate columns
	meta_X = hstack((data_x, yhat1, yhat2))
	return meta_X

# make predictions with stacked model
def stack_prediction(model1, model2, meta_model, X):
	# make predictions
	yhat1 = model1.predict_proba(X)[:, 0]
	yhat2 = model2.predict_proba(X)[:, 0]
	# create input dataset
	meta_X = create_meta_dataset(X, yhat1, yhat2)
	# predict
	return meta_model.predict(meta_X)

# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
# split
X, X_val, y, y_val = train_test_split(X, y, test_size=0.33)
# collect out of sample predictions
data_x, data_y, knn_yhat, cart_yhat = list(), list(), list(), list()
kfold = KFold(n_splits=10, shuffle=True)
for train_ix, test_ix in kfold.split(X):
	# get data
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	data_x.extend(test_X)
	data_y.extend(test_y)
	# fit and make predictions with cart
	model1 = DecisionTreeClassifier()
	model1.fit(train_X, train_y)
	yhat1 = model1.predict_proba(test_X)[:, 0]
	cart_yhat.extend(yhat1)
	# fit and make predictions with cart
	model2 = KNeighborsClassifier()
	model2.fit(train_X, train_y)
	yhat2 = model2.predict_proba(test_X)[:, 0]
	knn_yhat.extend(yhat2)
# construct meta dataset
meta_X = create_meta_dataset(data_x, knn_yhat, cart_yhat)
# fit final submodels
model1 = DecisionTreeClassifier()
model1.fit(X, y)
model2 = KNeighborsClassifier()
model2.fit(X, y)
# construct meta classifier
meta_model = LogisticRegression(solver='liblinear')
meta_model.fit(meta_X, data_y)
# evaluate sub models on hold out dataset
acc1 = accuracy_score(y_val, model1.predict(X_val))
acc2 = accuracy_score(y_val, model2.predict(X_val))
print('Model1 Accuracy: %.3f, Model2 Accuracy: %.3f' % (acc1, acc2))
# evaluate meta model on hold out dataset
yhat = stack_prediction(model1, model2, meta_model, X_val)
acc = accuracy_score(y_val, yhat)
print('Meta Model Accuracy: %.3f' % (acc))
```

运行该示例首先报告决策树和 kNN 模型的准确性，然后报告元模型在保持数据集上的表现，这在训练期间是看不到的。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到元模型在两个子模型上都表现出色。

```py
Model1 Accuracy: 0.670, Model2 Accuracy: 0.930
Meta-Model Accuracy: 0.955
```

尝试一项消融性研究，仅用模型 1、模型 2 重新运行该示例，并且模型 1 和模型 2 都不作为元模型的输入，以确认来自子模型的预测实际上正在为元模型增加价值，这可能是有趣的。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)
*   [如何用 Keras 开发 Python 深度学习神经网络的堆叠集成](https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/)
*   [如何用 Python 从零开始实现堆叠泛化(堆叠)](https://machinelearningmastery.com/implementing-stacking-scratch-python/)
*   [如何在 Keras 创建深度学习模型的装袋集成](https://machinelearningmastery.com/how-to-create-a-random-split-cross-validation-and-bagging-ensemble-for-deep-learning-in-keras/)
*   [深度学习神经网络的集成学习方法](https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/)

### 书

*   [应用预测建模](https://amzn.to/32M80ta)，2013。

### 文章

*   [交叉验证(统计)，维基百科](https://en.wikipedia.org/wiki/Cross-validation_(statistics))。
*   一起学习，维基百科。

### 蜜蜂

*   [sklearn . dataset . make _ blobs API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)。
*   [sklearn.model_selection。KFold API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.KFold.html) 。
*   [sklearn . neighborsclassifier API](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)。
*   [硬化. tree .决策树分类器 API](https://Sklearn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) 。
*   [sklearn . metrics . accuracy _ score API](http://Sklearn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)。
*   [sklearn.linear_model。物流配送应用编程接口](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。
*   [sklearn . model _ selection . train _ test _ split API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。

## 摘要

在本教程中，您发现了机器学习中的异常预测。

具体来说，您了解到:

*   超差预测是对未用于训练模型的数据进行的一种超差预测。
*   在对看不见的数据进行预测时，超差预测最常用于估计模型的表现。
*   叠外预测可以用来构建一个集合模型，称为叠化概化或叠化集合。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。