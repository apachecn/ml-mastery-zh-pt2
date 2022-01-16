# 如何在机器学习中训练测试集

> 原文：<https://machinelearningmastery.com/train-to-the-test-set-in-machine-learning/>

对测试集的训练是一种过拟合，即准备一个模型，以增加泛化误差为代价，有意在给定的测试集上获得良好的表现。

这是一种在机器学习竞赛中常见的过拟合类型，其中提供了完整的训练数据集，并且只提供了测试集的输入部分。对测试集进行**训练的一种方法是构建一个与测试集最相似的训练集，然后将其用作训练模型的基础。该模型在测试集上的表现预计会更好，但在训练数据集和未来的任何新数据上的表现很可能会更差。**

虽然过拟合测试集是不可取的，但作为一个思维实验进行探索，并提供更多关于机器学习竞赛和避免过拟合的见解，可能会很有趣。

在本教程中，您将发现如何针对分类和回归问题有意识地训练测试集。

完成本教程后，您将知道:

*   对测试集的训练是机器学习竞赛中可能出现的一种数据泄露。
*   对测试集进行训练的一种方法是创建一个与所提供的测试集最相似的训练数据集。
*   如何使用 KNN 模型构建训练数据集，并用真实数据集训练测试集。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Train to the Test Set in Machine Learning](img/266ee73c1d1b5c819f73c2f8383b6c0e.png)

如何训练到机器学习中的测试集
图片由 [ND Strupler](https://www.flickr.com/photos/strupler/3804343976/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  训练到测试集
2.  训练到测试集进行分类
3.  训练回归测试集

## 训练到测试集

在应用机器学习中，我们寻求一种使用训练数据集学习输入和输出变量之间关系的模型。

我们的希望和目标是学习一种关系，这种关系可以推广到训练数据集之外的新例子。这个目标促使我们在对训练期间未使用的数据进行预测时，使用像 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)这样的重采样技术来估计模型的表现。

在机器学习竞赛的情况下，就像 Kaggle 上的竞赛一样，我们可以访问完整的训练数据集和测试数据集的输入，并且需要对测试数据集进行预测。

这导致了一种可能的情况，即我们可能会意外地或选择将模型训练到测试集。也就是说，调整模型行为以在测试数据集上获得最佳表现，而不是使用类似 k-fold 交叉验证的技术开发一个总体表现良好的模型。

> 另一种更明显的信息泄露途径，有时可以在机器学习比赛中看到，在比赛中训练和测试集数据同时给出。

—第 56 页，[特征工程和选择:预测模型的实用方法](https://amzn.to/2ylMqSW)，2019 年。

**训练到测试集往往是个坏主意**。

这是一种显式的数据泄露。然而，这是一个有趣的思想实验。

对测试集进行训练的一种方法是设计一个与测试集最相似的训练数据集。例如，我们可以丢弃训练集中与测试集差异太大的所有行，只在训练集中与测试集最相似的行上进行训练。

> 虽然测试集数据通常会使结果数据不可见，但只使用与测试集数据最相似的训练集样本就可以“训练测试”。这可能会很好地提高该特定测试集的模型表现分数，但可能会破坏在更广泛的数据集上进行预测的模型。

—第 56 页，[特征工程和选择:预测模型的实用方法](https://amzn.to/2ylMqSW)，2019 年。

我们会期望模型超过测试集，但这是这个思想实验的全部意义。

让我们探索这种方法来训练本教程中的测试集。

我们可以使用 k 近邻模型来选择那些与测试集最相似的训练集实例。[kneighgboresgressor](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)和[kneighgborsclassifier](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)都提供了[kneighgbors()函数](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.kneighbors)，该函数会将与给定数据(如测试集)最相似的行的索引返回到训练数据集中。

```py
...
# get the most similar neighbor for each point in the test set
neighbor_ix = knn.kneighbors(X_test, 2, return_distance=False)
ix = neighbor_ix[:,0]
```

我们可能需要尝试从选定的行索引中删除重复项。

```py
...
# remove duplicate rows
ix = unique(ix)
```

然后，我们可以使用这些行索引来构建定制的训练数据集并拟合模型。

```py
...
# create a training dataset from selected instances
X_train_neigh, y_train_neigh = X_train[ix], y_train[ix]
```

假设我们使用 KNN 模型从测试集构建训练集，我们也将使用相同类型的模型对测试集进行预测。这不是必需的，但它使示例更简单。

使用这种方法，我们现在可以对分类和回归数据集的测试集进行训练实验。

## 训练到测试集进行分类

我们将使用糖尿病数据集作为探索分类问题测试集训练的基础。

每份记录都描述了女性的医疗细节，预测是未来五年内糖尿病的发作。

*   [数据集详细信息:皮马-印第安人-糖尿病.名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)
*   [数据集:pima-印度人-diabetes.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)

数据集有 8 个输入变量和 768 行数据；输入变量都是数字，目标有两个类标签，例如，它是一个二进制分类任务。

下面提供了数据集前五行的示例。

```py
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
...
```

首先，我们可以直接从网址加载数据集，将其分成输入和输出元素，然后将数据集分成训练集和测试集，为测试集保留 30%的空间。然后，我们可以通过在训练集上训练 KNN 模型并在测试集上进行预测来评估具有默认模型超参数的模型。

下面列出了完整的示例。

```py
# example of evaluating a knn model on the diabetes classification dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
df = read_csv(url, header=None)
data = df.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define model
model = KNeighborsClassifier()
# fit model
model.fit(X_train, y_train)
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % (accuracy * 100))
```

运行该示例首先加载数据集并总结行数和列数，符合我们的预期。然后报告列车和测试集的形状，显示测试集中大约有 230 行。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

最后，该模型的分类准确率约为 77.056%。

```py
(768, 8) (768,)
(537, 8) (231, 8) (537,) (231,)
Accuracy: 77.056
```

现在，让我们看看通过准备一个直接为其训练的模型，我们是否能在测试集上获得更好的表现。

首先，我们将使用训练集中更简单的示例为测试集中的每一行构建一个训练数据集。

```py
...
# select examples that are most similar to the test set
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# get the most similar neighbor for each point in the test set
neighbor_ix = knn.kneighbors(X_test, 1, return_distance=False)
ix = neighbor_ix[:,0]
# create a training dataset from selected instances
X_train_neigh, y_train_neigh = X_train[ix], y_train[ix]
print(X_train_neigh.shape, y_train_neigh.shape)
```

接下来，我们将在这个新的数据集上训练模型，并像以前一样在测试集上评估它。

```py
...
# define model
model = KNeighborsClassifier()
# fit model
model.fit(X_train_neigh, y_train_neigh)
```

下面列出了完整的示例。

```py
# example of training to the test set for the diabetes dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
df = read_csv(url, header=None)
data = df.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# select examples that are most similar to the test set
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# get the most similar neighbor for each point in the test set
neighbor_ix = knn.kneighbors(X_test, 1, return_distance=False)
ix = neighbor_ix[:,0]
# create a training dataset from selected instances
X_train_neigh, y_train_neigh = X_train[ix], y_train[ix]
print(X_train_neigh.shape, y_train_neigh.shape)
# define model
model = KNeighborsClassifier()
# fit model
model.fit(X_train_neigh, y_train_neigh)
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % (accuracy * 100))
```

运行该示例，我们可以看到新训练数据集的报告大小与测试集的大小相同，正如我们预期的那样。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

我们可以看到，与在整个训练数据集上训练模型相比，通过训练测试集，我们已经实现了表现的提升。在这种情况下，我们实现了大约 79.654%的分类准确率，而在使用整个训练数据集时，分类准确率为 77.056%。

```py
(768, 8) (768,)
(537, 8) (231, 8) (537,) (231,)
(231, 8) (231,)
Accuracy: 79.654
```

您可能希望尝试从训练集中为测试集中的每个示例选择不同数量的邻居，看看是否可以获得更好的表现。

此外，您可能想尝试在训练集中保留唯一的行索引，看看这是否有所不同。

最后，保留最终验证数据集并比较不同的“训练到测试集”技术如何影响保持数据集的表现可能会很有趣。例如，查看对测试集的训练如何影响泛化错误。

在下面的评论中报告你的发现。

现在我们知道如何训练测试集进行分类，让我们看一个回归的例子。

## 训练回归测试集

我们将使用房屋数据集作为探索回归问题测试集训练的基础。

房屋数据集涉及房屋及其附近地区的详细信息，以千美元为单位预测房价。

*   [数据集详细信息:房屋名称](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)
*   [数据集:housing.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)

这是一个回归问题，意味着我们预测的是一个数值。有 506 个观测值，有 13 个输入变量和一个输出变量。

下面列出了前五行的示例。

```py
0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98,24.00
0.02731,0.00,7.070,0,0.4690,6.4210,78.90,4.9671,2,242.0,17.80,396.90,9.14,21.60
0.02729,0.00,7.070,0,0.4690,7.1850,61.10,4.9671,2,242.0,17.80,392.83,4.03,34.70
0.03237,0.00,2.180,0,0.4580,6.9980,45.80,6.0622,3,222.0,18.70,394.63,2.94,33.40
0.06905,0.00,2.180,0,0.4580,7.1470,54.20,6.0622,3,222.0,18.70,396.90,5.33,36.20
...
```

首先，我们可以加载数据集，分割它，并使用整个训练数据集直接在其上评估 KNN 模型。我们将使用平均绝对误差(MAE)报告这个回归类的表现。

下面列出了完整的示例。

```py
# example of evaluating a knn model on the housing regression dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
data = df.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define model
model = KNeighborsRegressor()
# fit model
model.fit(X_train, y_train)
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

运行该示例首先加载数据集并总结行数和列数，符合我们的预期。然后报告列车和测试集的形状，显示测试集中大约有 150 行。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

最后，模型的 MAE 据报道约为 4.488。

```py
(506, 13) (506,)
(354, 13) (152, 13) (354,) (152,)
MAE: 4.488
```

现在，让我们看看我们是否可以通过准备一个经过训练的模型来在测试集上获得更好的表现。

首先，我们将使用训练集中更简单的示例为测试集中的每一行构建一个训练数据集。

```py
...
# select examples that are most similar to the test set
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# get the most similar neighbor for each point in the test set
neighbor_ix = knn.kneighbors(X_test, 1, return_distance=False)
ix = neighbor_ix[:,0]
# create a training dataset from selected instances
X_train_neigh, y_train_neigh = X_train[ix], y_train[ix]
print(X_train_neigh.shape, y_train_neigh.shape)
```

接下来，我们将在这个新的数据集上训练模型，并像以前一样在测试集上评估它。

```py
...
# define model
model = KNeighborsClassifier()
# fit model
model.fit(X_train_neigh, y_train_neigh)
```

下面列出了完整的示例。

```py
# example of training to the test set for the housing dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
data = df.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)
# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# select examples that are most similar to the test set
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
# get the most similar neighbor for each point in the test set
neighbor_ix = knn.kneighbors(X_test, 1, return_distance=False)
ix = neighbor_ix[:,0]
# create a training dataset from selected instances
X_train_neigh, y_train_neigh = X_train[ix], y_train[ix]
print(X_train_neigh.shape, y_train_neigh.shape)
# define model
model = KNeighborsRegressor()
# fit model
model.fit(X_train_neigh, y_train_neigh)
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

运行该示例，我们可以看到新训练数据集的报告大小与测试集的大小相同，正如我们预期的那样。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

我们可以看到，与在整个训练数据集上训练模型相比，通过训练测试集，我们已经实现了表现的提升。在这种情况下，与使用整个训练数据集时的 4.488 相比，我们获得了大约 4.433 的 MAE。

同样，在构建新的训练集时，您可能希望探索使用不同数量的邻居，看看在训练数据集中保留唯一的行是否有所不同。在下面的评论中报告你的发现。

```py
(506, 13) (506,)
(354, 13) (152, 13) (354,) (152,)
(152, 13) (152,)
MAE: 4.433
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [特征工程和选择:预测模型的实用方法](https://amzn.to/2ylMqSW)，2019。

### 蜜蜂

*   [sklearn . neighborsrgressor API](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)。
*   [sklearn . neighborsclassifier API](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)。

## 摘要

在本教程中，您发现了如何针对分类和回归问题有意识地训练测试集。

具体来说，您了解到:

*   对测试集的训练是机器学习竞赛中可能出现的一种数据泄露。
*   对测试集进行训练的一种方法是创建一个与所提供的测试集最相似的训练数据集。
*   如何使用 KNN 模型构建训练数据集，并用真实数据集训练测试集。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。