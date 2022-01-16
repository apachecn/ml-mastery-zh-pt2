# Python 中的 4 种自动异常值检测算法

> 原文：<https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/>

最后更新于 2020 年 8 月 17 日

分类或回归数据集中异常值的存在会导致拟合不佳和预测建模表现降低。

在给定大量输入变量的情况下，对于大多数机器学习数据集来说，用简单的统计方法识别和**去除异常值**是具有挑战性的。相反，可以在建模管道中使用自动异常值检测方法并进行比较，就像可以应用于数据集的其他数据准备转换一样。

在本教程中，您将发现如何使用自动异常值检测和移除来提高机器学习预测建模表现。

完成本教程后，您将知道:

*   自动异常值检测模型提供了一种替代统计技术的方法，该技术具有大量输入变量，这些变量具有复杂且未知的相互关系。
*   如何正确地对训练数据集应用自动异常值检测和去除，避免数据泄露？
*   如何评估和比较预测建模管道和从训练数据集中移除的异常值。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Model-Based Outlier Detection and Removal in Python](img/e6cc2f4cf55a023c1f9f6e1a460ea955.png)

Python 中基于模型的异常值检测和移除
图片由[佐尔坦·沃尔斯](https://www.flickr.com/photos/94941635@N07/16120767979/)提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  异常值的检测和去除
2.  数据集和表现基线
    1.  房价回归数据集
    2.  基线模型表现
3.  自动异常值检测
    1.  隔离林
    2.  最小协方差行列式
    3.  局部异常因子
    4.  一等 SVM

## 异常值的检测和去除

[异常值](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)是数据集中不符合某种方式的观测值。

也许最常见或最熟悉的异常值类型是远离其余观测值或观测值重心的观测值。

当我们有一个或两个变量时，这很容易理解，我们可以将数据可视化为直方图或散点图，尽管当我们有许多定义高维输入特征空间的输入变量时，这变得非常具有挑战性。

在这种情况下，用于识别异常值的简单统计方法可能会失效，例如使用标准差或四分位数范围的方法。

在训练机器学习算法进行预测建模时，从数据中识别和移除异常值可能很重要。

异常值会偏斜统计度量和数据分布，对底层数据和关系提供误导性的表示。在建模之前从训练数据中去除异常值可以使数据更好地匹配，进而产生更巧妙的预测。

谢天谢地，有各种基于模型的自动方法来识别输入数据中的异常值。重要的是，每种方法对异常值的定义略有不同，提供了准备可评估和比较的训练数据集的替代方法，就像建模管道中的任何其他数据准备步骤一样。

在我们深入研究自动异常值检测方法之前，让我们首先选择一个标准的机器学习数据集，作为我们研究的基础。

## 数据集和表现基线

在本节中，我们将首先选择一个标准的机器学习数据集，并在该数据集上建立表现基线。

这将为下一节探索数据准备的异常值识别和移除方法提供背景。

### 房价回归数据集

我们将使用房价回归数据集。

这个数据集有 13 个输入变量，描述了房子和郊区的属性，并要求以千美元为单位预测郊区房子的中值。

您可以在此了解有关数据集的更多信息:

*   [房价数据集(housing.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
*   [房价数据集描述(房屋.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

不需要下载数据集，因为我们将自动下载它作为我们工作示例的一部分。

打开数据集并查看原始数据。下面列出了前几行数据。

我们可以看到，这是一个带有数值输入变量的回归预测建模问题，每个变量都有不同的尺度。

```py
0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98,24.00
0.02731,0.00,7.070,0,0.4690,6.4210,78.90,4.9671,2,242.0,17.80,396.90,9.14,21.60
0.02729,0.00,7.070,0,0.4690,7.1850,61.10,4.9671,2,242.0,17.80,392.83,4.03,34.70
0.03237,0.00,2.180,0,0.4580,6.9980,45.80,6.0622,3,222.0,18.70,394.63,2.94,33.40
0.06905,0.00,2.180,0,0.4580,7.1470,54.20,6.0622,3,222.0,18.70,396.90,5.33,36.20
...
```

数据集有许多数值输入变量，这些变量具有未知且复杂的关系。我们不知道这个数据集中是否存在异常值，尽管我们可能猜测可能存在一些异常值。

下面的示例加载数据集并将其拆分为输入和输出列，将其拆分为训练和测试数据集，然后总结数据数组的形状。

```py
# load and summarize the dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# summarize the shape of the dataset
print(X.shape, y.shape)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the train and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

运行该示例，我们可以看到数据集被正确加载，并且有 506 行数据，包含 13 个输入变量和一个目标变量。

数据集被分成训练集和测试集，其中 339 行用于模型训练，167 行用于模型评估。

```py
(506, 13) (506,)
(339, 13) (167, 13) (339,) (167,)
```

接下来，让我们在这个数据集上评估一个模型，并建立一个表现基线。

### 基线模型表现

这是一个回归预测建模问题，意味着我们将预测一个数值。所有输入变量也是数字。

在这种情况下，我们将拟合线性回归算法，并通过在测试数据集上训练模型并对测试数据进行预测来评估模型表现，并使用平均绝对误差(MAE)来评估预测。

下面列出了在数据集上评估线性回归模型的完整示例。

```py
# evaluate model on the raw dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

运行该示例适合并评估模型，然后报告 MAE。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型实现了大约 3.417 的 MAE。这提供了一个表现基线，我们可以据此比较不同的异常值识别和去除程序。

```py
MAE: 3.417
```

接下来，我们可以尝试从训练数据集中移除异常值。

## 自动异常值检测

Sklearn 库提供了许多内置的自动方法来识别数据中的异常值。

在本节中，我们将回顾四种方法，并比较它们在房价数据集上的表现。

每个方法都将被定义，然后适合训练数据集。然后，拟合模型将预测训练数据集中哪些示例是异常值，哪些不是(所谓的内联值)。然后将从训练数据集中移除异常值，然后将模型拟合到剩余的示例中，并在整个测试数据集中进行评估。

在整个训练数据集上拟合异常值检测方法是无效的，因为这会导致[数据泄露](https://machinelearningmastery.com/data-leakage-machine-learning/)。也就是说，模型可以访问测试集中未用于训练模型的数据(或关于数据的信息)。这可能导致对模型表现的乐观估计。

在进行预测之前，我们可以尝试检测“*新数据*”上的异常值，例如测试集，但是如果检测到异常值，我们该怎么办？

一种方法可能是返回“ *None* ”，表示模型无法对这些异常情况做出预测。这可能是一个有趣的扩展，适合您的项目。

### 隔离林

隔离森林，简称 iForest，是一种基于树的异常检测算法。

它基于对正常数据进行建模，从而隔离特征空间中数量少且不同的异常。

> ……我们提出的方法利用了两个异常的数量属性:I)它们是由较少实例组成的少数，以及 ii)它们具有与正常实例非常不同的属性值。

——[隔离森林](https://ieeexplore.ieee.org/abstract/document/4781136)，2008 年。

Sklearn 库在 [IsolationForest 类](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)中提供了隔离林的实现。

也许模型中最重要的超参数是“*污染*”参数，该参数用于帮助估计数据集中异常值的数量。这是一个介于 0.0 和 0.5 之间的值，默认设置为 0.1。

```py
...
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)
```

一旦识别出来，我们就可以从训练数据集中移除异常值。

```py
...
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
```

将这些联系在一起，下面列出了在房屋数据集上评估线性模型的完整示例，使用隔离林识别并移除异常值。

```py
# evaluate model performance with outliers removed using isolation forest
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

运行该示例适合并评估模型，然后报告 MAE。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，该模型识别并移除了 34 个异常值，并获得了约 3.189 的 MAE，这比得分约为 3.417 的基线有所改善。

```py
(339, 13) (339,)
(305, 13) (305,)
MAE: 3.189
```

### 最小协方差行列式

如果输入变量具有高斯分布，则可以使用简单的统计方法来检测异常值。

例如，如果数据集有两个输入变量，并且都是高斯的，那么特征空间形成多维高斯，并且可以使用这种分布的知识来识别远离该分布的值。

这种方法可以通过定义一个覆盖正常数据的超球(椭球)来推广，超出这个形状的数据被认为是异常值。对于多变量数据，这种技术的有效实现被称为最小协方差行列式，简称 MCD。

> 最小协方差行列式(MCD)方法是一种高度稳健的多元定位和散射估计方法，其快速算法是可用的。[……]它也是一个方便有效的异常值检测工具。

——[最小协方差行列式与延拓](https://arxiv.org/abs/1709.07045)，2017。

Sklearn 库通过[椭圆包络类](https://Sklearn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html)提供对该方法的访问。

它提供了“*污染*”参数，该参数定义了在实践中观察到的异常值的预期比率。在这种情况下，我们将它设置为 0.01 的值，通过一点点反复试验发现。

```py
...
# identify outliers in the training dataset
ee = EllipticEnvelope(contamination=0.01)
yhat = ee.fit_predict(X_train)
```

一旦识别出异常值，就可以像我们在前面的示例中所做的那样，从训练数据集中移除异常值。

将这些联系在一起，下面列出了使用椭圆包络(最小协变行列式)方法从房屋数据集中识别和移除异常值的完整示例。

```py
# evaluate model performance with outliers removed using elliptical envelope
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mean_absolute_error
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
ee = EllipticEnvelope(contamination=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

运行该示例适合并评估模型，然后报告 MAE。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到椭圆包络方法仅识别并移除了 4 个异常值，导致 MAE 从基线的 3.417 下降到 3.388。

```py
(339, 13) (339,)
(335, 13) (335,)
MAE: 3.388
```

### 局部异常因子

识别异常值的一个简单方法是定位那些在特征空间中远离其他示例的示例。

这对于低维度(很少特征)的特征空间可以很好地工作，尽管随着特征数量的增加，它会变得不那么可靠，这被称为维度的诅咒。

局部离群因子，简称 LOF，是一种试图利用最近邻概念进行异常值检测的技术。每个例子都被分配了一个分数，根据其局部邻域的大小来衡量孤立的程度或异常值出现的可能性。得分最高的例子更有可能是异常值。

> 我们为数据集中的每个对象引入一个局部离群值(LOF)，指示其离群程度。

——[LOF:识别基于密度的局部异常值](https://dl.acm.org/citation.cfm?id=335388)，2000 年。

Sklearn 库在[localhoutlierfactor 类](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)中提供了这种方法的实现。

该模型提供“*污染*”参数，即数据集中异常值的预期百分比，被指示并默认为 0.1。

```py
...
# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
```

将这些联系在一起，下面列出了使用局部异常因子方法从住房数据集中识别和移除异常值的完整示例。

```py
# evaluate model performance with outliers removed using local outlier factor
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

运行该示例适合并评估模型，然后报告 MAE。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到局部离群因子方法识别并移除了 34 个离群值，与隔离林的数量相同，导致 MAE 从基线的 3.417 下降到 3.356。更好，但不如隔离森林，这表明发现并移除了一组不同的异常值。

```py
(339, 13) (339,)
(305, 13) (305,)
MAE: 3.356
```

### 一等 SVM

最初为二进制分类开发的[支持向量机](https://machinelearningmastery.com/support-vector-machines-for-machine-learning/)或 SVM 算法可用于单类分类。

当对一个类建模时，该算法捕获多数类的密度，并将密度函数极值上的例子分类为异常值。SVM 的这种修改被称为一级 SVM。

> …一种计算二进制函数的算法，该二进制函数应该捕获输入空间中概率密度所在的区域(它的支持)，也就是说，一个函数使得大部分数据将位于该函数非零的区域。

——[估计高维分布的支持](https://dl.acm.org/citation.cfm?id=1119749)，2001。

虽然 SVM 是一种分类算法，而一类 SVM 也是一种分类算法，但它可以用于发现回归和类别数据集的输入数据中的异常值。

Sklearn 库在 [OneClassSVM 类](https://Sklearn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)中提供了一个一类 SVM 的实现。

该类提供了“ *nu* ”参数，该参数指定了数据集中异常值的近似比率，默认为 0.1。在这种情况下，我们将它设置为 0.01，稍微试错一下就发现了。

```py
...
# identify outliers in the training dataset
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)
```

将这些联系在一起，下面列出了使用一类 SVM 方法从住房数据集中识别和移除异常值的完整示例。

```py
# evaluate model performance with outliers removed using one class SVM
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(url, header=None)
# retrieve the array
data = df.values
# split into input and output elements
X, y = data[:, :-1], data[:, -1]
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
ee = OneClassSVM(nu=0.01)
yhat = ee.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
```

运行该示例适合并评估模型，然后报告 MAE。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到只有三个异常值被识别和移除，并且模型实现了大约 3.431 的 MAE，这并不比实现 3.417 的基线模型更好。也许通过更多的调整可以获得更好的表现。

```py
(339, 13) (339,)
(336, 13) (336,)
MAE: 3.431
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 相关教程

*   [不平衡数据集的单类分类算法](https://machinelearningmastery.com/one-class-classification-algorithms/)
*   [如何去除机器学习的异常值](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)

### 报纸

*   [隔离森林](https://ieeexplore.ieee.org/abstract/document/4781136)，2008 年。
*   [最小协方差行列式与延拓](https://arxiv.org/abs/1709.07045)，2017。
*   [LOF:识别基于密度的局部异常值](https://dl.acm.org/citation.cfm?id=335388)，2000。
*   [估计高维分布的支持](https://dl.acm.org/citation.cfm?id=1119749)，2001。

### 蜜蜂

*   [新奇和异常检测，Sklearn 用户指南](https://Sklearn.org/stable/modules/outlier_detection.html)。
*   [硬化。协方差。椭圆包络 API](https://Sklearn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html) 。
*   [硬化. svm.OneClassSVM API](https://Sklearn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) 。
*   [硬化。邻居。局部外显性因子 API](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) 。
*   [硬化。一起。绝缘林 API](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) 。

## 摘要

在本教程中，您发现了如何使用自动异常值检测和移除来提高机器学习预测建模表现。

具体来说，您了解到:

*   自动异常值检测模型提供了一种替代统计技术的方法，该技术具有大量输入变量，这些变量具有复杂且未知的相互关系。
*   如何正确地对训练数据集应用自动异常值检测和去除，避免数据泄露？
*   如何评估和比较预测建模管道和从训练数据集中移除的异常值。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。