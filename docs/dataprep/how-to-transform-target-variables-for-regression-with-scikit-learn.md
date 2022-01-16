# 如何在 Python 中转换回归的目标变量

> 原文：<https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-Sklearn/>

最后更新于 2020 年 10 月 1 日

数据准备是应用机器学习的一大部分。

正确准备您的训练数据可能意味着平庸和非凡的结果之间的区别，即使使用非常简单的线性算法。

对于输入变量来说，执行数据准备操作(如缩放)相对简单，并且已经通过 Pipeline Sklearn 类在 Python 中成为常规操作。

在必须预测数值的回归预测建模问题上，对目标变量进行缩放和执行其他数据转换也很关键。这可以在 Python 中使用**transformed targetgressor**类来实现。

在本教程中，您将发现如何使用 TransformedTargetRegressor 来使用 Sklearn Python 机器学习库缩放和转换回归的目标变量。

完成本教程后，您将知道:

*   缩放输入和目标数据对机器学习的重要性。
*   将数据转换应用于目标变量的两种方法。
*   如何在真实回归数据集上使用 TransformedTargetRegressor。

**用我的新书[机器学习的数据准备](https://machinelearningmastery.com/data-preparation-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Transform Target Variables for Regression With Sklearn](img/fd409ff60bfb607708b65a722f09ed46.png)

如何用 Scikit 转换回归的目标变量-学习
图片由[唐·海尼斯](https://flickr.com/photos/kiskadee_3/37926034661/)提供，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  数据缩放的重要性
2.  如何缩放目标变量
3.  使用 TransformedTargetRegressor 的示例

## 数据缩放的重要性

在不同变量之间，数值范围不同的数据是很常见的。

例如，一个变量可能以英尺为单位，另一个以米为单位，等等。

如果所有变量都缩放到相同的范围，一些机器学习算法的表现会好得多，例如将所有变量缩放到 0 到 1 之间的值，称为归一化。

这影响了使用输入加权和的算法，如线性模型和神经网络，以及使用距离度量的模型，如支持向量机和 k 近邻。

因此，缩放输入数据是一种很好的做法，甚至可以尝试其他数据变换，例如使用幂变换使数据更正常(更好地符合[高斯概率分布](https://machinelearningmastery.com/continuous-probability-distributions-for-machine-learning/))。

这也适用于称为目标变量的输出变量，例如在建模回归预测建模问题时预测的数值。

对于回归问题，通常需要缩放或转换输入和目标变量。

缩放输入变量很简单。在 Sklearn 中，您可以手动使用缩放对象，或者使用更方便的 Pipeline，它允许您在使用模型之前将一系列数据转换对象链接在一起。

[管道](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)将为您拟合训练数据上的比例对象，并将变换应用于新数据，例如当使用模型进行预测时。

例如:

```py
...
# prepare the model with input scaling
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', LinearRegression())])
# fit pipeline
pipeline.fit(train_x, train_y)
# make predictions
yhat = pipeline.predict(test_x)
```

挑战在于，Sklearn 中缩放目标变量的等效机制是什么？

## 如何缩放目标变量

有两种方法可以缩放目标变量。

第一种是手动管理转换，第二种是使用新的自动方式管理转换。

1.  手动转换目标变量。
2.  自动转换目标变量。

### 1.目标变量的手动转换

手动管理目标变量的缩放涉及手动创建缩放对象并将其应用于数据。

它包括以下步骤:

1.  创建变换对象，例如最小最大缩放器。
2.  在训练数据集上拟合变换。
3.  将转换应用于训练和测试数据集。
4.  反转任何预测的变换。

例如，如果我们想要规范化一个目标变量，我们将首先定义并训练一个[最小最大缩放器对象](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html):

```py
...
# create target scaler object
target_scaler = MinMaxScaler()
target_scaler.fit(train_y)
```

然后，我们将转换列车并测试目标变量数据。

```py
...
# transform target variables
train_y = target_scaler.transform(train_y)
test_y = target_scaler.transform(test_y)
```

然后我们将拟合我们的模型，并使用该模型进行预测。

在用误差度量来使用或评估预测之前，我们必须反转变换。

```py
...
# invert transform on predictions
yhat = model.predict(test_X)
yhat = target_scaler.inverse_transform(yhat)
```

这是一个痛点，因为这意味着您不能使用 Sklearn 中的便利功能，如 [cross_val_score()](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) 来快速评估模型。

### 2.目标变量的自动转换

另一种方法是自动管理变换和逆变换。

这可以通过使用包装给定模型和缩放对象的[transformed targetgressior](https://Sklearn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html)对象来实现。

它将使用用于拟合模型的相同训练数据准备目标变量的变换，然后对调用 predict()时提供的任何新数据应用该逆变换，以正确的比例返回预测。

要使用 TransformedTargetRegressor，它是通过指定要在目标上使用的模型和转换对象来定义的；例如:

```py
...
# define the target transform wrapper
wrapped_model = TransformedTargetRegressor(regressor=model, transformer=MinMaxScaler())
```

稍后，TransformedTargetRegressor 实例可以像任何其他模型一样通过调用 fit()函数进行拟合，并通过调用 predict()函数进行预测。

```py
...
# use the target transform wrapper
wrapped_model.fit(train_X, train_y)
yhat = wrapped_model.predict(test_X)
```

这要容易得多，并且允许您使用像 *cross_val_score()* 这样的有用函数来评估模型

现在我们已经熟悉了 TransformedTargetRegressor，让我们看一个在真实数据集上使用它的例子。

## 使用 TransformedTargetRegressor 的示例

在本节中，我们将演示如何在真实数据集上使用 TransformedTargetRegressor。

我们将使用波士顿住房回归问题，该问题有 13 个输入和一个数字目标，需要学习郊区特征和房价之间的关系。

数据集可以从这里下载:

*   [波士顿住房数据集(housing.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)

下载数据集，保存在当前工作目录下，名称为“ *housing.csv* ”。

在数据集中，您应该看到所有变量都是数字。

```py
0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98,24.00
0.02731,0.00,7.070,0,0.4690,6.4210,78.90,4.9671,2,242.0,17.80,396.90,9.14,21.60
0.02729,0.00,7.070,0,0.4690,7.1850,61.10,4.9671,2,242.0,17.80,392.83,4.03,34.70
0.03237,0.00,2.180,0,0.4580,6.9980,45.80,6.0622,3,222.0,18.70,394.63,2.94,33.40
0.06905,0.00,2.180,0,0.4580,7.1470,54.20,6.0622,3,222.0,18.70,396.90,5.33,36.20
...
```

您可以在此了解有关此数据集和列含义的更多信息:

*   [波士顿房屋数据详情(房屋.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

我们可以确认数据集可以作为 NumPy 数组正确加载，并将其拆分为输入和输出变量。

下面列出了完整的示例。

```py
# load and summarize the dataset
from numpy import loadtxt
# load data
dataset = loadtxt('housing.csv', delimiter=",")
# split into inputs and outputs
X, y = dataset[:, :-1], dataset[:, -1]
# summarize dataset
print(X.shape, y.shape)
```

运行该示例会打印数据集输入和输出部分的形状，显示 13 个输入变量、一个输出变量和 506 行数据。

```py
(506, 13) (506,)
```

我们现在可以准备一个使用 TransformedTargetRegressor 的例子。

预测该问题目标平均值的[朴素回归](https://Sklearn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)模型可以达到约 6.659 的[平均绝对误差](https://Sklearn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) (MAE)。我们将力争做得更好。

在这个例子中，我们将拟合一个[huberrelater](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)对象，并使用一个管道来规范化输入变量。

```py
...
# prepare the model with input scaling
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', HuberRegressor())])
```

接下来，我们将定义一个 TransformedTargetRegressor 实例，并将回归器设置为管道，将转换器设置为一个 MinMaxScaler 对象的实例。

```py
...
# prepare the model with target scaling
model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
```

然后，我们可以使用 10 倍交叉验证对输入和输出变量进行标准化来评估模型。

```py
...
# evaluate model
cv = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
```

将这些结合在一起，完整的示例如下所示。

```py
# example of normalizing input and output variables for regression.
from numpy import mean
from numpy import absolute
from numpy import loadtxt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
# load data
dataset = loadtxt('housing.csv', delimiter=",")
# split into inputs and outputs
X, y = dataset[:, :-1], dataset[:, -1]
# prepare the model with input scaling
pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', HuberRegressor())])
# prepare the model with target scaling
model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
# evaluate model
cv = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# convert scores to positive
scores = absolute(scores)
# summarize the result
s_mean = mean(scores)
print('Mean MAE: %.3f' % (s_mean))
```

运行该示例使用输入和输出变量的规范化来评估模型。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们实现了大约 3.1 的 MAE，比实现了大约 6.6 的天真模型好得多。

```py
Mean MAE: 3.191
```

我们不限于使用缩放对象；例如，我们还可以探索对目标变量使用其他数据变换，如 [PowerTransformer](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) ，可以使每个变量更像高斯(使用 [Yeo-Johnson 变换](https://en.wikipedia.org/wiki/Power_transform))并提高线性模型的表现。

默认情况下，PowerTransformer 还会在执行转换后对每个变量进行标准化。

下面列出了在房屋数据集的输入和目标变量上使用电力变压器的完整示例。

```py
# example of power transform input and output variables for regression.
from numpy import mean
from numpy import absolute
from numpy import loadtxt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
# load data
dataset = loadtxt('housing.csv', delimiter=",")
# split into inputs and outputs
X, y = dataset[:, :-1], dataset[:, -1]
# prepare the model with input scaling
pipeline = Pipeline(steps=[('power', PowerTransformer()), ('model', HuberRegressor())])
# prepare the model with target scaling
model = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())
# evaluate model
cv = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# convert scores to positive
scores = absolute(scores)
# summarize the result
s_mean = mean(scores)
print('Mean MAE: %.3f' % (s_mean))
```

运行该示例使用输入和输出变量的幂变换来评估模型。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们看到 MAE 进一步提高到大约 2.9。

```py
Mean MAE: 2.926
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 应用程序接口

*   [回归 Sklearn API 中的转化目标。](https://Sklearn.org/stable/modules/compose.html#transforming-target-in-regression)
*   [硬化。化合物。转化银螯合物 API。](https://Sklearn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html)
*   [硬化。预处理。MinMaxScaler API。](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
*   [硬化。预处理。PowerTransformer API。](https://Sklearn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)

### 资料组

*   [波士顿住房数据集(housing.csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)
*   [波士顿房屋数据详情(房屋.名称)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

## 摘要

在本教程中，您发现了如何在 Sklearn 中使用 TransformedTargetRegressor 来缩放和转换回归的目标变量。

具体来说，您了解到:

*   缩放输入和目标数据对机器学习的重要性。
*   将数据转换应用于目标变量的两种方法。
*   如何在真实回归数据集上使用 TransformedTargetRegressor。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。