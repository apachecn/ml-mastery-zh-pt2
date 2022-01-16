# 如何用 Python 开发多输出回归模型

> 原文：<https://machinelearningmastery.com/multi-output-regression-models-with-python/>

最后更新于 2021 年 4 月 27 日

多输出回归是一种回归问题，在给定输入示例的情况下，涉及预测两个或多个数值。

一个例子可能是预测给定输入的坐标，例如预测 x 和 y 值。另一个例子是多步时间序列预测，包括预测给定变量的多个未来时间序列。

许多机器学习算法被设计用于预测单个数值，简称为回归。有些算法本质上支持多输出回归，如线性回归和决策树。还有一些特殊的变通模型可以用来包装和使用那些不支持预测多个输出的算法。

在本教程中，您将发现如何为多输出回归开发机器学习模型。

完成本教程后，您将知道:

*   机器学习中的多输出回归问题。
*   如何开发内在支持多输出回归的机器学习模型？
*   如何开发包装器模型，允许不支持多输出的算法用于多输出回归。

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 8 月更新**:包装器模型的详细示例。

![How to Develop Multioutput Regression Models in Python](img/393b3c6c732ff572379f6d3f4e96bd30.png)

如何在 Python 中开发多输出回归模型
图片由 [a_terracini](https://flickr.com/photos/arterracini/32096684665/) 提供，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  多输出回归问题
    1.  检查 Scikit-学习版本
    2.  多输出回归测试问题
2.  固有多输出回归算法
    1.  多输出回归的线性回归
    2.  多输出回归的 k 近邻
    3.  使用交叉验证评估多输出回归
3.  包装多输出回归算法
4.  直接多输出回归
5.  链式多输出回归

## 多输出回归问题

回归是指涉及预测数值的预测建模问题。

例如，预测尺寸、重量、数量、销售数量和点击数量是回归问题。通常，给定输入变量，预测单个数值。

一些回归问题需要预测两个或更多的数值。例如，预测 x 和 y 坐标。

这些问题被称为多输出回归或多输出回归。

*   **回归**:预测给定输入的单个数值输出。
*   **多输出回归**:给定一个输入，预测两个或多个数值输出。

在多输出回归中，输出通常依赖于输入并且相互依赖。这意味着输出通常不是相互独立的，可能需要一个模型来一起预测两个输出，或者每个输出取决于其他输出。

多步时间序列预测可以被认为是一种多输出回归，其中预测一系列未来值，并且每个预测值取决于该序列中的先前值。

有许多处理多输出回归的策略，我们将在本教程中探讨其中的一些。

### 检查套件-学习版本

首先，确认您安装了 Sklearn 库的现代版本。

这很重要，因为我们将在本教程中探索的一些模型需要一个现代版本的库。

您可以使用下面的代码示例检查库的版本:

```py
# check Sklearn version
import sklearn
print(sklearn.__version__)
```

运行该示例将打印库的版本。

在撰写本文时，这大约是 0.22 版本。您需要使用 Sklearn 或更高版本。

```py
0.22.1
```

### 多输出回归测试问题

我们可以定义一个测试问题，用来演示不同的建模策略。

我们将使用[make _ revolution()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)创建一个多输出回归的测试数据集。我们将生成 1000 个具有 10 个输入特征的示例，其中五个是冗余的，五个是信息丰富的。这个问题需要预测两个数值。

*   **问题输入** : 10 个数字变量。
*   **问题输出** : 2 个数值变量。

下面的示例生成数据集并总结形状。

```py
# example of multioutput regression test problem
from sklearn.datasets import make_regression
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# summarize dataset
print(X.shape, y.shape)
```

运行该示例会创建数据集，并总结数据集的输入和输出元素的形状以进行建模，从而确认所选的配置。

```py
(1000, 10) (1000, 2)
```

接下来，我们直接来看建模这个问题。

## 固有多输出回归算法

一些回归机器学习算法直接支持多输出。

这包括 Sklearn 库中实现的大多数流行的机器学习算法，例如:

*   linearregrecession _ 及相关)
*   KNeighborsRegressor
*   决策树回归器
*   随机森林回归器(及相关)

让我们看几个具体的例子。

### 多输出回归的线性回归

以下示例在多输出回归数据集上拟合线性回归模型，然后使用拟合模型进行单次预测。

```py
# linear regression for multioutput regression
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define model
model = LinearRegression()
# fit model
model.fit(X, y)
# make a prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = model.predict([row])
# summarize prediction
print(yhat[0])
```

运行该示例符合模型，然后对一个输入进行预测，确认模型预测了两个所需值。

```py
[-11.73511093  52.78406297]
```

### 多输出回归的 k 近邻

以下示例在多输出回归数据集上拟合 k 近邻模型，然后用拟合模型进行单次预测。

```py
# k-nearest neighbors for multioutput regression
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define model
model = KNeighborsRegressor()
# fit model
model.fit(X, y)
# make a prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = model.predict([row])
# summarize prediction
print(yhat[0])
```

运行该示例符合模型，然后对一个输入进行预测，确认模型预测了两个所需值。

```py
[-11.73511093  52.78406297]
```

### 多输出回归的决策树

以下示例在多输出回归数据集上拟合决策树模型，然后使用拟合模型进行单次预测。

```py
# decision tree for multioutput regression
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define model
model = DecisionTreeRegressor()
# fit model
model.fit(X, y)
# make a prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = model.predict([row])
# summarize prediction
print(yhat[0])
```

运行该示例符合模型，然后对一个输入进行预测，确认模型预测了两个所需值。

```py
[49.93137149 64.08484989]
```

### 使用交叉验证评估多输出回归

我们可能希望使用 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)来评估多输出回归。

这可以通过与评估任何其他机器学习模型相同的方式来实现。

我们将使用三次重复的 10 倍交叉验证来拟合和评估测试问题的*决策树回归器*模型。我们将使用平均绝对误差(MAE)表现指标作为分数。

下面列出了完整的示例。

```py
# evaluate multioutput regression model with k-fold cross-validation
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define model
model = DecisionTreeRegressor()
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例评估决策树模型在测试问题上的多输出回归的表现。报告了所有折叠和所有重复的平均和标准偏差。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

重要的是，误差是跨两个输出变量报告的，而不是每个输出变量单独的误差分数。

```py
MAE: 51.817 (2.863)
```

## 包装多输出回归算法

并非所有回归算法都支持多输出回归。

一个例子是[支持向量机](https://machinelearningmastery.com/support-vector-machines-for-machine-learning/)，虽然对于回归，它被称为支持向量回归，或[支持向量回归](https://Sklearn.org/stable/modules/generated/sklearn.svm.SVR.html)。

此算法不支持回归问题的多个输出，并且会产生错误。我们可以用下面列出的例子来证明这一点。

```py
# failure of support vector regression for multioutput regression (causes an error)
from sklearn.datasets import make_regression
from sklearn.svm import LinearSVR
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)
# define model
model = LinearSVR()
# fit model
# (THIS WILL CAUSE AN ERROR!)
model.fit(X, y)
```

运行该示例会报告一条错误消息，指示模型不支持多输出回归。

```py
ValueError: bad input shape (1000, 2)
```

使用为预测多输出回归的一个值而设计的回归模型的一种变通方法是将多输出回归问题分成多个子问题。

最明显的方法是将一个多输出回归问题分解成多个单输出回归问题。

例如，如果一个多输出回归问题需要预测三个值 *y1* 、 *y2* 和 *y3* ，给定一个输入 *X* ，那么这个问题可以分为三个单输出回归问题:

*   **问题 1** :给定 *X* ，预测 *y1* 。
*   **问题 2** :给定 *X* ，预测 *y2* 。
*   **问题 3** :给定 *X* ，预测 *y3* 。

实现这种技术有两种主要方法。

第一种方法是为每个要预测的产值开发一个单独的回归模型。我们可以认为这是一种直接的方法，因为每个目标值都是直接建模的。

第二种方法是第一种方法的扩展，只是模型被组织成一个链。来自第一个模型的预测被作为第二个模型的输入的一部分，并且输出到输入依赖的过程沿着模型链重复。

*   **直接多输出**:为每个要预测的数值开发一个独立的模型。
*   **链式多输出**:开发一系列相关模型，以匹配要预测的数值数量。

让我们依次仔细看看这些技术。

## 直接多输出回归

多输出回归的直接方法包括将回归问题分成一个单独的问题，用于预测每个目标变量。

这假设输出是相互独立的，这可能不是一个正确的假设。然而，这种方法可以对一系列问题提供令人惊讶的有效预测，并且可能值得尝试，至少作为表现基线。

例如，你的问题的输出实际上可能大部分是独立的，如果不是完全独立的话，这个策略可以帮助你找到答案。

这种方法得到了以回归模型为参数的[multipoutputruler](https://Sklearn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)类的支持。然后，它将为问题中的每个输出创建所提供模型的一个实例。

下面的例子演示了我们如何首先创建一个单输出回归模型，然后使用*multi outputerrors*类包装回归模型并添加对多输出回归的支持。

```py
...
# define base model
model = LinearSVR()
# define the direct multioutput wrapper model
wrapper = MultiOutputRegressor(model)
```

我们可以用一个关于我们的综合多输出回归问题的工作示例来演示这个策略。

下面的例子演示了使用[重复的 k 倍交叉验证](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/)和报告所有折叠和重复的平均绝对误差(MAE)用线性支持向量回归机评估*多输出处理器*类。

下面列出了完整的示例。

```py
# example of evaluating direct multioutput regression with an SVM model
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define base model
model = LinearSVR()
# define the direct multioutput wrapper model
wrapper = MultiOutputRegressor(model)
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(wrapper, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行这个例子报告了直接包装模型的平均值和标准偏差。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到由直接多输出回归策略包装的线性支持向量回归模型实现了大约 0.419 的 MAE。

```py
MAE: 0.419 (0.024)
```

我们还可以使用直接多输出回归包装器作为最终模型，并对新数据进行预测。

首先，模型适合所有可用数据，然后可以调用 *predict()* 函数对新数据进行预测。

下面的示例在我们的合成多输出回归数据集上演示了这一点。

```py
# example of making a prediction with the direct multioutput regression model
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define base model
model = LinearSVR()
# define the direct multioutput wrapper model
wrapper = MultiOutputRegressor(model)
# fit the model on the whole dataset
wrapper.fit(X, y)
# make a single prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = wrapper.predict([row])
# summarize the prediction
print('Predicted: %s' % yhat[0])
```

运行该示例使直接包装模型适合整个数据集，然后用于对新的数据行进行预测，就像我们在应用程序中使用该模型时可能做的那样。

```py
Predicted: [50.01932887 64.49432991]
```

现在我们已经熟悉了直接多输出回归包装器的使用，让我们看看链式方法。

## 链式多输出回归

将单输出回归模型用于多输出回归的另一种方法是创建模型的线性序列。

序列中的第一个模型使用输入并预测一个输出；第二模型使用来自第一模型的输入和输出来进行预测；第三个模型使用前两个模型的输入和输出进行预测，依此类推。

例如，如果一个多输出回归问题需要预测三个值 *y1* 、 *y2* 和 *y3* ，给定一个输入 *X* ，那么这可以被划分为如下三个相关的单输出回归问题:

*   **问题 1** :给定 *X* ，预测 *y1* 。
*   **问题 2** :给定 *X* 和 *yhat1* ，预测 *y2* 。
*   **问题 3** :给定 *X，yhat1，yhat2* ，预测 *y3* 。

这可以使用 Sklearn 库中的[returnorchain](https://Sklearn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html)类来实现。

模型的顺序可以基于数据集中输出的顺序(默认)或通过“*顺序*参数指定。例如，*阶=[0，1]* 将首先预测第 0 个输出，然后是第 1 个输出，而*阶=[1，0]* 将首先预测最后一个输出变量，然后是我们测试问题中的第一个输出变量。

下面的例子演示了我们如何首先创建单输出回归模型，然后使用*回归链*类包装回归模型并添加对多输出回归的支持。

```py
...
# define base model
model = LinearSVR()
# define the chained multioutput wrapper model
wrapper = RegressorChain(model, order=[0,1])
```

我们可以用一个关于我们的综合多输出回归问题的工作示例来演示这个策略。

下面的例子演示了使用[重复的 k 倍交叉验证](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/)和报告所有折叠和重复的平均绝对误差(MAE)用线性支持向量回归机评估*回归链*类。

下面列出了完整的示例。

```py
# example of evaluating chained multioutput regression with an SVM model
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define base model
model = LinearSVR()
# define the chained multioutput wrapper model
wrapper = RegressorChain(model)
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(wrapper, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行这个例子报告了链式包装模型的平均值和标准偏差。
请注意，在运行示例时，您可能会看到一个*收敛警告*，可以安全地忽略。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到由链式多输出回归策略包装的线性支持向量回归模型实现了大约 0.643 的 MAE。

```py
MAE: 0.643 (0.313)
```

我们还可以使用链式多输出回归包装器作为最终模型，并对新数据进行预测。

首先，模型适合所有可用数据，然后可以调用 *predict()* 函数对新数据进行预测。

下面的示例在我们的合成多输出回归数据集上演示了这一点。

```py
# example of making a prediction with the chained multioutput regression model
from sklearn.datasets import make_regression
from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# define base model
model = LinearSVR()
# define the chained multioutput wrapper model
wrapper = RegressorChain(model)
# fit the model on the whole dataset
wrapper.fit(X, y)
# make a single prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = wrapper.predict([row])
# summarize the prediction
print('Predicted: %s' % yhat[0])
```

运行该示例使链式包装模型适合整个数据集，然后用于对新的数据行进行预测，就像我们在应用程序中使用该模型时可能做的那样。

```py
Predicted: [50.03206    64.73673318]
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 蜜蜂

*   [多类和多标签算法，API](https://Sklearn.org/stable/modules/multiclass.html) 。
*   [sklearn . datasets . make _ revolution API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)。
*   [硬化。多倍体。多倍体回归 API](https://Sklearn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html) 。
*   [硬化。多倍体。回归 API](https://Sklearn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html) 。

## 摘要

在本教程中，您发现了如何为多输出回归开发机器学习模型。

具体来说，您了解到:

*   机器学习中的多输出回归问题。
*   如何开发内在支持多输出回归的机器学习模型？
*   如何开发包装器模型，允许不支持多输出的算法用于多输出回归。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。