# Python 中的多元自适应回归样条（MARS）

> 原文：<https://machinelearningmastery.com/multivariate-adaptive-regression-splines-mars-in-python/>

最后更新于 2021 年 4 月 27 日

**多元自适应回归样条**，或 **MARS** ，是一种用于复杂非线性回归问题的算法。

该算法包括找到一组简单的线性函数，这些函数合计起来产生最佳的预测表现。通过这种方式，MARS 是一种简单线性函数的集成，并且可以在具有许多输入变量和复杂非线性关系的挑战性回归问题上获得良好的表现。

在本教程中，您将发现如何在 Python 中开发多元自适应回归样条模型。

完成本教程后，您将知道:

*   多元非线性回归预测建模问题的 MARS 算法。
*   如何使用 py-earth API 开发与 Sklearn 兼容的 MARS 模型？
*   如何用 MARS 模型对回归预测建模问题进行评估和预测。

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Multivariate Adaptive Regression Splines (MARS) in Python](img/6f193d5bcf346492248fbd6bd65544fb.png)

Python 中的多元自适应回归样条(MARS)
图片由 [Sei F](https://www.flickr.com/photos/125983633@N03/40163338071/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  多元自适应回归样条
2.  火星 Python API
3.  回归的火星工作实例

## 多元自适应回归样条

多元自适应回归样条，简称 MARS，是一种为多元非线性回归问题设计的算法。

回归问题是模型必须预测数值的问题。多变量意味着输入变量不止一个(通常是几十个)，非线性意味着输入变量和目标变量之间的关系不是线性的，意味着不能用直线来描述(例如，它是弯曲的或弯曲的)。

> MARS 是回归的自适应过程，非常适合高维问题(即大量输入)。它可以看作是逐步线性回归的推广

—第 321 页，[统计学习的要素](https://amzn.to/31SA3bt)，2016。

MARS 算法包括发现一组简单的分段线性函数来表征数据，并将其集合起来进行预测。在某种意义上，模型是线性函数的集合。

A [分段线性函数](https://en.wikipedia.org/wiki/Piecewise)是由较小的函数组成的函数。在这种情况下，它是直接输出 0 或输入值的函数。

一个输入变量的“*右功能*”包括为该变量选择一个特定值，并为低于该值的所有值输出一个 0，为高于所选值的所有值输出该值。

*   f(x) = x 如果 x >值，则为 0

或者相反，可以使用“*左功能*”，其中小于所选值的值直接输出，大于所选值的值输出零。

*   f(x) = x 如果 x

这被称为**铰链函数**，其中选择的值或分割点是函数的“*结*”。它也被称为神经网络中的[校正线性函数](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)。

这些函数也被称为“*样条线*”，因此得名该算法。

> 每个函数都是分段线性的，在值 t 处有一个结。在……的术语中，这些是线性样条。

—第 322 页，[统计学习的要素](https://amzn.to/31SA3bt)，2016。

MARS 算法生成许多这样的函数，称为一个或多个输入变量的基函数。

然后，从这些具有目标变量的基函数的输出中学习线性回归模型。这意味着每个基函数的输出由一个系数加权。通过对模型中所有基函数的加权输出求和来进行预测。

MARS 算法的关键是如何选择基函数。这包括两个步骤:生长或生成阶段称为前向阶段，修剪或精炼阶段称为后向阶段。

*   **前向阶段**:生成模型的候选基函数。
*   **后向阶段**:从模型中删除基础函数。

前一阶段包括生成基函数并添加到模型中。像决策树一样，训练数据集中每个输入变量的每个值都被视为基函数的候选值。

> 切割点是如何确定的？通过创建具有候选特征的线性回归模型，将每个预测器的每个数据点评估为候选切割点，并计算相应的模型误差。

—第 146 页，[应用预测建模](https://amzn.to/3iFPHhq)，2013 年。

对于相同分割点的分段线性函数的左右版本，函数总是成对添加。只有当生成的一对函数减少了整个模型产生的误差时，才会将其添加到模型中。

向后阶段包括选择要从模型中删除的函数，一次一个。只有当某项功能不会对表现产生影响(中性)或提升预测表现时，才会从模型中删除该功能。

> 一旦创建了完整的特征集，该算法将依次移除对模型方程没有显著贡献的单个特征。这个“修剪”过程评估每个预测变量，并通过将其包含在模型中来估计误差率降低了多少。

—第 148 页，[应用预测建模](https://amzn.to/3iFPHhq)，2013 年。

使用训练数据集的[交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)来评估后向阶段中模型表现的变化，简称为广义交叉验证或 GCV。因此，可以估计每个分段线性模型对模型表现的影响。

模型使用的函数数量是自动确定的，因为当没有进一步的改进时，修剪过程将停止。

唯一需要考虑的两个关键超参数是要生成的候选函数的总数(通常设置为非常大的数字)和要生成的函数的程度。

> ……有两个与 MARS 模型相关的调优参数:添加到模型中的特性的程度和保留的术语数量。后一个参数可以使用默认修剪过程(使用 GCV)自动确定，由用户设置或使用外部重采样技术确定。

—第 149 页，[应用预测建模](https://amzn.to/3iFPHhq)，2013 年。

度是每个分段线性函数考虑的输入变量的数量。默认情况下，该值设置为 1，但可以设置为更大的值，以允许模型捕获输入变量之间的复杂交互。程度通常保持较小，以限制模型的计算复杂性(内存和执行时间)。

MARS 算法的一个好处是，它只使用提升模型表现的输入变量。与装袋和随机森林集成算法非常相似，MARS 实现了自动类型的特征选择。

> ……模型自动进行特征选择；模型方程独立于不涉及任何最终模型特征的预测变量。这一点不能低估。

—第 149 页，[应用预测建模](https://amzn.to/3iFPHhq)，2013 年。

现在我们已经熟悉了 MARS 算法，让我们看看如何用 Python 开发 MARS 模型。

## 火星 Python API

Sklearn 库中没有提供 MARS 算法；相反，必须使用第三方库。

MARS 由 [py-earth Python 库](https://github.com/Sklearn-contrib/py-earth)提供。

“*地球*”是“*火星*”(行星)上的一个玩法，也是 R 中提供 Mars 算法的[包的名字。](https://cran.r-project.org/web/packages/earth/index.html)

py-earth Python 包是以 R 版本命名的 MARS 的 Python 实现，提供了与 Sklearn 机器学习库的完全可比性。

第一步是安装 py-earth 库。我建议使用 pip 包管理器，从命令行使用以下命令:

```py
sudo pip install sklearn-contrib-py-earth
```

安装后，我们可以加载库，并在 Python 脚本中打印版本，以确认它安装正确。

```py
# check pyearth version
import pyearth
# display version
print(pyearth.__version__)
```

运行脚本将加载 py-earth 库并打印库版本号。

您的版本号应该相同或更高。

```py
0.1.0
```

通过创建[地球类](https://contrib.Sklearn.org/py-earth/content.html#pyearth.Earth)的实例，可以使用默认模型超参数创建火星模型。

```py
...
# define the model
model = Earth()
```

一旦创建，模型就可以直接拟合训练数据。

```py
...
# fit the model on training dataset
model.fit(X, y)
```

默认情况下，您可能不需要设置任何算法超参数。

该算法自动发现要使用的基函数的数量和类型。

基函数的最大数量由“ *max_terms* ”参数配置，并设置为与输入变量数量成比例的大数，上限为 400。

分段线性函数的度，即每个基函数中考虑的输入变量的数量，由“ *max_degree* 参数控制，默认为 1。

一旦拟合，该模型可用于对新数据进行预测。

```py
...
Xnew = ...
# make a prediction
yhat = model.predict(Xnew)
```

通过调用 *summary()* 函数，可以创建拟合模型的概要。

```py
...
# print a summary of the fit model
print(model.summary())
```

摘要返回模型中使用的基函数列表，以及在训练数据集上通过广义交叉验证(GCV)估计的模型的估计表现。

下面提供了一个总结输出的例子，我们可以看到该模型有 19 个基函数，估计的均方误差约为 25。

```py
Earth Model
--------------------------------------
Basis Function   Pruned  Coefficient
--------------------------------------
(Intercept)      No      313.89
h(x4-1.88408)    No      98.0124
h(1.88408-x4)    No      -99.2544
h(x17-1.82851)   No      99.7349
h(1.82851-x17)   No      -99.9265
x14              No      96.7872
x15              No      85.4874
h(x6-1.10441)    No      76.4345
h(1.10441-x6)    No      -76.5954
x9               No      76.5097
h(x3+2.41424)    No      73.9003
h(-2.41424-x3)   No      -73.2001
x0               No      71.7429
x2               No      71.297
x19              No      67.6034
h(x11-0.575217)  No      66.0381
h(0.575217-x11)  No      -65.9314
x18              No      62.1124
x12              No      38.8801
--------------------------------------
MSE: 25.5896, GCV: 25.8266, RSQ: 0.9997, GRSQ: 0.9997
```

现在我们已经熟悉了使用 py-earth API 开发 MARS 模型，让我们来看一个成功的例子。

## 回归的火星工作实例

在本节中，我们将看一个为回归预测建模问题评估和使用 MARS 模型的工作示例。

首先，我们必须定义一个回归数据集。

我们将使用[make _ revolution()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)创建一个包含 20 个特征(列)和 10，000 个示例(行)的合成回归问题。下面的示例创建并总结了合成数据集的形状。

```py
# define a synthetic regression dataset
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=10000, n_features=20, n_informative=15, noise=0.5, random_state=7)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例会创建数据集并汇总行数和列数，与我们的预期相符。

```py
(10000, 20) (10000,)
```

接下来，我们可以在数据集上评估一个 MARS 模型。

我们将使用默认的超参数来定义模型。

```py
...
# define the model
model = Earth()
```

我们将使用[重复的 k 倍交叉验证](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/)来评估模型，这在评估回归模型时是一个很好的实践。

在这种情况下，我们将使用三次重复和 10 次折叠。

```py
...
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
```

我们将使用平均绝对误差(简称 MAE)来评估模型表现。

Sklearn API 将使 MAE 分数为负，以便将其最大化，这意味着分数范围将从负无穷大(最差)到 0(最佳)。

```py
...
# evaluate the model and collect results
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
```

最后，我们将报告模型的表现，作为所有重复和交叉验证折叠的平均 MAE 分数。

```py
...
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

将这些联系在一起，下面列出了在回归数据集上评估 MARS 模型的完整示例。

```py
# evaluate multivariate adaptive regression splines for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from pyearth import Earth
# define dataset
X, y = make_regression(n_samples=10000, n_features=20, n_informative=15, noise=0.5, random_state=7)
# define the model
model = Earth()
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect results
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
```

运行该示例评估 MARS 模型的表现，并报告 MAE 分数的平均值和标准偏差。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到 MARS 算法在合成回归数据集上实现了大约 4.0(忽略符号)的平均 MAE。

```py
MAE: -4.041 (0.085)
```

我们可能希望使用 MARS 作为我们的最终模型，并使用它对新数据进行预测。

这需要首先在所有可用数据上定义和拟合模型。

```py
...
# define the model
model = Earth()
# fit the model on the whole dataset
model.fit(X, y)
```

然后我们可以调用 *predict()* 函数，传入新的输入数据，以便进行预测。

```py
...
# make a prediction for a single row of data
yhat = model.predict([row])
```

下面列出了拟合 MARS 最终模型并对单行新数据进行预测的完整示例。

```py
# make a prediction with multivariate adaptive regression splines for regression
from sklearn.datasets import make_regression
from pyearth import Earth
# define dataset
X, y = make_regression(n_samples=10000, n_features=20, n_informative=15, noise=0.5, random_state=7)
# define the model
model = Earth()
# fit the model on the whole dataset
model.fit(X, y)
# define a single row of data
row = [-0.6305395, -0.1381388, -1.23954844, 0.32992515, -0.36612979, 0.74962718, 0.21532504, 0.90983424, -0.60309177, -1.46455027, -0.06788126, -0.30329357, -0.60350541, 0.7369983, 0.21774321, -1.2365456, 0.69159078, -0.16074843, -1.39313206, 1.16044301]
# make a prediction for a single row of data
yhat = model.predict([row])
# summarize the prediction
print('Prediction: %d' % yhat[0])
```

运行该示例使 MARS 模型适用于所有可用数据，然后进行单一回归预测。

```py
Prediction: -393
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [多元自适应回归样条](https://www.jstor.org/stable/2241837?seq=1)，1991。
*   [多元自适应回归样条](https://journals.sagepub.com/doi/abs/10.1177/096228029500400303)导论，1995。

### 书

*   第 9.4 节 MARS:多元自适应回归样条，[统计学习的要素](https://amzn.to/31SA3bt)，2016。
*   第 7.2 节多元自适应回归样条，[应用预测建模](https://amzn.to/3iFPHhq)，2013。

### 蜜蜂

*   [py-earth 项目，GitHub](https://github.com/Sklearn-contrib/py-earth) 。
*   [Py-地球文件](https://contrib.Sklearn.org/py-earth/)。
*   [sklearn . datasets . make _ revolution API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)。

### 文章

*   [多元自适应回归样条，维基百科](https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_spline)。
*   [分段，维基百科](https://en.wikipedia.org/wiki/Piecewise)。

## 摘要

在本教程中，您发现了如何在 Python 中开发多元自适应回归样条模型。

具体来说，您了解到:

*   多元非线性回归预测建模问题的 MARS 算法。
*   如何使用 py-earth API 开发与 Sklearn 兼容的 MARS 模型？
*   如何用 MARS 模型对回归预测建模问题进行评估和预测。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。