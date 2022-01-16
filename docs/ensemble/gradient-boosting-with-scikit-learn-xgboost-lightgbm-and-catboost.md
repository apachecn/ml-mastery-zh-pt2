# 将 Sklearn、XGBoost、LightGBM 和 CatBoost 用于梯度提升

> 原文：<https://machinelearningmastery.com/gradient-boosting-with-Sklearn-xgboost-lightgbm-and-catboost/>

最后更新于 2021 年 4 月 27 日

梯度提升是一种强大的集成机器学习算法。

它适用于结构化预测建模问题，如表格数据的分类和回归，通常是机器学习竞赛中获胜解决方案的主要算法或主要算法之一，就像 Kaggle 上的算法一样。

有许多可用的梯度提升实现，包括 SciPy 中的标准实现和高效的第三方库。每一个都使用不同的接口，甚至不同的算法名称。

在本教程中，您将发现如何在 Python 中使用梯度提升模型进行分类和回归。

为 Python 中梯度提升的四种主要实现提供了标准化的代码示例，供您复制粘贴并在自己的预测建模项目中使用。

完成本教程后，您将知道:

*   梯度提升是一种集成算法，通过最小化误差梯度来适应增强的决策树。
*   如何使用 Sklearn 评估和使用梯度提升，包括梯度提升机和基于直方图的算法。
*   如何评估和使用第三方梯度提升算法，包括 XGBoost、LightGBM 和 CatBoost。

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Gradient Boosting with Sklearn, XGBoost, LightGBM, and CatBoost](img/b874fe3e70b3ccb9ab98de9f992c11c1.png)

使用 Sklearn、XGBoost、LightGBM 和 CatBoost 进行梯度提升
图片由[约翰](https://flickr.com/photos/shebalso/441861081/)提供，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  梯度提升概述
2.  使用 Scikit 进行梯度提升-学习
    1.  库安装
    2.  测试问题
    3.  梯度升压
    4.  基于直方图的梯度提升
3.  带 XGBoost 的梯度升压
    1.  库安装
    2.  用于分类的扩展包
    3.  用于回归的 XGBoost
4.  用 LightGBM 进行梯度升压
    1.  库安装
    2.  用于分类的 LightGBM
    3.  回归之光
5.  利用 CatBoost 实现梯度升压
    1.  库安装
    2.  分类的分类增强
    3.  回归的助力

## 梯度提升概述

梯度提升是指一类集成机器学习算法，可用于分类或回归预测建模问题。

梯度提升也称为梯度树增强、随机梯度提升(扩展)和梯度提升机，简称 GBM。

集成是由决策树模型构建的。树被一次一个地添加到集合中，并且适合于校正由先前模型产生的预测误差。这是一种称为 boosting 的集成机器学习模型。

使用任意可微损失函数和梯度下降优化算法拟合模型。这给这项技术起了一个名字，“梯度提升”，因为随着模型的拟合，损失梯度被最小化，很像一个神经网络。

梯度提升是一种有效的机器学习算法，并且通常是用于在表格和类似结构化数据集上赢得机器学习竞赛(如 Kaggle)的主要算法或主要算法之一。

**注**:在本教程中，我们将不讨论梯度提升算法背后的理论。

有关梯度提升算法的更多信息，请参见教程:

*   [机器学习梯度提升算法的简单介绍](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)

该算法提供了超参数，这些超参数应该而且可能必须针对特定数据集进行调整。虽然有许多超参数需要调整，但最重要的可能如下:

*   模型中树或估计量的数量。
*   模型的学习率。
*   随机模型的行列采样率。
*   最大树深。
*   最小树重。
*   正则项α和λ。

**注**:在本教程中，我们将不探讨如何配置或调整梯度提升算法的配置。

有关调整梯度提升算法的超参数的更多信息，请参见教程:

*   [如何配置梯度提升算法](https://machinelearningmastery.com/configure-gradient-boosting-algorithm/)

Python 中有许多梯度提升算法的实现。也许最常用的实现是 Sklearn 库提供的版本。

可以使用额外的第三方库来提供算法的计算高效的替代实现，这些实现通常在实践中获得更好的结果。例子包括 XGBoost 库、LightGBM 库和 CatBoost 库。

**你有不同喜欢的梯度提升实现吗？**
在下面的评论里告诉我。

在预测建模项目中使用梯度提升时，您可能希望测试算法的每个实现。

本教程提供了梯度提升算法在分类和回归预测建模问题上的每个实现的示例，您可以将其复制粘贴到项目中。

让我们依次看一下每一个。

**注**:本教程中我们不是比较算法的表现。相反，我们提供代码示例来演示如何使用每个不同的实现。因此，我们使用合成测试数据集来演示评估和预测每个实现。

本教程假设您已经安装了 Python 和 SciPy。如果您需要帮助，请参阅教程:

*   [如何用 Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 设置机器学习的 Python 环境

## 使用 Sklearn 进行梯度提升

在本节中，我们将回顾如何在 [Sklearn 库](https://Sklearn.org/)中使用梯度提升算法实现。

### 库安装

首先，让我们安装库。

不要跳过这一步，因为您需要确保安装了最新版本。

您可以使用 pip Python 安装程序安装 Sklearn 库，如下所示:

```py
sudo pip install Sklearn
```

有关特定于您的平台的其他安装说明，请参见:

*   [安装 Sklearn](https://Sklearn.org/stable/install.html)

接下来，让我们确认库已安装，并且您使用的是现代版本。

运行以下脚本打印库版本号。

```py
# check Sklearn version
import sklearn
print(sklearn.__version__)
```

运行该示例时，您应该会看到以下版本号或更高版本号。

```py
0.22.1
```

### 测试问题

我们将演示用于分类和回归的梯度提升算法。

因此，我们将使用 Sklearn 库中的合成测试问题。

#### 类别数据集

我们将使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建一个测试二进制类别数据集。

数据集将有 1，000 个示例，有 10 个输入要素，其中 5 个是信息性的，其余 5 个是冗余的。我们将修复随机数种子，以确保每次运行代码时得到相同的例子。

下面列出了创建和汇总数据集的示例。

```py
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集，并确认样本和要素的预期数量。

```py
(1000, 10) (1000,)
```

#### 回归数据集

我们将使用[make _ revolution()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_regression.html)创建一个测试回归数据集。

与类别数据集一样，回归数据集将有 1，000 个示例，有 10 个输入要素，其中 5 个是信息性的，其余 5 个是冗余的。

```py
# test regression dataset
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

运行该示例将创建数据集，并确认样本和要素的预期数量。

```py
(1000, 10) (1000,)
```

接下来，让我们看看如何在 Sklearn 中开发梯度提升模型。

### 梯度升压

Sklearn 库通过*GradientBoostingClassifier*和*gradientboostingretriever*类提供回归和分类的 GBM 算法。

让我们依次仔细看看每一个。

#### 分级梯度推进机

下面的示例首先使用重复的 k 倍交叉验证在测试问题上评估一个[gradientboosting 分类器](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)，并报告平均准确度。然后在所有可用数据上拟合单一模型，并进行单一预测。

下面列出了完整的示例。

```py
# gradient boosting for classification in Sklearn
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# evaluate the model
model = GradientBoostingClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = GradientBoostingClassifier()
model.fit(X, y)
# make a single prediction
row = [[2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]]
yhat = model.predict(row)
print('Prediction: %d' % yhat[0])
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先使用重复的 k 倍交叉验证报告模型的评估，然后使用模型对整个数据集进行单次预测的结果。

```py
Accuracy: 0.915 (0.025)
Prediction: 1
```

#### 回归梯度提升机

下面的示例首先使用重复的 K 折交叉验证对测试问题评估一个[gradientboostingrevoller](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)，并报告平均绝对误差。然后在所有可用数据上拟合单一模型，并进行单一预测。

下面列出了完整的示例。

```py
# gradient boosting for regression in Sklearn
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# evaluate the model
model = GradientBoostingRegressor()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = GradientBoostingRegressor()
model.fit(X, y)
# make a single prediction
row = [[2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
yhat = model.predict(row)
print('Prediction: %.3f' % yhat[0])
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先使用重复的 k 倍交叉验证报告模型的评估，然后使用模型对整个数据集进行单次预测的结果。

```py
MAE: -11.854 (1.121)
Prediction: -80.661
```

### 基于直方图的梯度提升

Sklearn 库提供了梯度提升算法的替代实现，称为基于直方图的梯度提升。

这是一种实现梯度树增强的替代方法，灵感来自 LightGBM 库(稍后将详细描述)。这个实现是通过*历史梯度提升分类器*和*历史梯度提升回归器*类提供的。

基于直方图的梯度提升方法的主要优势是速度。这些实现旨在更快地适应训练数据。

在编写本文时，这是一个实验性的实现，需要在代码中添加以下代码行，以便能够访问这些类。

```py
from sklearn.experimental import enable_hist_gradient_boosting
```

如果没有这一行，您将看到如下错误:

```py
ImportError: cannot import name 'HistGradientBoostingClassifier'
```

或者

```py
ImportError: cannot import name 'HistGradientBoostingRegressor'
```

让我们仔细看看如何使用这个实现。

#### 基于直方图的梯度提升分类机

下面的示例首先使用重复的 k 倍交叉验证对测试问题评估一个[HistGradientBoostingCollector](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)，并报告平均准确度。然后在所有可用数据上拟合单一模型，并进行单一预测。

下面列出了完整的示例。

```py
# histogram-based gradient boosting for classification in Sklearn
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# evaluate the model
model = HistGradientBoostingClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = HistGradientBoostingClassifier()
model.fit(X, y)
# make a single prediction
row = [[2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]]
yhat = model.predict(row)
print('Prediction: %d' % yhat[0])
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先使用重复的 k 倍交叉验证报告模型的评估，然后使用模型对整个数据集进行单次预测的结果。

```py
Accuracy: 0.935 (0.024)
Prediction: 1
```

#### 基于直方图的梯度提升回归机

下面的示例首先使用重复的 K 折交叉验证对测试问题评估一个[HistGradientBoostingResolver](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)，并报告平均绝对误差。然后在所有可用数据上拟合单一模型，并进行单一预测。

下面列出了完整的示例。

```py
# histogram-based gradient boosting for regression in Sklearn
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# evaluate the model
model = HistGradientBoostingRegressor()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = HistGradientBoostingRegressor()
model.fit(X, y)
# make a single prediction
row = [[2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
yhat = model.predict(row)
print('Prediction: %.3f' % yhat[0])
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先使用重复的 k 倍交叉验证报告模型的评估，然后使用模型对整个数据集进行单次预测的结果。

```py
MAE: -12.723 (1.540)
Prediction: -77.837
```

## 带 XGBoost 的梯度升压

[XGBoost](https://xgboost.ai/) ，是“*极限梯度提升*的缩写，是一个提供梯度提升算法高效实现的库。

XGBoost 实现的主要好处是计算效率和更好的模型表现。

有关 XGBoost 的优势和功能的更多信息，请参见教程:

*   [应用机器学习 XGBoost 的温和介绍](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

### 库安装

您可以使用 pip Python 安装程序安装 XGBoost 库，如下所示:

```py
sudo pip install xgboost
```

有关特定于您的平台的其他安装说明，请参见:

*   [XGBoost 安装指南](https://xgboost.readthedocs.io/en/latest/build.html)

接下来，让我们确认库已安装，并且您使用的是现代版本。

运行以下脚本打印库版本号。

```py
# check xgboost version
import xgboost
print(xgboost.__version__)
```

运行该示例时，您应该会看到以下版本号或更高版本号。

```py
1.0.1
```

XGBoost 库提供了包装器类，因此高效的算法实现可以与 Sklearn 库一起使用，特别是通过 *XGBClassifier* 和*xgbreversor*类。

让我们依次仔细看看每一个。

### 用于分类的扩展包

下面的示例首先使用重复的 k 倍交叉验证评估测试问题上的[xbindicator](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)，并报告平均准确度。然后对所有可用数据拟合单一模型，并进行单一预测。

下面列出了完整的示例。

```py
# xgboost for classification
from numpy import asarray
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# evaluate the model
model = XGBClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = XGBClassifier()
model.fit(X, y)
# make a single prediction
row = [2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]
row = asarray(row).reshape((1, len(row)))
yhat = model.predict(row)
print('Prediction: %d' % yhat[0])
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先使用重复的 k 倍交叉验证报告模型的评估，然后使用模型对整个数据集进行单次预测的结果。

```py
Accuracy: 0.936 (0.019)
Prediction: 1
```

### 用于回归的 XGBoost

下面的示例首先使用重复的 k 倍交叉验证对测试问题评估一个[xgbreversor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)，并报告平均绝对误差。然后在所有可用数据上拟合单一模型，并进行单一预测。

下面列出了完整的示例。

```py
# xgboost for regression
from numpy import asarray
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# evaluate the model
model = XGBRegressor(objective='reg:squarederror')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = XGBRegressor(objective='reg:squarederror')
model.fit(X, y)
# make a single prediction
row = [2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]
row = asarray(row).reshape((1, len(row)))
yhat = model.predict(row)
print('Prediction: %.3f' % yhat[0])
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先使用重复的 k 倍交叉验证报告模型的评估，然后使用模型对整个数据集进行单次预测的结果。

```py
MAE: -15.048 (1.316)
Prediction: -93.434
```

## 用 LightGBM 进行梯度升压

[LightGBM](https://github.com/microsoft/LightGBM) 是 Light Gradient Boosted Machine 的缩写，是微软开发的一个库，提供了一个高效的梯度提升算法的实现。

LightGBM 的主要好处是对训练算法的改变，这使得过程大大加快，并且在许多情况下，产生了更有效的模型。

有关 LightGBM 算法的更多技术细节，请参见论文:

*   [LightGBM:一种高效的梯度提升决策树](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)，2017。

### 库安装

您可以使用 pip Python 安装程序安装 LightGBM 库，如下所示:

```py
sudo pip install lightgbm
```

有关特定于您的平台的其他安装说明，请参见:

*   [LightGBM 安装指南](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

接下来，让我们确认库已安装，并且您使用的是现代版本。

运行以下脚本打印库版本号。

```py
# check lightgbm version
import lightgbm
print(lightgbm.__version__)
```

运行该示例时，您应该会看到以下版本号或更高版本号。

```py
2.3.1
```

LightGBM 库提供了包装类，因此高效的算法实现可以与 Sklearn 库一起使用，特别是通过 *LGBMClassifier* 和*lgbmreversor*类。

让我们依次仔细看看每一个。

### 用于分类的 LightGBM

下面的示例首先使用重复的 k 倍交叉验证对测试问题评估一个 [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) ，并报告平均准确度。然后在所有可用数据上拟合单一模型，并进行单一预测。

下面列出了完整的示例。

```py
# lightgbm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# evaluate the model
model = LGBMClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = LGBMClassifier()
model.fit(X, y)
# make a single prediction
row = [[2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]]
yhat = model.predict(row)
print('Prediction: %d' % yhat[0])
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先使用重复的 k 倍交叉验证报告模型的评估，然后使用模型对整个数据集进行单次预测的结果。

```py
Accuracy: 0.934 (0.021)
Prediction: 1
```

### 回归之光

下面的示例首先使用重复的 k 倍交叉验证对测试问题评估一个[lgbmrejector](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)，并报告平均绝对误差。然后在所有可用数据上拟合单一模型，并进行单一预测。

下面列出了完整的示例。

```py
# lightgbm for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# evaluate the model
model = LGBMRegressor()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = LGBMRegressor()
model.fit(X, y)
# make a single prediction
row = [[2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
yhat = model.predict(row)
print('Prediction: %.3f' % yhat[0])
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先使用重复的 k 倍交叉验证报告模型的评估，然后使用模型对整个数据集进行单次预测的结果。

```py
MAE: -12.739 (1.408)
Prediction: -82.040
```

### 利用 CatBoost 实现梯度升压

[CatBoost](https://catboost.ai/) 是在 [Yandex](https://en.wikipedia.org/wiki/Yandex) 开发的第三方库，提供了梯度 boosting 算法的高效实现。

CatBoost 的主要好处(除了计算速度的提高)是支持分类输入变量。这使库的名称为“类别梯度提升”

有关 CatBoost 算法的更多技术细节，请参见论文:

*   [CatBoost:分类特征支持的梯度提升](https://arxiv.org/abs/1810.11363)，2017。

### 库安装

您可以使用 pip Python 安装程序安装 CatBoost 库，如下所示:

```py
sudo pip install catboost
```

有关特定于您的平台的其他安装说明，请参见:

*   cat boost 安装指南

接下来，让我们确认库已安装，并且您使用的是现代版本。

运行以下脚本打印库版本号。

```py
# check catboost version
import catboost
print(catboost.__version__)
```

运行该示例时，您应该会看到以下版本号或更高版本号。

```py
0.21
```

CatBoost 库提供了包装器类，因此高效的算法实现可以与 Sklearn 库一起使用，特别是通过*CatBoost 分类器*和*CatBoost 渐行渐远器*类。

让我们依次仔细看看每一个。

### 分类的分类增强

下面的示例首先使用重复的 k 倍交叉验证在测试问题上评估一个 [CatBoostClassifier](https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html) ，并报告平均准确度。然后在所有可用数据上拟合单一模型，并进行单一预测。

下面列出了完整的示例。

```py
# catboost for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# evaluate the model
model = CatBoostClassifier(verbose=0, n_estimators=100)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = CatBoostClassifier(verbose=0, n_estimators=100)
model.fit(X, y)
# make a single prediction
row = [[2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]]
yhat = model.predict(row)
print('Prediction: %d' % yhat[0])
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先使用重复的 k 倍交叉验证报告模型的评估，然后使用模型对整个数据集进行单次预测的结果。

```py
Accuracy: 0.931 (0.026)
Prediction: 1
```

### 回归的助力

下面的示例首先使用重复的 k 倍交叉验证对测试问题评估一个[catbootstregressor](https://catboost.ai/docs/concepts/python-reference_catboostregressor.html)，并报告平均绝对误差。然后在所有可用数据上拟合单一模型，并进行单一预测。

下面列出了完整的示例。

```py
# catboost for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot
# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# evaluate the model
model = CatBoostRegressor(verbose=0, n_estimators=100)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = CatBoostRegressor(verbose=0, n_estimators=100)
model.fit(X, y)
# make a single prediction
row = [[2.02220122, 0.31563495, 0.82797464, -0.30620401, 0.16003707, -1.44411381, 0.87616892, -0.50446586, 0.23009474, 0.76201118]]
yhat = model.predict(row)
print('Prediction: %.3f' % yhat[0])
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例首先使用重复的 k 倍交叉验证报告模型的评估，然后使用模型对整个数据集进行单次预测的结果。

```py
MAE: -9.281 (0.951)
Prediction: -74.212
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何用 Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 设置机器学习的 Python 环境
*   [机器学习梯度提升算法的简单介绍](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
*   [如何配置梯度提升算法](https://machinelearningmastery.com/configure-gradient-boosting-algorithm/)
*   [应用机器学习 XGBoost 的温和介绍](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

### 报纸

*   [随机梯度提升](https://www.sciencedirect.com/science/article/pii/S0167947301000652)，2002。
*   [xboost:一种可扩展的树推进系统](https://arxiv.org/abs/1603.02754)，2016。
*   [LightGBM:一种高效的梯度提升决策树](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)，2017。
*   [CatBoost:分类特征支持的梯度提升](https://arxiv.org/abs/1810.11363)，2017。

### 蜜蜂

*   [Scikit-学习主页](https://Sklearn.org/)。
*   硬化。API 集。
*   [XGBoost 主页](https://xgboost.ai/)。
*   [XGBoost Python API](https://xgboost.readthedocs.io/en/latest/python/python_api.html) 。
*   [LightGBM 项目](https://github.com/microsoft/LightGBM)。
*   [【light GBM python API】](https://lightgbm.readthedocs.io/en/latest/Python-API.html)。
*   cat boost 主页。
*   cat boost API。

### 文章

*   [梯度提升，维基百科](https://en.wikipedia.org/wiki/Gradient_boosting)。
*   [XGBoost，维基百科](https://en.wikipedia.org/wiki/XGBoost)。

## 摘要

在本教程中，您发现了如何在 Python 中使用梯度提升模型进行分类和回归。

具体来说，您了解到:

*   梯度提升是一种集成算法，通过最小化误差梯度来适应增强的决策树。
*   如何使用 Sklearn 评估和使用梯度提升，包括梯度提升机和基于直方图的算法。
*   如何评估和使用包括 XGBoost、LightGBM 和 CatBoost 在内的第三方梯度提升算法。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。