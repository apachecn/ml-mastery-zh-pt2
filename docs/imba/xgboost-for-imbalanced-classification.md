# 如何为不平衡分类配置 XGBoost

> 原文：<https://machinelearningmastery.com/xgboost-for-imbalanced-classification/>

最后更新于 2020 年 8 月 21 日

XGBoost 算法对于广泛的回归和分类预测建模问题是有效的。

它是随机梯度提升算法的有效实现，并提供了一系列超参数，可对模型训练过程进行精细控制。尽管该算法总体上表现良好，即使在不平衡类别数据集上也是如此，但它提供了一种调整训练算法的方法，以更加关注具有倾斜类分布的数据集的少数类的误分类。

XGBoost 的这个修改版本被称为类加权 XGBoost 或成本敏感 XGBoost，并且可以在具有严重类不平衡的二进制分类问题上提供更好的表现。

在本教程中，您将发现用于不平衡分类的加权 XGBoost。

完成本教程后，您将知道:

*   梯度提升如何从高层次工作，以及如何开发一个用于分类的 XGBoost 模型。
*   如何修改 XGBoost 训练算法，以在训练过程中加权与正类重要性成比例的误差梯度。
*   如何为 XGBoost 训练算法配置正类权重，如何网格搜索不同的配置。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Configure XGBoost for Imbalanced Classification](img/4a70d24c586606b4ada55eb9d9e14644.png)

如何配置不平衡分类的 xboost
图片由 [flowcomm](https://flickr.com/photos/flowcomm/46907520995/) 提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  不平衡类别数据集
2.  用于分类的 XGBoost 模型
3.  类别不平衡的加权扩展
4.  调整类别加权超参数

## 不平衡类别数据集

在我们深入到不平衡分类的 XGBoost 之前，让我们首先定义一个不平衡类别数据集。

我们可以使用[make _ classification()](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)sci kit-learn 函数定义一个合成的不平衡两类类别数据集。我们将生成 10，000 个少数与多数类比例大约为 1:100 的示例。

```py
...
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
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
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
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

![Scatter Plot of Binary Classification Dataset With 1 to 100 Class Imbalance](img/887db0673d3ed330186d46e0c141ff6e.png)

1 到 100 类不平衡的二进制类别数据集的散点图

## 用于分类的 XGBoost 模型

XGBoost 是**极限梯度提升**的简称，是随机梯度提升机学习算法的高效实现。

随机梯度提升算法，也称为梯度提升机或树增强，是一种强大的机器学习技术，在一系列具有挑战性的机器学习问题上表现良好，甚至最好。

> 树提升已经被证明在许多标准分类基准上给出了最先进的结果。

——[xboost:一个可扩展的树提升系统](https://arxiv.org/abs/1603.02754)，2016。

这是一个决策树算法的集合，其中新的树修复了那些已经是模型一部分的树的错误。树被添加，直到模型没有进一步的改进。

XGBoost 提供了随机梯度提升算法的高效实现，并提供了对一组模型超参数的访问，这些参数旨在提供对模型训练过程的控制。

> XGBoost 成功背后最重要的因素是它在所有场景中的可扩展性。该系统在单台机器上的运行速度比现有的流行解决方案快十倍以上，并且在分布式或内存有限的环境中可扩展到数十亿个示例。

——[xboost:一个可扩展的树提升系统](https://arxiv.org/abs/1603.02754)，2016。

XGBoost 是一种有效的机器学习模型，即使在类分布有偏差的数据集上也是如此。

在对不平衡分类的 XGBoost 算法进行任何修改或调整之前，测试默认的 XGBoost 模型并建立表现基线是很重要的。

虽然 XGBoost 库有自己的 Python API，但是我们可以通过 [XGBClassifier 包装类](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)将 XGBoost 模型与 Sklearn API 一起使用。模型的一个实例可以像任何其他用于模型评估的 Sklearn 类一样被实例化和使用。例如:

```py
...
# define model
model = XGBClassifier()
```

我们将使用重复交叉验证来评估模型，重复三次 [10 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)。

模型表现将使用重复和所有折叠的平均值[曲线下的 ROC 面积](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/) (ROC AUC)来报告。

```py
...
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.5f' % mean(scores))
```

将这些联系在一起，下面列出了在不平衡分类问题上定义和评估默认 XGBoost 模型的完整示例。

```py
# fit xgboost on an imbalanced classification dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# define model
model = XGBClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.5f' % mean(scores))
```

运行该示例会评估不平衡数据集上的默认 XGBoost 模型，并报告平均 ROC AUC。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

我们可以看到模型有技巧，实现了大于 0.5 的 ROC AUC，在这种情况下实现了 0.95724 的平均得分。

```py
Mean ROC AUC: 0.95724
```

这为针对默认 XGBoost 算法执行的任何超参数调整提供了一个比较基准。

## 类别不平衡的加权扩展

尽管 XGBoost 算法在解决各种挑战性问题时表现良好，但它提供了大量超参数，其中许多参数需要进行调整，以便在给定的数据集上充分利用该算法。

该实现提供了一个超参数，用于针对不平衡分类问题调整算法的行为；这是**刻度 _ pos _ 重量**超参数。

默认情况下， *scale_pos_weight* 超参数设置为值 1.0，在提升决策树时，相对于负示例，具有权衡正示例的效果。对于不平衡的二进制类别数据集，负类指多数类(类 0)，正类指少数类(类 1)。

XGBoost 被训练成最小化损失函数，梯度提升中的“*梯度*”指的是该损失函数的陡度，例如误差量。一个小的梯度意味着一个小的误差，反过来，一个修正误差的模型的小变化。训练期间的大误差梯度反过来导致大的校正。

*   **小梯度**:对模型的小误差或修正。
*   **大梯度**:对模型误差或修正较大。

梯度被用作拟合随后添加的树的基础，以增强或校正由决策树集合的现有状态产生的错误。

*scale_pos_weight* 值用于缩放正类的梯度。

这具有模型在正类训练期间产生的缩放误差的影响，并且鼓励模型过度校正它们。反过来，这可以帮助模型在对正类进行预测时获得更好的表现。推得太远，可能会导致模型过拟合正类，代价是负类或两个类的表现都变差。

因此， *scale_pos_weight* 可以用来训练一个类加权或成本敏感版本的 XGBoost 进行不平衡分类。

为 *scale_pos_weight* 超参数设置的合理默认值是类分布的倒数。例如，对于少数与多数类中示例比例为 1 比 100 的数据集， *scale_pos_weight* 可以设置为 100。这将使模型对少数类(正类)造成的分类错误产生 100 倍的影响，反过来，比多数类造成的错误产生 100 倍的修正。

例如:

```py
...
# define model
model = XGBClassifier(scale_pos_weight=100)
```

XGBoost 文档提出了一种快速估计该值的方法，该方法使用训练数据集作为多数类中的示例总数除以少数类中的示例总数。

*   scale _ pos _ weight = total _ 负值 _ 示例/total _ 正值 _ 示例

例如，我们可以为我们的合成类别数据集计算这个值。考虑到我们用来定义数据集的权重，我们预计该值约为 100，或者更准确地说，99。

```py
...
# count examples in each class
counter = Counter(y)
# estimate scale_pos_weight value
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)
```

下面列出了估计*刻度 _ 位置 _ 重量* XGBoost 超参数值的完整示例。

```py
# estimate a value for the scale_pos_weight xgboost hyperparameter
from sklearn.datasets import make_classification
from collections import Counter
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# count examples in each class
counter = Counter(y)
# estimate scale_pos_weight value
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)
```

运行该示例会创建数据集，并将 *scale_pos_weight* 超参数的值估计为 99，如我们所料。

```py
Estimate: 99.000
```

我们将在 XGBoost 模型的配置中直接使用这个值，并使用重复的 k 倍交叉验证来评估它在数据集上的表现。

我们预计 ROC AUC 会有一些改进，尽管根据数据集的难度和所选的 XGBoost 模型配置，这并不能保证。

下面列出了完整的示例。

```py
# fit balanced xgboost on an imbalanced classification dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# define model
model = XGBClassifier(scale_pos_weight=99)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.5f' % mean(scores))
```

运行该示例准备合成不平衡类别数据集，然后使用重复交叉验证评估 XGBoost 训练算法的类加权版本。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到表现的适度提升，从上一节中 *scale_pos_weight=1* 时的约 0.95724 的 ROC AUC 提升到 *scale_pos_weight=99* 时的 0.95990 的值。

```py
Mean ROC AUC: 0.95990
```

## 调整类别加权超参数

设置*刻度 _ 位置 _ 重量*的启发式方法在许多情况下都是有效的。

然而，使用不同的类权重可以获得更好的表现，这也将取决于用于评估模型的表现度量的选择。

在本节中，我们将网格搜索一系列不同的类权重，以获得类加权的 XGBoost，并发现哪一个导致最佳的 ROC AUC 分数。

我们将对正类尝试以下权重:

*   1(默认)
*   Ten
*   Twenty-five
*   Fifty
*   Seventy-five
*   99(推荐)
*   One hundred
*   One thousand

这些可以定义为[网格搜索参数，如下所示:](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

```py
...
# define grid
weights = [1, 10, 25, 50, 75, 99, 100, 1000]
param_grid = dict(scale_pos_weight=weights)
```

我们可以使用重复交叉验证对这些参数执行网格搜索，并使用 ROC AUC 估计模型表现:

```py
...
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
```

一旦执行，我们可以将最佳配置以及所有结果总结如下:

```py
...
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

将这些联系在一起，下面的例子在不平衡的数据集上为 XGBoost 算法搜索八个不同的正类权重。

我们可能会认为启发式类加权是表现最好的配置。

```py
# grid search positive class weights with xgboost for imbalance classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=7)
# define model
model = XGBClassifier()
# define grid
weights = [1, 10, 25, 50, 75, 99, 100, 1000]
param_grid = dict(scale_pos_weight=weights)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X, y)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

运行该示例使用重复的 k 倍交叉验证评估每个正类别权重，并报告最佳配置和相关的平均 ROC AUC 分数。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到 *scale_pos_weight=99* 正类加权达到了最好的平均 ROC 得分。这与一般启发式算法的配置相匹配。

有趣的是，几乎所有大于默认值 1 的值都有更好的平均 ROC AUC，即使是激进的值 1000。有趣的是，99 的值比 100 的值表现得更好，如果我没有按照 XGBoost 文档中的建议计算启发式方法，我可能会使用 100。

```py
Best: 0.959901 using {'scale_pos_weight': 99}
0.957239 (0.031619) with: {'scale_pos_weight': 1}
0.958219 (0.027315) with: {'scale_pos_weight': 10}
0.958278 (0.027438) with: {'scale_pos_weight': 25}
0.959199 (0.026171) with: {'scale_pos_weight': 50}
0.959204 (0.025842) with: {'scale_pos_weight': 75}
0.959901 (0.025499) with: {'scale_pos_weight': 99}
0.959141 (0.025409) with: {'scale_pos_weight': 100}
0.958761 (0.024757) with: {'scale_pos_weight': 1000}
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [XGBoost:一个可扩展的树木提升系统](https://arxiv.org/abs/1603.02754)，2016。

### 书

*   [从不平衡数据集中学习](https://amzn.to/307Xlva)，2018。
*   [不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

### 蜜蜂

*   [sklearn . datasets . make _ classification API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
*   [xgboost。xgbcclassifier API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)。
*   [XGBoost 参数，API 文档](https://xgboost.readthedocs.io/en/latest/parameter.html)。
*   [参数调整注释，应用编程接口文档](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)。

## 摘要

在本教程中，您发现了用于不平衡分类的加权 XGBoost。

具体来说，您了解到:

*   梯度提升如何从高层次工作，以及如何开发一个用于分类的 XGBoost 模型。
*   如何修改 XGBoost 训练算法，以在训练过程中加权与正类重要性成比例的误差梯度。
*   如何为 XGBoost 训练算法配置正类权重，如何网格搜索不同的配置。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。