# 不平衡分类的成本敏感逻辑回归

> 原文：<https://machinelearningmastery.com/cost-sensitive-logistic-regression/>

最后更新于 2020 年 10 月 26 日

逻辑回归不直接支持不平衡分类。

相反，用于拟合逻辑回归模型的训练算法必须被修改以考虑偏斜分布。这可以通过指定用于影响训练期间更新逻辑回归系数的数量的类加权配置来实现。

加权可以对多数类的例子中的错误较少地惩罚模型，而对少数类的例子中的错误较多地惩罚模型。结果是逻辑回归的一个版本，在不平衡分类任务上表现更好，通常被称为成本敏感或加权逻辑回归。

在本教程中，您将发现不平衡分类的成本敏感逻辑回归。

完成本教程后，您将知道:

*   标准逻辑回归如何不支持不平衡分类？
*   如何在拟合系数时用类权重修正逻辑回归来加权模型误差？
*   如何为逻辑回归配置类权重，如何网格搜索不同的类权重配置。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **更新 2020 年 2 月**:修正了重量计算中的错别字。
*   **2020 年 10 月更新**:修正了余额比例描述中的错别字。

![Cost-Sensitive Logistic Regression for Imbalanced Classification](img/be2c4c7262f0de8ec0b0ad69fab9cba1.png)

不平衡分类的成本敏感逻辑回归
图片由[海军](https://flickr.com/photos/sinava/32927010408/)提供，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  不平衡类别数据集
2.  不平衡分类的逻辑回归
3.  基于 Sklearn 的加权逻辑回归
4.  网格搜索加权逻辑回归

## 不平衡类别数据集

在我们深入研究不平衡分类的逻辑回归的修改之前，让我们首先定义一个不平衡类别数据集。

我们可以使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)定义一个合成的不平衡两类类别数据集。我们将生成 10，000 个少数与多数类比例大约为 1:100 的示例。

```py
...
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
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
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
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

![Scatter Plot of Binary Classification Dataset With 1 to 100 Class Imbalance](img/58c465ce34eb523da84a8ba6613e03a3.png)

1 到 100 类不平衡的二进制类别数据集的散点图

接下来，我们可以在数据集上拟合标准逻辑回归模型。

我们将使用重复交叉验证来评估模型，重复三次 [10 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)。模式表现将使用重复和所有折叠的平均[曲线下面积(ROC AUC)](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/) 来报告。

```py
...
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

将这些联系在一起，下面列出了不平衡分类问题的标准逻辑回归的完整例子。

```py
# fit a logistic regression model on an imbalanced classification dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估不平衡数据集上的标准逻辑回归模型，并报告平均 ROC AUC。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

我们可以看到模型有技巧，实现了 0.5 以上的 ROC AUC，在这种情况下实现了 0.985 的平均得分。

```py
Mean ROC AUC: 0.985
```

这为标准逻辑回归算法的任何修改提供了比较基准。

## 不平衡分类的逻辑回归

逻辑回归是二分类任务的有效模型，尽管默认情况下，它在不平衡分类中无效。

可以修改逻辑回归，使其更适合逻辑回归。

使用优化算法拟合逻辑回归算法的系数，该优化算法最小化训练数据集上模型的[负对数似然(损失)](https://machinelearningmastery.com/logistic-regression-with-maximum-likelihood-estimation/)。

*   将总和 I 最小化为 n-(log(yhat _ I)* y _ I+log(1–yhat _ I)*(1–y _ I))

这包括重复使用模型进行预测，然后在减少模型损失的方向上调整系数。

可以修改给定系数集的损失计算，以考虑类别平衡。

默认情况下，每个类别的误差可以被认为具有相同的权重，比如 1.0。这些权重可以根据每个类别的重要性进行调整。

*   将总和 I 最小化为 n-(w0 * log(yhat _ I)* y _ I+w1 * log(1–yhat _ I)*(1–y _ I))

加权被应用于损失，使得较小的权重值导致较小的误差值，并且反过来，较少更新模型系数。较大的权重值导致较大的误差计算，进而导致模型系数的更多更新。

*   **小权重**:重要性小，对模型系数更新少。
*   **大权重**:更重要，对模型系数更新更多。

因此，逻辑回归的修改版本被称为加权逻辑回归、类加权逻辑回归或成本敏感逻辑回归。

权重有时被称为重要性权重。

虽然实现起来很简单，但加权逻辑回归的挑战在于为每个类选择权重。

## 基于 Sklearn 的加权逻辑回归

Sklearn Python 机器学习库提供了支持类加权的逻辑回归的实现。

[后勤分类类](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)提供了可以指定为模型超参数的 class_weight 参数。class_weight 是一个字典，它定义了每个类标签(例如 0 和 1)以及在拟合模型时计算负对数似然时应用的权重。

例如，每个类别 0 和 1 的 1 比 1 权重可以定义如下:

```py
...
# define model
weights = {0:1.0, 1:1.0}
model = LogisticRegression(solver='lbfgs', class_weight=weights)
```

类别称重可以多种方式定义；例如:

*   **领域专长**，通过与主题专家交谈确定。
*   **调谐**，由超参数搜索如网格搜索确定。
*   **启发式**，使用一般最佳实践指定。

使用类别权重的最佳实践是使用训练数据集中类别分布的倒数。

例如，训练数据集的类分布是少数类与多数类的比例为 1:100。该比率的倒数可以用于多数类的 1 和少数类的 100；例如:

```py
...
# define model
weights = {0:1.0, 1:100.0}
model = LogisticRegression(solver='lbfgs', class_weight=weights)
```

我们也可以使用分数来定义相同的比率，并获得相同的结果；例如:

```py
...
# define model
weights = {0:0.01, 1:1.0}
model = LogisticRegression(solver='lbfgs', class_weight=weights)
```

我们可以使用上一节中定义的相同评估过程来评估带有类权重的逻辑回归算法。

我们期望类别加权的逻辑回归比没有任何类别加权的标准逻辑回归表现得更好。

下面列出了完整的示例。

```py
# weighted logistic regression model on an imbalanced classification dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
# define model
weights = {0:0.01, 1:1.0}
model = LogisticRegression(solver='lbfgs', class_weight=weights)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例准备合成不平衡类别数据集，然后使用重复交叉验证评估逻辑回归的类加权版本。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

报告了平均 ROC AUC 评分，在这种情况下显示出比逻辑回归的未加权版本更好的评分，0.989 比 0.985。

```py
Mean ROC AUC: 0.989
```

Sklearn 库为类加权提供了最佳实践启发式的实现。

它通过 [compute_class_weight()函数](https://Sklearn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)实现，计算如下:

*   n _ samples/(n _ class * n _ samples _ with _ class)

我们可以在数据集上手动测试这个计算。例如，我们的数据集中有 10，000 个示例，0 类有 9900 个，1 类有 100 个。

类别 0 的权重计算如下:

*   加权= n _ samples/(n _ class * n _ samples _ with _ class)
*   权重= 10000 / (2 * 9900)
*   权重= 10000 / 19800
*   权重= 0.05

类别 1 的权重计算如下:

*   加权= n _ samples/(n _ class * n _ samples _ with _ class)
*   权重= 10000 / (2 * 100)
*   权重= 10000 / 200
*   权重= 50

我们可以通过调用 *compute_class_weight()* 函数并将 *class_weight* 指定为“*平衡*来确认这些计算例如:

```py
# calculate heuristic class weighting
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets import make_classification
# generate 2 class dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
# calculate class weighting
weighting = compute_class_weight('balanced', [0,1], y)
print(weighting)
```

运行该示例，我们可以看到，对于类 0，我们可以实现大约 0.5 的权重，对于类 1，我们可以实现 50 的权重。

这些值与我们的手动计算相匹配。

```py
[ 0.50505051 50\. ]
```

这些值也与我们上面的启发式计算相匹配，用于反演训练数据集中的类分布比率；例如:

*   0.5:50 == 1:100

通过将 *class_weight* 参数设置为“平衡”，我们可以将默认的类平衡直接用于[logisticreduce](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)类例如:

```py
...
# define model
model = LogisticRegression(solver='lbfgs', class_weight='balanced')
```

下面列出了完整的示例。

```py
# weighted logistic regression for class imbalance with heuristic weights
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs', class_weight='balanced')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

运行该示例给出的平均 ROC AUC 与我们通过手动指定反向类比率获得的相同。

```py
Mean ROC AUC: 0.989
```

## 网格搜索加权逻辑回归

使用与训练数据成反比的类权重只是一种启发。

使用不同的类权重可以获得更好的表现，这也将取决于用于评估模型的表现度量的选择。

在本节中，我们将网格搜索一系列不同的加权逻辑回归的类别权重，并发现哪一个导致最佳的 ROC AUC 分数。

我们将对类别 0 和 1 尝试以下权重:

*   {0:100,1:1}
*   {0:10,1:1}
*   {0:1,1:1}
*   {0:1,1:10}
*   {0:1,1:100}

这些可以定义为[网格搜索参数，如下所示:](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

```py
...
# define grid
balance = [{0:100,1:1}, {0:10,1:1}, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}]
param_grid = dict(class_weight=balance)
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

将这些联系在一起，下面的示例在不平衡的数据集上搜索五个不同的类别权重进行逻辑回归。

我们可能会认为启发式类加权是表现最好的配置。

```py
# grid search class weights with logistic regression for imbalance classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# define grid
balance = [{0:100,1:1}, {0:10,1:1}, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}]
param_grid = dict(class_weight=balance)
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

运行该示例使用重复的 k 倍交叉验证评估每个类别权重，并报告最佳配置和相关的平均 ROC AUC 分数。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到 1:100 的多数对少数类权重获得了最佳的平均 ROC 分数。这与一般启发式算法的配置相匹配。

探索更严格的类别权重，看看它们对平均 ROC AUC 评分的影响，可能会很有趣。

```py
Best: 0.989077 using {'class_weight': {0: 1, 1: 100}}
0.982498 (0.016722) with: {'class_weight': {0: 100, 1: 1}}
0.983623 (0.015760) with: {'class_weight': {0: 10, 1: 1}}
0.985387 (0.013890) with: {'class_weight': {0: 1, 1: 1}}
0.988044 (0.010384) with: {'class_weight': {0: 1, 1: 10}}
0.989077 (0.006865) with: {'class_weight': {0: 1, 1: 100}}
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [罕见事件数据中的逻辑回归](https://dash.harvard.edu/handle/1/4125045)，2001。
*   [基于选择样本的选择概率估计](https://www.jstor.org/stable/1914121)，1977。

### 书

*   [从不平衡数据集中学习](https://amzn.to/307Xlva)，2018。
*   [不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

### 蜜蜂

*   [sklearn . utils . class _ weight . compute _ class _ weight API](https://Sklearn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)。
*   [sklearn.linear_model。物流配送应用编程接口](https://Sklearn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。
*   [sklearn.model_selection。GridSearchCV API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 。

## 摘要

在本教程中，您发现了不平衡分类的成本敏感逻辑回归。

具体来说，您了解到:

*   标准逻辑回归如何不支持不平衡分类？
*   如何在拟合系数时用类权重修正逻辑回归来加权模型误差？
*   如何为逻辑回归配置类权重，如何网格搜索不同的类权重配置。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。