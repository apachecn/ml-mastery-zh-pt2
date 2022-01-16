# 如何校准不平衡分类的概率

> 原文：<https://machinelearningmastery.com/probability-calibration-for-imbalanced-classification/>

最后更新于 2020 年 8 月 21 日

许多机器学习模型能够预测类成员的概率或类似概率的分数。

概率为评估和比较模型提供了所需的粒度级别，特别是在不平衡分类问题上，像 [ROC 曲线](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)这样的工具用于解释预测，ROC AUC 度量用于比较模型表现，两者都使用概率。

不幸的是，许多模型预测的概率或类似概率的分数没有被校准。这意味着他们在某些情况下可能过于自信，而在其他情况下则信心不足。更糟糕的是，不平衡分类任务中存在的严重倾斜的类分布可能导致预测概率中甚至更多的偏差，因为它们过度倾向于预测多数类。

因此，在评估非线性机器学习模型的表现之前，校准其预测概率通常是一个好主意。此外，在处理不平衡的数据集时，通常校准概率是一种好的做法，即使是像逻辑回归这样的模型，当类标签平衡时，也可以预测校准良好的概率。

在本教程中，您将发现如何校准不平衡分类的预测概率。

完成本教程后，您将知道:

*   对于不平衡的分类问题，需要校准概率来从模型中获得最大收益。
*   如何校准非线性模型(如支持向量机、决策树和 KNN)的预测概率。
*   如何在具有偏斜类分布的数据集上网格搜索不同的概率校准方法？

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Calibrate Probabilities for Imbalanced Classification](img/6ed5f60197c1c233f9c23c7400311deb.png)

如何校准不平衡分类的概率
图片由[丹尼斯·贾维斯](https://flickr.com/photos/archer10/33621284946/)提供，版权所有。

## 教程概述

本教程分为五个部分；它们是:

1.  未校准概率问题
2.  如何校准概率
3.  具有校准概率的 SVM
4.  具有校准概率的决策树
5.  KNN 网格搜索概率校准

## 未校准概率问题

许多机器学习算法可以预测一个概率或一个类似概率的分数来表示类别成员。

例如，逻辑回归可以直接预测类别成员的概率，支持向量机可以预测不是概率但可以解释为概率的分数。

概率可以用来衡量那些需要概率预测的问题的不确定性。在不平衡分类中尤其如此，在不平衡分类中，就评估和选择模型而言，清晰的类标签通常是不够的。预测的概率为更精细的模型评估和选择提供了基础，例如通过使用 [ROC 和精确-召回诊断图](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)、ROC AUC 等指标以及阈值移动等技术。

因此，当处理不平衡的分类任务时，使用预测概率的机器学习模型通常是优选的。问题是很少有机器学习模型校准了概率。

> ……为了被有效地解释为概率，分数应该被校准。

—第 57 页，[从不平衡数据集](https://amzn.to/307Xlva)中学习，2018。

校准概率意味着概率反映真实事件的可能性。

如果你考虑到在分类中，我们有正确与否的类标签，而不是概率，这可能会令人困惑。澄清一下，回想一下在二进制分类中，我们预测的是 0 类或 1 类的阴性或阳性病例。如果以 0.8 的概率预测 100 个例子，那么如果概率被校准，80%的例子将具有类别 1，20%将具有类别 0。这里，[校准](https://en.wikipedia.org/wiki/Calibration_(statistics))是预测概率与阳性病例发生的一致性。

未校准的概率表明概率分数存在偏差，这意味着概率在某些情况下过于自信或不够自信。

*   **校准概率**。概率与事件的真实可能性相匹配。
*   **未校准概率**。概率过于自信和/或不够自信。

这对于没有使用概率框架训练的机器学习模型和具有偏斜分布的训练数据(如不平衡分类任务)来说是常见的。

未校准概率有两个主要原因；它们是:

*   未使用概率框架训练的算法。
*   训练数据中的偏差。

很少有机器学习算法产生校准的概率。这是因为对于预测校准概率的模型，它必须明确地在概率框架下训练，例如最大似然估计。提供校准概率的一些算法示例包括:

*   逻辑回归。
*   线性判别分析。
*   朴素贝叶斯。
*   人工神经网络。

许多算法要么预测一个类似概率的分数，要么预测一个类标签，为了产生一个类似概率的分数，这些算法必须被强制。因此，这些算法通常需要在使用前校准它们的“T0”概率。例子包括:

*   支持向量机。
*   决策树。
*   决策树的集合(装袋、随机森林、梯度提升)。
*   k-最近邻居。

训练数据集中的偏差，例如类分布中的偏差，意味着模型将自然地预测多数类的概率高于少数类的平均概率。

问题是，模型可能会过度补偿，把太多的焦点放在多数类上。这甚至适用于通常产生校准概率的模型，如逻辑回归。

> …尽管表面上整体校准良好，但在不平衡场景中通过监督学习获得的类概率估计系统地低估了少数类实例的概率。

——[不平衡数据的类概率估计不可靠(以及如何修复)](https://ieeexplore.ieee.org/abstract/document/6413859)，2012。

## 如何校准概率

概率通过重新调整它们的值来校准，以便它们更好地匹配训练数据中观察到的分布。

> …我们希望估计的类别概率能够反映样本的真实潜在概率。也就是说，预测的类概率(或类概率值)需要很好地校准。为了更好地校准，概率必须有效地反映感兴趣事件的真实可能性。

—第 249 页，[应用预测建模](https://amzn.to/2kXE35G)，2013 年。

对训练数据进行概率预测，并将概率分布与预期概率进行比较，并进行调整以提供更好的匹配。这通常涉及分割训练数据集，并使用一部分来训练模型，另一部分作为验证集来缩放概率。

有两种主要的技术来缩放预测概率；它们是[普拉特缩放](https://en.wikipedia.org/wiki/Platt_scaling)和[等张回归](https://en.wikipedia.org/wiki/Isotonic_regression)。

*   **普拉特缩放**。转换概率的逻辑回归模型。
*   **等渗回归**。转换概率的加权最小二乘回归模型

普拉特缩放是一种更简单的方法，它是为了将支持向量机的输出缩放到概率值而开发的。它包括学习逻辑回归模型来执行分数到校准概率的转换。等渗回归是一种更复杂的加权最小二乘回归模型。它需要更多的训练数据，尽管它也更强大、更通用。这里，等渗简单地指原始概率到重新标度值的单调递增映射。

> 当预测概率中的失真为乙状线形状时，普拉特缩放最有效。等渗回归是一种更强大的校准方法，可以校正任何单调失真。

——[用监督学习预测好概率](https://dl.acm.org/citation.cfm?id=1102430)，2005。

Sklearn 库通过[校准分类器类](https://Sklearn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)提供对用于校准概率的普拉特标度和等渗回归方法的访问。

这是一个模型的包装(像 SVM)。首选的缩放技术通过“*方法*参数定义，该参数可以是“*乙状结肠*”(普拉特缩放)或“*等张*”(等张回归)。

交叉验证用于缩放模型的预测概率，通过“ *cv* ”参数设置。这意味着模型在训练集上是合适的，在测试集上是校准的，并且这个过程对于 k 倍重复 k 次，其中预测概率在运行中是平均的。

设置“ *cv* ”参数取决于可用的数据量，尽管可以使用 3 或 5 等值。重要的是，分割是分层的，这在不平衡数据集上使用概率校准时很重要，因为不平衡数据集通常很少有正类的例子。

```py
...
# example of wrapping a model with probability calibration
model = ...
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
```

现在我们知道了如何校准概率，让我们来看一些在不平衡类别数据集上校准模型概率的例子。

## 具有校准概率的 SVM

在本节中，我们将回顾如何在不平衡类别数据集上校准 SVM 模型的概率。

首先，让我们使用[make _ classion()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)定义一个数据集。我们将生成 10，000 个示例，其中 99%属于负案例(类别 0)，1%属于正案例(类别 1)。

```py
...
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
```

接下来，我们可以定义一个具有默认超参数的 SVM。这意味着模型不会针对数据集进行调整，但会提供一致的比较基础。

```py
...
# define model
model = SVC(gamma='scale')
```

然后，我们可以使用重复分层 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)在数据集上评估该模型，重复三次，每次重复 10 倍。

我们将使用 ROC AUC 评估模型，并计算所有重复和折叠的平均得分。中华民国 AUC 将利用 SVM 提供的未经校准的类概率分数。

```py
...
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

将这些联系在一起，完整的示例如下所示。

```py
# evaluate svm with uncalibrated probabilities for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = SVC(gamma='scale')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例在不平衡类别数据集上用未校准的概率评估 SVM。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到 SVM 实现了约 0.804 的 ROC AUC。

```py
Mean ROC AUC: 0.804
```

接下来，我们可以尝试使用*校准分类器*类来包装 SVM 模型并预测校准概率。

我们使用分层的 10 倍交叉验证来评估模型；这意味着每个文件夹上有 9000 个示例用于训练，1000 个用于测试。

通过*校准分类器*和 3 倍，一倍的 9000 个例子将被分成 6000 个用于训练模型，3000 个用于校准概率。这不会留下少数类的许多例子，例如 10 倍交叉验证中的 90/10，然后校准 60/30。

使用校准时，重要的是根据您选择的模型评估方案来计算这些数字，或者调整折叠数量以确保数据集足够大，或者甚至在需要时切换到更简单的训练/测试分割，而不是交叉验证。可能需要实验。

我们将像以前一样定义 SVM 模型，然后用等渗回归定义*校准分类器*，然后通过重复分层 k 倍交叉验证评估校准模型。

```py
...
# define model
model = SVC(gamma='scale')
# wrap the model
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
```

因为默认情况下 SVM 概率没有被校准，我们期望校准它们会导致 ROC AUC 的改进，该 AUC 基于它们的概率明确评估模型。

将这些联系在一起，下面列出了用校准概率评估 SVM 的完整例子。

```py
# evaluate svm with calibrated probabilities for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = SVC(gamma='scale')
# wrap the model
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(calibrated, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例在不平衡类别数据集上评估具有校准概率的 SVM。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到 SVM 实现了 ROC AUC 从约 0.804 到约 0.875 的提升。

```py
Mean ROC AUC: 0.875
```

概率校准可以结合对算法或数据集的其他修改来评估，以解决倾斜的类分布。

例如，SVM 提供了“ *class_weight* ”参数，该参数可以设置为“ *balanced* ”来调整保证金，以有利于少数族裔。我们可以将这一变化纳入 SVM 模型，并校准概率，我们可能会看到模型技能的进一步提升；例如:

```py
...
# define model
model = SVC(gamma='scale', class_weight='balanced')
```

将这些联系在一起，下面列出了具有校准概率的类加权 SVM 的完整示例。

```py
# evaluate weighted svm with calibrated probabilities for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = SVC(gamma='scale', class_weight='balanced')
# wrap the model
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(calibrated, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例在不平衡类别数据集上评估具有校准概率的类加权 SVM。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到，SVM 实现了中华民国 AUC 从约 0.875 到约 0.966 的进一步提升。

```py
Mean ROC AUC: 0.966
```

## 具有校准概率的决策树

[决策树](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)是另一种不自然产生概率的高效机器学习。

相反，直接预测类别标签，并且可以基于落入为新示例预测的树叶中的训练数据集中的示例的分布来估计类似概率的分数。因此，来自决策树的概率分数应该在被评估和用于选择模型之前被校准。

我们可以使用[决策树分类器 Sklearn 类](https://Sklearn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)来定义决策树。

该模型可以在我们的合成不平衡类别数据集上用未校准的概率进行评估。

下面列出了完整的示例。

```py
# evaluate decision tree with uncalibrated probabilities for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = DecisionTreeClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例在不平衡类别数据集上评估具有未校准概率的决策树。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到决策树实现了大约 0.842 的 ROC AUC。

```py
Mean ROC AUC: 0.842
```

然后，我们可以使用校准包装器评估相同的模型。

在这种情况下，我们将使用通过将“*方法*”参数设置为“ *sigmoid* 而配置的普拉特缩放方法。

```py
...
# wrap the model
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
```

下面列出了用校准概率评估不平衡分类决策树的完整示例。

```py
# decision tree with calibrated probabilities for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = DecisionTreeClassifier()
# wrap the model
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(calibrated, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估不平衡类别数据集上具有校准概率的决策树。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到决策树实现了 ROC AUC 从大约 0.842 到大约 0.859 的提升。

```py
Mean ROC AUC: 0.859
```

## 用 KNN 进行网格搜索概率校准

概率校准对方法和使用方法的方式都很敏感。

因此，最好在模型上测试一套不同的概率校准方法，以便发现最适合数据集的方法。一种方法是将校准方法和交叉验证折叠视为超参数，并对其进行调整。在本节中，我们将研究使用网格搜索来调整这些超参数。

k-最近邻算法，或称 KNN 算法，是另一种非线性机器学习算法，它直接预测类标签，并且必须被修改以产生类似概率的分数。这通常涉及到在邻域中使用类标签的分布。

我们可以使用默认邻域大小为 5 的 [KNeighborsClassifier 类](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)在合成不平衡类别数据集上评估具有未校准概率的 KNN。

下面列出了完整的示例。

```py
# evaluate knn with uncalibrated probabilities for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = KNeighborsClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例在不平衡类别数据集上用未校准的概率评估 KNN。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到 KNN 实现了约 0.864 的 ROC AUC。

```py
Mean ROC AUC: 0.864
```

知道概率依赖于邻域大小并且是未校准的，我们期望一些校准将使用 ROC AUC 提高模型的表现。

我们将使用[网格搜索不同的配置，而不是抽查*校准分类器*类的一个配置。](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

首先，模型和校准包装器的定义与之前一样。

```py
...
# define model
model = KNeighborsClassifier()
# wrap the model
calibrated = CalibratedClassifierCV(model)
```

我们将测试“*乙状结肠*”和“*等张*”“法”值，以及[2，3，4]中不同的“ *cv* 值。回想一下“ *cv* ”控制用于估计校准概率的训练数据集的分割。

我们可以将参数网格定义为一个带有参数名称的字典，我们想要调整*校准分类器*并提供要尝试的值列表。这将测试 3 * 2 或 6 种不同的组合。

```py
...
# define grid
param_grid = dict(cv=[2,3,4], method=['sigmoid','isotonic'])
```

然后，我们可以用模型和参数网格定义 *GridSearchCV* ，并使用我们之前使用的相同的重复分层 k 倍交叉验证来评估每个参数组合。

```py
...
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=calibrated, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X, y)
```

评估后，我们将总结 ROC AUC 最高的配置，然后列出所有组合的结果。

```py
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

将这些联系在一起，下面列出了使用 KNN 模型进行不平衡分类的网格搜索概率校准的完整示例。

```py
# grid search probability calibration with knn for imbalance classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = KNeighborsClassifier()
# wrap the model
calibrated = CalibratedClassifierCV(model)
# define grid
param_grid = dict(cv=[2,3,4], method=['sigmoid','isotonic'])
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=calibrated, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
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

运行该示例使用不平衡类别数据集上的一组不同类型的校准概率来评估 KNN。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到最佳结果是用 2 的“ *cv* ”和“*方法*的“*等张*值”获得的，平均 ROC AUC 约为 0.895，比没有校准时的 0.864 有所提高。

```py
Best: 0.895120 using {'cv': 2, 'method': 'isotonic'}
0.895084 (0.062358) with: {'cv': 2, 'method': 'sigmoid'}
0.895120 (0.062488) with: {'cv': 2, 'method': 'isotonic'}
0.885221 (0.061373) with: {'cv': 3, 'method': 'sigmoid'}
0.881924 (0.064351) with: {'cv': 3, 'method': 'isotonic'}
0.881865 (0.065708) with: {'cv': 4, 'method': 'sigmoid'}
0.875320 (0.067663) with: {'cv': 4, 'method': 'isotonic'}
```

这提供了一个模板，您可以使用它来评估自己模型上的不同概率校准配置。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何以及何时使用带有 scikit 的校准分类模型-学习](https://machinelearningmastery.com/calibrated-classification-model-in-Sklearn/)

### 报纸

*   [用监督学习预测好概率](https://dl.acm.org/citation.cfm?id=1102430)，2005。
*   [不平衡数据的类概率估计不可靠(以及如何修正)](https://ieeexplore.ieee.org/abstract/document/6413859)，2012。

### 书

*   [从不平衡数据集中学习](https://amzn.to/307Xlva)，2018。
*   [不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。
*   [应用预测建模](https://amzn.to/2kXE35G)，2013。

### 蜜蜂

*   [硬化。校准。校准后的分类 CVI API](https://Sklearn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)。
*   [硬化. svm.SVC API](https://Sklearn.org/stable/modules/generated/sklearn.svm.SVC.html) 。
*   [硬化. tree .决策树分类器 API](https://Sklearn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) 。
*   [sklearn . neighborsclassifier API](https://Sklearn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)。
*   [sklearn.model_selection。GridSearchCV API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 。

### 文章

*   [校准(统计)，维基百科](https://en.wikipedia.org/wiki/Calibration_(statistics))。
*   [概率分类，维基百科](https://en.wikipedia.org/wiki/Probabilistic_classification)。
*   [普拉特缩放，维基百科](https://en.wikipedia.org/wiki/Platt_scaling)。
*   [等渗回归，维基百科](https://en.wikipedia.org/wiki/Isotonic_regression)。

## 摘要

在本教程中，您发现了如何校准不平衡分类的预测概率。

具体来说，您了解到:

*   对于不平衡的分类问题，需要校准概率来从模型中获得最大收益。
*   如何校准非线性模型(如支持向量机、决策树和 KNN)的预测概率。
*   如何在类分布偏斜的数据集上网格搜索不同的概率校准方法？

你有什么问题吗？
在下面的评论中提问，我会尽力回答。