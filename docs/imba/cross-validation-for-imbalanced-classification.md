# 如何为不平衡分类修复 K 折交叉验证

> 原文：<https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/>

最后更新于 2020 年 7 月 31 日

模型评估包括使用可用的数据集来拟合模型，并在对未知示例进行预测时估计其表现。

这是一个具有挑战性的问题，因为用于拟合模型的训练数据集和用于评估模型的测试集都必须足够大，并且能够代表潜在的问题，以便对模型表现的最终估计不会过于乐观或悲观。

模型评估最常用的两种方法是训练/测试分割和 k-fold 交叉验证程序。总的来说，这两种方法都非常有效，尽管它们可能导致误导性结果，并且当用于具有严重类别不平衡的分类问题时，可能会失败。取而代之的是，这些技术必须被修改，以通过类别标签对采样进行分层，这被称为分层的列车测试分割或**分层的 k 倍交叉验证**。

在本教程中，您将发现如何在不平衡的数据集上评估分类器模型。

完成本教程后，您将知道:

*   使用训练/测试分割和交叉验证在数据集上评估分类器的挑战。
*   当在不平衡的数据集上评估分类器时，k-fold 交叉验证和训练-测试分割的简单应用将如何失败。
*   如何使用修改后的 k-fold 交叉验证和训练测试分割来保留数据集中的类分布。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Use k-Fold Cross-Validation for Imbalanced Classification](img/fb9c25a00d0f16791cfe0d8fde7af8d2.png)

如何使用 k-Fold 交叉验证进行不平衡分类
图片由 [Bonnie Moreland](https://flickr.com/photos/icetsarina/36119331606/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  评估分类广告的挑战
2.  k 折叠交叉验证失败
3.  修复不平衡分类的交叉验证

## 评估分类广告的挑战

评估一个分类模型是具有挑战性的，因为在使用之前我们不会知道一个模型有多好。

相反，我们必须使用已经有目标或结果的可用数据来评估模型的表现。

模型评估不仅仅是评估一个模型；它包括测试不同的数据准备方案、不同的学习算法，以及针对表现良好的学习算法的不同超参数。

*   模型=数据准备+学习算法+超参数

理想情况下，可以选择和使用具有最佳分数(使用您选择的度量)的模型构建过程(数据准备、学习算法和超参数)。

最简单的模型评估过程是将数据集分成两部分，一部分用于训练模型，另一部分用于测试模型。因此，数据集的各个部分分别以它们的函数、训练集和测试集来命名。

如果您收集的数据集非常大并且代表问题，这将非常有效。所需示例的数量因问题而异，但可能需要数千、数十万或数百万个示例才足够。

训练和测试的 50/50 分割将是理想的，尽管更偏斜的分割是常见的，例如训练和测试集的 67/33 或 80/20。

我们很少有足够的数据来使用模型的训练/测试分割评估来获得表现的无偏估计。相反，我们的数据集通常比首选数据集小得多，因此必须在该数据集上使用重采样策略。

分类器最常用的模型评估方案是 10 重交叉验证程序。

[k 折叠交叉验证程序](https://machinelearningmastery.com/k-fold-cross-validation/)包括将训练数据集分割成 *k* 折叠。第一个 *k-1* 折叠用于训练模型，保持的第 *k* 折叠用作测试集。重复此过程，每个折叠都有机会用作保持测试集。对总共 *k* 个模型进行拟合和评估，并将模型的表现计算为这些运行的平均值。

事实证明，与单一训练/测试分割相比，该过程对小训练数据集上的模型表现给出了不太乐观的估计。 *k=10* 的值已被证明对各种数据集大小和模型类型都有效。

## k 折叠交叉验证失败

可悲的是，k-fold 交叉验证不适合评估不平衡的分类器。

> 10 倍交叉验证，特别是机器学习中最常用的误差估计方法，可以很容易地在类不平衡的情况下分解，即使偏斜没有之前考虑的那么极端。

—第 188 页，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

原因是数据被拆分成*k*-折叠，概率分布均匀。

这对于具有平衡的类分布的数据可能工作得很好，但当分布严重倾斜时，一个或多个折叠可能很少或没有来自少数类的示例。这意味着一些或可能许多模型评估将具有误导性，因为模型只需要正确预测多数类。

我们可以用一个例子来具体说明。

首先，我们可以定义一个少数与多数类分布比为 1:100 的数据集。

这可以通过使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建合成数据集来实现，指定示例数(1，000)、类数(2)和每个类的权重(99%和 1%)。

```py
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)
```

下面的示例生成了合成二进制类别数据集，并总结了类别分布。

```py
# create a binary classification dataset
from numpy import unique
from sklearn.datasets import make_classification
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)
# summarize dataset
classes = unique(y)
total = len(y)
for c in classes:
	n_examples = len(y[y==c])
	percent = n_examples / total * 100
	print('> Class=%d : %d/%d (%.1f%%)' % (c, n_examples, total, percent))
```

运行该示例会创建数据集，并总结每个类中的示例数量。

通过设置 *random_state* 参数，它确保我们每次运行代码时都能得到相同的随机生成的例子。

```py
> Class=0 : 990/1000 (99.0%)
> Class=1 : 10/1000 (1.0%)
```

总共 10 个例子在少数民族阶层并不多。如果我们使用 10 倍，在理想情况下，我们会在每个折叠中得到一个例子，这不足以训练一个模型。为了演示，我们将使用 5 倍。

在理想情况下，我们将在每个折叠中有 10/5 或两个示例，这意味着在训练数据集中有 4*2 (8)个折叠的示例，在给定的测试数据集中有 1*2 个折叠(2)。

首先，我们将使用 [KFold 类](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.KFold.html)将数据集随机拆分为 5 倍，并检查每个训练和测试集的组成。下面列出了完整的示例。

```py
# example of k-fold cross-validation with an imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
# enumerate the splits and summarize the distributions
for train_ix, test_ix in kfold.split(X):
	# select rows
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# summarize train and test composition
	train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
	test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
	print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
```

运行该示例会创建相同的数据集，并枚举数据的每个分割，显示训练集和测试集的类分布。

我们可以看到，在这种情况下，有一些分裂对训练集和测试集有预期的 8/2 分裂，还有一些分裂要差得多，比如 6/4(乐观)和 10/0(悲观)。

在这些数据分割的基础上评估一个模型不会给出一个可靠的表现估计。

```py
>Train: 0=791, 1=9, Test: 0=199, 1=1
>Train: 0=793, 1=7, Test: 0=197, 1=3
>Train: 0=794, 1=6, Test: 0=196, 1=4
>Train: 0=790, 1=10, Test: 0=200, 1=0
>Train: 0=792, 1=8, Test: 0=198, 1=2
```

如果我们使用简单的数据集训练/测试分割，我们可以证明存在类似的问题，尽管问题没有那么严重。

我们可以使用 [train_test_split()函数](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)来创建数据集的 50/50 分割，平均来说，如果我们多次执行这种分割，每个数据集中会出现少数民族类的五个示例。

下面列出了完整的示例。

```py
# example of train/test split with an imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)
# split into train/test sets with same class ratio
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# summarize
train_0, train_1 = len(trainy[trainy==0]), len(trainy[trainy==1])
test_0, test_1 = len(testy[testy==0]), len(testy[testy==1])
print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
```

运行该示例会创建与之前相同的数据集，并将其拆分为随机训练和测试拆分。

在这种情况下，我们可以看到训练集中只有三个少数民族类的例子，测试集中有七个。

以这种方式评估模型不会给他们提供足够多的可以学习的例子，太多的可以评估的例子，并且很可能会给出很差的表现。你可以想象，如果随机吐槽更严重，情况会变得更糟。

```py
>Train: 0=497, 1=3, Test: 0=493, 1=7
```

## 修复不平衡分类的交叉验证

解决方案是在使用 k 倍交叉验证或训练测试分割时，不要随机分割数据。

具体来说，我们可以随机分割数据集，尽管在每个子集中保持相同的类分布。这被称为**分层**或**分层采样**，目标变量( *y* )类用于控制采样过程。

例如，我们可以使用 k-fold 交叉验证的一个版本，它保留了每个 fold 中不平衡的类分布。它被称为**分层 K 折交叉验证**，并将在数据的每个分割中强制类别分布，以匹配完整训练数据集中的分布。

> ……通常情况下，特别是在阶级不平衡的情况下，使用分层的 10 倍交叉验证，这确保在所有折叠中尊重原始分布中发现的阳性与阴性样本的比例。

—第 205 页，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

我们可以用一个例子来具体说明。

我们可以使用[StrateFiedkfold 类](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)对拆分进行分层，正如其名称所示，该类支持分层的 k 倍交叉验证。

下面是相同的数据集和相同的交叉验证分层版本的示例。

```py
# example of stratified k-fold cross-validation with an imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# enumerate the splits and summarize the distributions
for train_ix, test_ix in kfold.split(X, y):
	# select rows
	train_X, test_X = X[train_ix], X[test_ix]
	train_y, test_y = y[train_ix], y[test_ix]
	# summarize train and test composition
	train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
	test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
	print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
```

运行该示例会像以前一样生成数据集，并总结每次拆分的训练集和测试集的类分布。

在这种情况下，我们可以看到每个分割都与我们在理想情况下的预期相匹配。

少数类中的每个示例都有一次机会在测试集中使用，并且每个数据分割的每个训练和测试集都有相同的类分布。

```py
>Train: 0=792, 1=8, Test: 0=198, 1=2
>Train: 0=792, 1=8, Test: 0=198, 1=2
>Train: 0=792, 1=8, Test: 0=198, 1=2
>Train: 0=792, 1=8, Test: 0=198, 1=2
>Train: 0=792, 1=8, Test: 0=198, 1=2
```

这个例子强调了需要首先为 *k* 选择一个值*k*-折叠交叉验证，以确保在训练和测试集中有足够数量的例子来拟合和评估模型(测试集中少数类的两个例子对于测试集来说可能太少)。

它还强调了使用分层 *k* 的要求——使用不平衡数据集进行多重交叉验证，以便为给定模型的每次评估保留训练和测试集中的类分布。

我们也可以使用分层版本的列车/测试分割。

这可以通过在调用 *train_test_split()* 时设置“*分层*参数，并将其设置为包含数据集目标变量的“ *y* ”变量来实现。由此，该函数将确定所需的类别分布，并确保列车和测试集都具有该分布。

我们可以用下面列出的一个工作示例来演示这一点。

```py
# example of stratified train/test split with an imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=1)
# split into train/test sets with same class ratio
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# summarize
train_0, train_1 = len(trainy[trainy==0]), len(trainy[trainy==1])
test_0, test_1 = len(testy[testy==0]), len(testy[testy==1])
print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
```

运行该示例会将数据集随机拆分为训练集和测试集，从而确保类分布得到保留，在这种情况下，每个数据集中会留下五个示例。

```py
>Train: 0=495, 1=5, Test: 0=495, 1=5
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [k 倍交叉验证的温和介绍](https://machinelearningmastery.com/k-fold-cross-validation/)

### 书

*   [不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

### 应用程序接口

*   [sklearn.model_selection。KFold API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.KFold.html) 。
*   [sklearn.model_selection。stratifedkfold API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)。
*   [sklearn . model _ selection . train _ test _ split API](https://Sklearn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)。

## 摘要

在本教程中，您发现了如何在不平衡的数据集上评估分类器模型。

具体来说，您了解到:

*   使用训练/测试分割和交叉验证在数据集上评估分类器的挑战。
*   当在不平衡的数据集上评估分类器时，k-fold 交叉验证和训练-测试分割的简单应用将如何失败。
*   如何使用修改后的 k-fold 交叉验证和训练测试分割来保留数据集中的类分布。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。