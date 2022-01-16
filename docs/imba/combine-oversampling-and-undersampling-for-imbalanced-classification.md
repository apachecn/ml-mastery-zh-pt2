# 如何为不平衡分类结合过采样和欠采样

> 原文：<https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/>

最后更新于 2021 年 5 月 11 日

重采样方法旨在从训练数据集中添加或移除示例，以改变类别分布。

一旦类别分布更加平衡，标准机器学习分类算法套件就可以成功地适用于转换后的数据集。

过采样方法复制或创建少数类中的新合成示例，而欠采样方法删除或合并多数类中的示例。两种类型的重采样在单独使用时都是有效的，尽管当两种类型的方法一起使用时会更有效。

在本教程中，您将发现如何将过采样和欠采样技术结合起来用于不平衡分类。

完成本教程后，您将知道:

*   如何定义要应用于训练数据集或评估分类器模型时的过采样和欠采样方法序列。
*   如何手动结合过采样和欠采样方法进行不平衡分类？
*   如何使用预定义且表现良好的重采样方法组合进行不平衡分类。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2021 年 1 月更新**:更新了 API 文档的链接。

![Combine Oversampling and Undersampling for Imbalanced Classification](img/513edcad911311b2b3e1da1bf8fa5719.png)

结合过采样和欠采样进行不平衡分类
图片由[拉德克·库恰尔斯基](https://flickr.com/photos/137294100@N08/43934817620/)提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  二元测试问题与决策树模型
2.  不平衡学习库
3.  手动组合过采样和欠采样方法
    1.  手动组合随机过采样和欠采样
    2.  手动组合 SMOTE 和随机欠采样
4.  使用预定义的重采样方法组合
    1.  SMOTE 和 Tomek 链接的组合欠采样
    2.  SMOTE 和编辑最近邻欠采样的组合

## 二元测试问题与决策树模型

在我们深入研究过采样和欠采样方法的组合之前，让我们定义一个合成数据集和模型。

我们可以使用 Sklearn 库中的 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)定义一个合成的二进制类别数据集。

例如，我们可以用两个输入变量和 1:100 的类分布创建 10，000 个示例，如下所示:

```py
...
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
```

然后，我们可以通过[散点图()Matplotlib 函数](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html)创建数据集的散点图，以了解每个类中示例的空间关系及其不平衡。

```py
...
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

将这些联系在一起，下面列出了创建不平衡类别数据集并绘制示例的完整示例。

```py
# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
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

运行该示例首先总结了类分布，显示了大约 1:100 的类分布，其中大约 10，000 个示例包含类 0，100 个示例包含类 1。

```py
Counter({0: 9900, 1: 100})
```

接下来，创建散点图，显示数据集中的所有示例。我们可以看到大量 0 类(蓝色)的例子和少量 1 类(橙色)的例子。

我们还可以看到，在属于类 0 的特征空间部分中，类与类 1 中的一些示例明显重叠。

![Scatter Plot of Imbalanced Classification Dataset](img/b45b9061e6070c500986c4ce8517e3c4.png)

不平衡类别数据集的散点图

我们可以在这个数据集上拟合一个[决策树分类器模型](https://Sklearn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)。这是一个很好的测试模型，因为它对训练数据集中的类分布很敏感。

```py
...
# define model
model = DecisionTreeClassifier()
```

我们可以使用[重复的分层 k 折叠交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)来评估模型，重复 3 次，重复 10 次。

[曲线下 ROC 面积(AUC)度量](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)可以用来估计模型的表现。对于严重不平衡的数据集，它可能是乐观的，尽管它确实正确地显示了模型表现的相对提高。

```py
...
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

将这些联系在一起，下面的例子评估了不平衡类别数据集上的决策树模型。

```py
# evaluates a decision tree model on the imbalanced dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# generate 2 class dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# define model
model = DecisionTreeClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例会报告数据集上决策树在三次重复的 10 倍交叉验证中的平均 ROC AUC(例如，30 次不同模型评估的平均值)。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这个例子中，你可以看到模型实现了大约 0.76 的 ROC AUC。这为这个数据集提供了一个基线，我们可以用它来比较训练数据集中过采样和欠采样方法的不同组合。

```py
Mean ROC AUC: 0.762
```

现在我们有了测试问题、模型和测试工具，让我们看看过采样和欠采样方法的手动组合。

## 不平衡学习库

在这些例子中，我们将使用[不平衡学习 Python 库](https://github.com/Sklearn-contrib/imbalanced-learn)提供的实现，可以通过 pip 安装如下:

```py
sudo pip install imbalanced-learn
```

您可以通过打印已安装库的版本来确认安装成功:

```py
# check version number
import imblearn
print(imblearn.__version__)
```

运行该示例将打印已安装库的版本号；例如:

```py
0.5.0
```

## 手动组合过采样和欠采样方法

不平衡学习 Python 库提供了一系列重采样技术，以及一个管道类，可用于创建应用于数据集的重采样方法的组合序列。

我们可以使用[管道](https://imbalanced-learn.org/stable/generated/imblearn.pipeline.Pipeline.html)构建一系列过采样和欠采样技术来应用于数据集。例如:

```py
# define resampling
over = ...
under = ...
# define pipeline
pipeline = Pipeline(steps=[('o', over), ('u', under)])
```

该流水线首先对数据集应用过采样技术，然后在返回最终结果之前对过采样变换的输出应用欠采样。它允许在数据集上按顺序堆叠或应用变换。

然后可以使用管道来转换数据集；例如:

```py
# fit and apply the pipeline
X_resampled, y_resampled = pipeline.fit_resample(X, y)
```

或者，模型可以作为管道的最后一步添加。

这允许管道被视为一个模型。当在训练数据集上拟合时，首先将变换应用于训练数据集，然后将变换后的数据集提供给模型，以便它可以进行拟合。

```py
...
# define model
model = ...
# define resampling
over = ...
under = ...
# define pipeline
pipeline = Pipeline(steps=[('o', over), ('u', under), ('m', model)])
```

回想一下，重采样仅应用于训练数据集，而不是测试数据集。

当在 [k 折叠交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)中使用时，整个变换和拟合序列应用于由交叉验证折叠组成的每个训练数据集。这很重要，因为变换和拟合都是在不知道保持集的情况下执行的，这避免了[数据泄露](https://machinelearningmastery.com/data-leakage-machine-learning/)。例如:

```py
...
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
```

现在我们知道了如何手动组合重采样方法，让我们看两个例子。

### 手动组合随机过采样和欠采样

组合重采样技术的一个很好的起点是从随机或简单的方法开始。

虽然它们很简单，而且单独应用时通常无效，但结合起来就可以有效。

随机过采样包括随机复制少数类中的示例，而随机欠采样包括从多数类中随机删除示例。

由于这两个转换是在不同的类上执行的，因此它们应用于训练数据集的顺序并不重要。

下面的示例定义了一个管道，该管道首先将少数类过采样到多数类的 10%，将多数类采样到比少数类多 50%，然后拟合决策树模型。

```py
...
# define model
model = DecisionTreeClassifier()
# define resampling
over = RandomOverSampler(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
# define pipeline
pipeline = Pipeline(steps=[('o', over), ('u', under), ('m', model)])
```

下面列出了在二分类问题上评估这种组合的完整示例。

```py
# combination of random oversampling and undersampling for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# define model
model = DecisionTreeClassifier()
# define resampling
over = RandomOverSampler(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
# define pipeline
pipeline = Pipeline(steps=[('o', over), ('u', under), ('m', model)])
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估转换系统和模型，并将表现总结为平均 ROC AUC。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到 ROC AUC 表现从无变换的 0.76 适度提升到随机过采样和欠采样的约 0.81。

```py
Mean ROC AUC: 0.814
```

### 手动组合 SMOTE 和随机欠采样

我们不限于使用随机重采样方法。

也许最流行的过采样方法是合成少数过采样技术，简称 SMOTE。

SMOTE 的工作方式是选择特征空间中靠近的示例，在特征空间中的示例之间绘制一条线，并沿着该线绘制一个新样本作为点。

该技术的作者建议在少数类上使用 SMOTE，然后在多数类上使用欠采样技术。

> SMOTE 和欠采样的组合比普通欠采样表现更好。

——[SMOTE:合成少数民族过采样技术](https://arxiv.org/abs/1106.1813)，2011 年。

我们可以将 SMOTE 与[随机欠采样](https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)结合起来。同样，这些过程的应用顺序并不重要，因为它们是在训练数据集的不同子集上执行的。

下面的管道实现了这种组合，首先应用 SMOTE 使少数类分布达到多数类的 10%，然后使用*随机欠采样*使多数类比少数类多 50%，然后拟合[决策树分类器](https://Sklearn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)。

```py
...
# define model
model = DecisionTreeClassifier()
# define pipeline
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under), ('m', model)]
```

下面的例子在我们的不平衡二进制分类问题上评估了这个组合。

```py
# combination of SMOTE and random undersampling for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# define model
model = DecisionTreeClassifier()
# define pipeline
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under), ('m', model)]
pipeline = Pipeline(steps=steps)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估转换系统和模型，并将表现总结为平均 ROC AUC。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到 ROC AUC 表现从大约 0.81 到大约 0.83 的另一个列表。

```py
Mean ROC AUC: 0.833
```

## 使用预定义的重采样方法组合

已经证明过采样和欠采样方法的组合是有效的，并且可以一起被认为是重采样技术。

两个例子是 SMOTE 与 Tomek 链接欠采样和 SMOTE 与编辑最近邻欠采样的组合。

不平衡学习 Python 库直接为这两种组合提供了实现。让我们依次仔细看看每一个。

### SMOTE 和 Tomek 链接的组合欠采样

SMOTE 是一种过采样方法，在少数民族类别中合成新的似是而非的例子。

Tomek Links 指的是一种用于识别数据集中具有不同类的最近邻对的方法。移除这些对中的一个或两个示例(例如多数类中的示例)具有使训练数据集中的决策边界不那么嘈杂或模糊的效果。

Gustavo Batista 等人在他们 2003 年发表的题为“平衡训练数据以自动标注关键词:案例研究”的论文中测试了组合这些方法

具体来说，首先应用 SMOTE 方法对少数类进行过采样以获得平衡分布，然后识别并移除多数类的 Tomek Links 中的示例。

> 在这项工作中，只有参与 Tomek 链接的多数类示例被删除，因为少数类示例被认为太罕见而不能被丢弃。[……]在我们的工作中，由于少数类示例是人为创建的，并且数据集目前是平衡的，因此形成 Tomek 链接的多数类和少数类示例都被删除了。

——[平衡训练数据自动标注关键词:案例研究](http://www.inf.ufrgs.br/maslab/pergamus/pubs/balancing-training-data-for.pdf)，2003。

这种组合被证明可以减少假阴性，但代价是增加了二进制分类任务的假阳性。

我们可以使用[smottomek 类](https://imbalanced-learn.org/stable/generated/imblearn.combine.SMOTETomek.html)来实现这个组合。

```py
...
# define resampling
resample = SMOTETomek()
```

SMOTE 配置可以通过“ *smote* ”参数设置，并采用已配置的 [SMOTE](https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.SMOTE.html) 实例。可以通过“tomek”参数设置 Tomek 链接配置，并采用已配置的 [TomekLinks](https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.TomekLinks.html) 对象。

默认情况下，使用 SMOTE 平衡数据集，然后从所有类中移除 Tomek 链接。这是另一篇探索这种组合的论文中使用的方法，标题为“几种平衡机器学习训练数据的方法的行为研究”

> ……我们建议将 Tomek 链接应用于过采样的训练集，作为一种数据清理方法。因此，不是只移除形成 Tomek 链接的大多数类示例，而是移除两个类中的示例。

——[平衡机器学习训练数据的几种方法的行为研究](https://dl.acm.org/citation.cfm?id=1007735)，2004。

或者，我们可以将组合配置为仅从多数类中移除链接，如 2003 年的论文中所述，方法是用实例*指定“ *tomek* ”参数，将“ *sampling_strategy* ”参数设置为仅欠采样“*多数*类；例如:*

```py
...
# define resampling
resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
```

在我们的二进制分类问题上，我们可以用决策树分类器来评估这种组合重采样策略。

下面列出了完整的示例。

```py
# combined SMOTE and Tomek Links resampling for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# define model
model = DecisionTreeClassifier()
# define resampling
resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
# define pipeline
pipeline = Pipeline(steps=[('r', resample), ('m', model)])
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估转换系统和模型，并将表现总结为平均 ROC AUC。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，这种组合的重采样策略似乎没有为该数据集上的模型提供好处。

```py
Mean ROC AUC: 0.815
```

### SMOTE 和编辑最近邻欠采样的组合

SMOTE 可能是最流行的过采样技术，可以与许多不同的欠采样技术相结合。

另一种非常流行的欠采样方法是编辑最近邻法，或 ENN 法则。该规则包括使用 *k=3* 最近邻来定位数据集中被错误分类然后被移除的那些示例。它可以应用于所有的类，也可以只应用于大多数类中的那些例子。

Gustavo Batista 等人在他们 2004 年的论文《几种平衡机器学习训练数据的方法的行为研究》中，探索了过采样和欠采样方法的许多组合，而不是孤立使用的方法

这包括以下组合:

*   压缩最近邻+托梅克链接
*   SMOTE + Tomek 左侧
*   最近邻居

关于这最后的组合，作者评论说，ENN 在降低多数类的采样上比托梅克链接更积极，提供了更深入的清理。他们应用这种方法，从多数和少数群体中删除例子。

> ……ENN 被用来从这两个类中移除示例。因此，任何被其三个最近邻居错误分类的例子都将从训练集中移除。

——[平衡机器学习训练数据的几种方法的行为研究](https://dl.acm.org/citation.cfm?id=1007735)，2004。

这可以通过不平衡学习库中的[SMOTENN 类](https://imbalanced-learn.org/stable/generated/imblearn.combine.SMOTEENN.html)来实现。

```py
...
# define resampling
resample = SMOTEENN()
```

SMOTE 配置可以通过“ *smote* 参数设置为 SMOTE 对象，ENN 配置可以通过“ *enn* 参数设置为[编辑最近邻居](https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.EditedNearestNeighbours.html)对象。SMOTE 默认平衡分布，其次是 ENN，默认从所有类中移除错误分类的示例。

我们可以通过将“ *enn* ”参数设置为一个*edited nearest neighbores*实例，并将 *sampling_strategy* 参数设置为“*more*”来更改 ENN，只从多数类中移除示例。

```py
...
# define resampling
resample = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
```

我们可以评估默认策略(编辑所有类中的示例)，并在不平衡的数据集上使用决策树分类器进行评估。

下面列出了完整的示例。

```py
# combined SMOTE and Edited Nearest Neighbors resampling for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# define model
model = DecisionTreeClassifier()
# define resampling
resample = SMOTEENN()
# define pipeline
pipeline = Pipeline(steps=[('r', resample), ('m', model)])
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估转换系统和模型，并将表现总结为平均 ROC AUC。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们看到随机欠采样方法的表现比 SMOTE 进一步提升，从约 0.81 提升到约 0.85。

```py
Mean ROC AUC: 0.856
```

这一结果突出表明，编辑过采样的少数族裔也可能是一个很容易被忽略的重要考虑因素。

这与 2004 年论文中的发现相同，作者发现 SMOTE 与 Tomek Links 和 SMOTE 与 ENN 在一系列数据集上表现良好。

> 我们的结果表明，对于几乎没有正(少数)例子的数据集，一般的过采样方法，特别是 Smote + Tomek 和 Smote + ENN(本工作中提出的两种方法)在实践中提供了非常好的结果。

——[平衡机器学习训练数据的几种方法的行为研究](https://dl.acm.org/citation.cfm?id=1007735)，2004。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [SMOTE:合成少数过采样技术](https://arxiv.org/abs/1106.1813)，2011。
*   [平衡训练数据进行关键词自动标注:案例研究](http://www.inf.ufrgs.br/maslab/pergamus/pubs/balancing-training-data-for.pdf)，2003。
*   [平衡机器学习训练数据的几种方法的行为研究](https://dl.acm.org/citation.cfm?id=1007735)，2004。

### 书

*   [从不平衡数据集中学习](https://amzn.to/307Xlva)，2018。
*   [不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

### 应用程序接口

*   [不平衡学习，GitHub](https://github.com/Sklearn-contrib/imbalanced-learn) 。
*   [过采样和欠采样相结合，不平衡学习用户指南](https://imbalanced-learn.org/stable/combine.html)。
*   [imblearn.over_sampling。randomoversacompler API](https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.RandomOverSampler.html)。
*   [imb learn . pipeline . pipeline API](https://imbalanced-learn.org/stable/generated/imblearn.pipeline.Pipeline.html)。
*   [imblearn.under_sampling。随机欠采样应用编程接口](https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)。
*   [imblearn.over_sampling。SMOTE API](https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.SMOTE.html) 。
*   [imb learn . combine . smotetomek API](https://imbalanced-learn.org/stable/generated/imblearn.combine.SMOTETomek.html)。
*   [imb learn . combine . SMOTENN API](https://imbalanced-learn.org/stable/generated/imblearn.combine.SMOTEENN.html)。

### 文章

*   [数据分析中的过采样和欠采样，维基百科](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)。

## 摘要

在本教程中，您发现了如何将过采样和欠采样技术结合起来用于不平衡分类。

具体来说，您了解到:

*   如何定义要应用于训练数据集或评估分类器模型时的过采样和欠采样方法序列。
*   如何手动结合过采样和欠采样方法进行不平衡分类？
*   如何使用预定义且表现良好的重采样方法组合进行不平衡分类。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。