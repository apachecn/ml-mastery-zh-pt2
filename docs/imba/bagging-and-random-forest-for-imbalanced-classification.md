# 用于不平衡分类的装袋和随机森林

> 原文：<https://machinelearningmastery.com/bagging-and-random-forest-for-imbalanced-classification/>

最后更新于 2021 年 1 月 5 日

Bagging 是一种集成算法，它在训练数据集的不同子集上拟合多个模型，然后组合来自所有模型的预测。

[随机森林](https://machinelearningmastery.com/random-forest-ensemble-in-python/)是 bagging 的扩展，它也随机选择每个数据样本中使用的特征子集。装袋和随机森林都已被证明对各种不同的预测建模问题有效。

虽然有效，但它们不适合具有倾斜类分布的分类问题。然而，已经提出了对算法的许多修改，以适应它们的行为，并使它们更好地适应严重的类不平衡。

在本教程中，您将发现如何使用装袋和随机森林进行不平衡分类。

完成本教程后，您将知道:

*   如何使用随机欠采样 Bagging 进行不平衡分类？
*   如何使用带有类权重和随机欠采样的随机森林进行不平衡分类。
*   如何使用装袋和增强相结合的简单集成进行不平衡分类。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2021 年 1 月更新**:更新了 API 文档的链接。

![Bagging and Random Forest for Imbalanced Classification](img/5f868402e2a7637550e44bfe53610ea3.png)

不平衡分类的装袋和随机森林
唐·格雷厄姆摄，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  不平衡分类的装袋
    1.  标准装袋
    2.  随机欠采样装袋
2.  不平衡分类的随机森林
    1.  标准随机森林
    2.  具有类权重的随机森林
    3.  具有自举类加权的随机森林
    4.  随机欠采样的随机森林
3.  不平衡分类的简单集成
    1.  轻松地在一起

## 不平衡分类的装袋

[Bootstrap Aggregation，简称 Bagging】，是一种集成机器学习算法。](https://machinelearningmastery.com/implement-bagging-scratch-python/)

它包括首先选择带有替换的训练数据集的随机样本，这意味着给定的样本可能在训练数据集中包含零个、一个或多个样本副本。这被称为引导样本。然后在每个数据样本上拟合一个弱学习模型。典型地，不使用修剪的决策树模型(例如，可能稍微过度训练它们的训练集)被用作弱学习器。最后，来自所有适合的弱学习器的预测被组合以做出单个预测(例如，聚集的)。

> 集合中的每个模型随后被用于生成新样本的预测，并且这些 m 个预测被平均以给出袋装模型的预测。

—第 192 页，[应用预测建模](https://amzn.to/2W8wnPS)，2013 年。

创建新的引导样本以及对样本进行拟合和添加树的过程可以继续，直到在验证数据集上看不到集成表现的进一步提高。

这个简单的过程通常比单一配置良好的决策树算法产生更好的表现。

按原样装袋将创建[引导样本](https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/)，该样本将不考虑不平衡类别数据集的倾斜类分布。因此，尽管该技术总体上表现良好，但如果存在严重的类别不平衡，它可能表现不佳。

### 标准装袋

在我们深入探索 bagging 的扩展之前，让我们评估一个没有 bagging 的标准 bagging 决策树集成，并将其用作比较点。

我们可以使用[装袋分类器 scikit-sklearn 类](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)创建一个配置大致相同的装袋决策树模型。

首先，让我们用 10，000 个例子来定义一个合成的不平衡二进制分类问题，其中 99%属于多数类，1%属于少数类。

```py
...
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
```

然后，我们可以定义标准的袋装决策树集成模型，以备评估。

```py
...
# define model
model = BaggingClassifier()
```

然后，我们可以使用重复分层 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)来评估该模型，重复 3 次，重复 10 次。

我们将使用所有折叠和重复的平均值 [ROC AUC 评分](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)来评估模型的表现。

```py
...
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
```

将这些联系在一起，下面列出了在不平衡类别数据集上评估标准袋装集成的完整示例。

```py
# bagged decision trees on an imbalanced classification problem
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = BaggingClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估模型并报告平均 ROC AUC 分数。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型获得了大约 0.87 的分数。

```py
Mean ROC AUC: 0.871
```

### 随机欠采样装袋

有许多方法可以使装袋适用于不平衡分类。

也许最直接的方法是在拟合弱学习器模型之前对自举样本应用数据重采样。这可能涉及对少数类进行过采样或对多数类进行欠采样。

> 在装袋过程中面临重采样阶段时，克服类不平衡问题的一个简单方法是在从原始数据集中随机抽取实例时考虑它们的类。

—第 175 页，[从不平衡数据集](https://amzn.to/307Xlva)中学习，2018。

对引导中的少数类进行过采样被称为过采样；同样，对引导中的多数类进行欠采样被称为欠采样，而将两种方法结合起来被称为过欠采样。

不平衡学习库提供了欠标记的实现。

具体来说，它提供了 bagging 的一个版本，在引导样本内的多数类上使用随机欠采样策略，以便平衡这两个类。这在[平衡分类器类](https://imbalanced-learn.org/stable/generated/imblearn.ensemble.BalancedBaggingClassifier.html)中提供。

```py
...
# define model
model = BalancedBaggingClassifier()
```

接下来，我们可以评估袋装决策树集成的修改版本，该版本在拟合每个决策树之前对多数类执行随机欠采样。

我们期望随机欠采样的使用将改善整体的表现。

该模型和前一模型的默认树数(*n _ estimates*)为 10。实际上，最好测试这个超参数的较大值，例如 100 或 1000。

下面列出了完整的示例。

```py
# bagged decision trees with random undersampling for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import BalancedBaggingClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = BalancedBaggingClassifier()
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估模型并报告平均 ROC AUC 分数。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到平均 ROC AUC 从没有任何数据重采样的约 0.87 提升到大多数类随机欠采样的约 0.96。

这不是一个真正的苹果对苹果的比较，因为我们使用的是来自两个不同库的相同算法实现，但是它表明了一个普遍的观点，即在适应弱学习器之前平衡引导提供了一些好处，当类分布是倾斜的。

```py
Mean ROC AUC: 0.962
```

虽然*BalancedBaggingClassifier*类使用了决策树，但是你可以测试不同的模型，比如 k 近邻等等。您可以在定义类时设置 *base_estimator* 参数，以使用不同的弱学习器分类器模型。

## 不平衡分类的随机森林

[随机森林](https://machinelearningmastery.com/implement-random-forest-scratch-python/)是决策树模型的另一个集合，可以认为是装袋后的改进。

像装袋一样，随机森林包括从训练数据集中选择自举样本，并在每个样本上拟合决策树。主要区别是没有使用所有的特性(变量或列)；取而代之的是，为每个引导样本选择一个小的、随机选择的特征(列)子集。这具有去相关决策树(使它们更独立)的效果，并且反过来改善集成预测。

> 集合中的每个模型随后被用于生成新样本的预测，并且这些 m 个预测被平均以给出森林的预测。由于算法在每次分割时随机选择预测因子，因此树的相关性必然会降低。

—第 199 页，[应用预测建模](https://amzn.to/2W8wnPS)，2013 年。

同样，随机森林在广泛的问题上非常有效，但是像 bagging 一样，标准算法在不平衡分类问题上的表现不是很好。

> 在学习极不平衡的数据时，很有可能一个自举样本包含很少或甚至没有少数类，导致预测少数类的树表现很差。

——[利用随机森林学习不平衡数据](https://statistics.berkeley.edu/tech-reports/666)，2004。

### 标准随机森林

在我们深入研究随机森林集成算法的扩展以使其更适合不平衡分类之前，让我们在合成数据集上拟合和评估随机森林算法。

我们可以使用 scikit 中的 [RandomForestClassifier](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) 类——学习并使用少量的树，在本例中为 10 棵树。

```py
...
# define model
model = RandomForestClassifier(n_estimators=10)
```

下面列出了在不平衡数据集上拟合标准随机森林集合的完整示例。

```py
# random forest for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = RandomForestClassifier(n_estimators=10)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估模型并报告平均 ROC AUC 分数。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型实现了大约 0.86 的平均 ROC AUC。

```py
Mean ROC AUC: 0.869
```

### 具有类权重的随机森林

修改不平衡分类决策树的一个简单技术是，在计算所选分割点的“*杂质*”分数时，改变每个类别的权重。

杂质测量训练数据集中给定分割的样本组的混合程度，通常用基尼系数或[熵](https://machinelearningmastery.com/what-is-information-entropy/)来测量。计算可能会有偏差，因此有利于少数群体的混合是有利的，允许多数群体有一些假阳性。

随机森林的这种修改被称为加权随机森林。

> 另一种使随机森林更适合从极度不平衡的数据中学习的方法遵循成本敏感学习的思想。由于射频分类器倾向于偏向多数类，我们将对少数类的错误分类施加更重的惩罚。

——[利用随机森林学习不平衡数据](https://statistics.berkeley.edu/tech-reports/666)，2004。

这可以通过在*随机森林分类器*类上设置*类权重*参数来实现。

此参数使用一个字典，其中包含每个类值(例如 0 和 1)到权重的映射。可以提供“ *balanced* ”的自变量值，以自动使用来自训练数据集的反向加权，从而将焦点放在少数类上。

```py
...
# define model
model = RandomForestClassifier(n_estimators=10, class_weight='balanced')
```

我们可以在我们的测试问题上测试随机森林的这个修改。虽然不是随机森林特有的，但我们希望有一些适度的改进。

下面列出了完整的示例。

```py
# class balanced random forest for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = RandomForestClassifier(n_estimators=10, class_weight='balanced')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估模型并报告平均 ROC AUC 分数。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型实现了平均 ROC AUC 从 0.86 到约 0.87 的适度提升。

```py
Mean ROC AUC: 0.871
```

### 具有自举类加权的随机森林

假设每个决策树是从引导样本构建的(例如，带有替换的随机选择)，数据样本中的类分布对于每个树将是不同的。

因此，基于每个引导样本中的类分布而不是整个训练数据集来改变类权重可能是有趣的。

这可以通过将 *class_weight* 参数设置为值“ *balanced_subsample* 来实现。

我们可以测试这种修改，并将结果与上面的“平衡”情况进行比较；下面列出了完整的示例。

```py
# bootstrap class balanced random forest for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = RandomForestClassifier(n_estimators=10, class_weight='balanced_subsample')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估模型并报告平均 ROC AUC 分数。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型实现了平均 ROC AUC 从 0.87 到约 0.88 的适度提升。

```py
Mean ROC AUC: 0.884
```

### 随机欠采样的随机森林

对随机森林的另一个有用的修改是对引导样本执行数据重采样，以便显式地改变类分布。

不平衡学习库中的[平衡随机森林分类器类](https://imbalanced-learn.org/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html)实现了这一点，并对 reach bootstrap 样本中的大多数类执行随机欠采样。这通常被称为平衡随机森林。

```py
...
# define model
model = BalancedRandomForestClassifier(n_estimators=10)
```

鉴于数据重采样技术的广泛成功，我们预计这将对模型表现产生更显著的影响。

我们可以在合成数据集上测试随机森林的这种修改，并比较结果。下面列出了完整的示例。

```py
# random forest with random undersampling for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = BalancedRandomForestClassifier(n_estimators=10)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估模型并报告平均 ROC AUC 分数。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到模型实现了平均 ROC AUC 从 0.89 到约 0.97 的适度提升。

```py
Mean ROC AUC: 0.970
```

## 不平衡分类的简单集成

当考虑将袋装集成用于不平衡分类时，自然的想法可能是使用多数类的随机重采样来创建具有平衡类分布的多个数据集。

具体来说，可以从少数类中的所有示例和从多数类中随机选择的样本创建数据集。然后模型或弱学习器可以适合这个数据集。该过程可以重复多次，并且可以使用整个模型集合的平均预测来进行预测。

这正是徐等人在 2008 年的论文《类不平衡学习的探索性欠采样》中提出的方法

子样本的选择性构造被视为多数类的一种欠采样。多个子样本的生成允许集合克服欠采样的缺点，其中有价值的信息从训练过程中被丢弃。

> ……欠采样是处理阶级不平衡的有效策略。然而，欠采样的缺点是它丢弃了许多潜在有用的数据。

——[班级不平衡学习的探索性欠采样](https://ieeexplore.ieee.org/document/4717268)，2008。

作者提出了该方法的变体，如简单集成和平衡级联。

让我们仔细看看轻松集成。

### 轻松地在一起

简单集成包括通过从少数类中选择所有示例，从多数类中选择一个子集，来创建训练数据集的平衡样本。

不是使用修剪的决策树，而是在每个子集上使用增强的决策树，特别是 AdaBoost 算法。

[AdaBoost](https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/) 的工作原理是首先在数据集上拟合一棵决策树，然后确定该树产生的错误，并根据这些错误对数据集中的示例进行加权，以便更多地关注错误分类的示例，而较少关注正确分类的示例。然后，在加权数据集上拟合后续树，以纠正错误。然后对给定数量的决策树重复该过程。

> 这意味着难以分类的样本将获得越来越大的权重，直到算法识别出能够正确分类这些样本的模型。因此，算法的每次迭代都需要学习数据的不同方面，重点是包含难以分类样本的区域。

—第 389 页，[应用预测建模](https://amzn.to/2W8wnPS)，2013 年。

来自不平衡学习库的[简易集成分类器类](https://imbalanced-learn.org/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html)提供了简易集成技术的实现。

```py
...
# define model
model = EasyEnsembleClassifier(n_estimators=10)
```

我们可以评估我们的综合不平衡分类问题的技术。

给定一种随机欠采样的使用，我们期望该技术总体上表现良好。

下面列出了完整的示例。

```py
# easy ensemble for imbalanced classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import EasyEnsembleClassifier
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# define model
model = EasyEnsembleClassifier(n_estimators=10)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
```

运行该示例评估模型并报告平均 ROC AUC 分数。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

在这种情况下，我们可以看到集成在数据集上表现良好，实现了约 0.96 的平均 ROC AUC，接近于在具有随机欠采样的随机森林的数据集上实现的平均 ROC AUC(0.97)。

```py
Mean ROC AUC: 0.968
```

虽然在每个子样本上使用了一个 AdaBoost 分类器，但是可以通过将 *base_estimator* 参数设置到模型来使用替代分类器模型。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 报纸

*   [利用随机森林学习不平衡数据](https://statistics.berkeley.edu/tech-reports/666)，2004。
*   [班级不平衡学习的探索性欠采样](https://ieeexplore.ieee.org/document/4717268)，2008。

### 书

*   [从不平衡数据集中学习](https://amzn.to/307Xlva)，2018。
*   [不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。
*   [应用预测建模](https://amzn.to/2W8wnPS)，2013。

### 蜜蜂

*   [im learn .一起。平衡负载分类器 API](https://imbalanced-learn.org/stable/generated/imblearn.ensemble.BalancedBaggingClassifier.html) 。
*   [硬化。一起。bagginclassifier API](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)。
*   [硬化。一起。随机应变分类 API](https://Sklearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) 。
*   [im learn . together . balance dradomforestclass ification API](https://imbalanced-learn.org/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html)。

## 摘要

在本教程中，您发现了如何使用装袋和随机森林进行不平衡分类。

具体来说，您了解到:

*   如何使用随机欠采样 Bagging 进行不平衡分类？
*   如何使用带有类权重和随机欠采样的随机森林进行不平衡分类。
*   如何使用装袋和增强相结合的简单集成进行不平衡分类。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。