# 用于不平衡分类的随机过采样和欠采样

> 原文：<https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/>

最后更新于 2021 年 1 月 5 日

不平衡数据集是指在类分布中存在严重偏差的数据集，例如少数类与多数类的比例为 1:100 或 1:1000。

训练数据集中的这种偏差会影响许多机器学习算法，导致一些算法完全忽略少数类。这是一个问题，因为预测最重要的通常是少数群体。

解决类不平衡问题的一种方法是对训练数据集进行随机重采样。随机对不平衡数据集进行重采样的两种主要方法是从多数类中删除示例，称为欠采样，以及从少数类中复制示例，称为过采样。

在本教程中，您将发现用于不平衡分类的随机过采样和欠采样

完成本教程后，您将知道:

*   随机重采样为不平衡数据集的类分布重新平衡提供了一种简单的技术。
*   随机过采样会复制训练数据集中少数类的示例，并可能导致某些模型的过拟合。
*   随机欠采样会从多数类中删除示例，并可能导致丢失对模型来说非常宝贵的信息。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2021 年 1 月更新**:更新了 API 文档的链接。

![Random Oversampling and Undersampling for Imbalanced Classification](img/0bf379f3b222c99133de9ed5254a5f52.png)

不平衡分类的随机过采样和欠采样
图片由 [RichardBH](https://flickr.com/photos/rbh/9783814043/) 提供，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  随机重采样不平衡数据集
2.  不平衡学习库
3.  随机过采样不平衡数据集
4.  随机欠采样不平衡数据集
5.  结合随机过采样和欠采样

## 随机重采样不平衡数据集

重采样包括创建训练数据集的新变换版本，其中所选示例具有不同的类分布。

这是一种简单有效的解决不平衡分类问题的策略。

> 应用重采样策略来获得更平衡的数据分布是解决不平衡问题的有效方法

——[不平衡分布下的预测建模综述](https://arxiv.org/abs/1505.01658)，2015 年。

最简单的策略是为转换后的数据集随机选择示例，称为随机重采样。

不平衡分类的随机重采样主要有两种方法；它们是过采样和欠采样。

*   **随机过采样**:随机复制小众类的例子。
*   **随机欠采样**:随机删除多数类中的例子。

随机过采样包括从少数类中随机选择示例，进行替换，并将它们添加到训练数据集中。随机欠采样包括从多数类中随机选择示例，并将其从训练数据集中删除。

> 在随机欠采样中，大多数类实例被随机丢弃，直到达到更平衡的分布。

—第 45 页，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013

这两种方法都可以重复，直到在训练数据集中达到期望的类分布，例如类之间的相等分割。

它们被称为“原始重采样”方法，因为它们对数据不做任何假设，也不使用试探法。这使得它们实现简单，执行快速，这对于非常大和复杂的数据集来说是理想的。

这两种技术都可以用于两类(二元)分类问题和具有一个或多个多数或少数类的多类分类问题。

重要的是，对类分布的更改仅应用于训练数据集。目的是影响模型的拟合。重采样不会应用于用于评估模型表现的测试或保持数据集。

一般来说，这些简单的方法可能是有效的，尽管这取决于所涉及的数据集和模型的细节。

让我们仔细看看每种方法以及如何在实践中使用它们。

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

## 随机过采样不平衡数据集

随机过采样包括随机复制少数民族的例子，并将其添加到训练数据集中。

训练数据集中的示例是随机选择的，并进行替换。这意味着可以多次选择来自少数民族类的例子，并将其添加到新的更加平衡的*训练数据集；它们是从原始训练数据集中选择的，添加到新的训练数据集中，然后返回或*替换原始数据集中的*，允许再次选择它们。*

 *这种技术对于那些受偏斜分布影响的机器学习算法是有效的，其中给定类的多个重复示例会影响模型的拟合。这可能包括迭代学习系数的算法，比如使用随机梯度下降的人工神经网络。它还会影响寻求数据良好分割的模型，如支持向量机和决策树。

调整目标类分布可能会很有用。在某些情况下，为严重不平衡的数据集寻求平衡分布可能会导致受影响的算法对少数类进行过度填充，从而导致泛化错误增加。这种效果在训练数据集上表现较好，但在保持数据集或测试数据集上表现较差。

> ……随机过采样可能会增加过拟合的可能性，因为它会精确复制少数类示例。例如，通过这种方式，符号分类器可以构建表面上准确但实际上覆盖了一个复制示例的规则。

—第 83 页，[从不平衡数据集](https://amzn.to/307Xlva)中学习，2018。

因此，为了深入了解该方法的影响，最好在过采样后监控训练数据集和测试数据集的表现，并将结果与原始数据集上的相同算法进行比较。

当拟合模型时，尤其是考虑到模型在训练数据集中一次又一次地看到相同的例子，少数类的例子数量的增加，尤其是如果类偏斜严重，也会导致计算成本的显著增加。

> ……在随机过采样中，少数类示例的随机副本被添加到数据中。这可能会增加过拟合的可能性，特别是对于较高的过采样率。此外，它可能会降低分类器的表现并增加计算量。

——[不平衡分布下的预测建模综述](https://arxiv.org/abs/1505.01658)，2015 年。

可以使用[随机过采样类](https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.RandomOverSampler.html)实现随机过采样。

可以定义该类，并采用 *sampling_strategy* 参数，该参数可以设置为“*少数民族*，以自动平衡少数民族类和多数民族类。

例如:

```py
...
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
```

这意味着，如果多数类有 1，000 个例子，少数类有 100 个，这个策略将对少数类进行过采样，使其有 1，000 个例子。

可以指定一个浮点值来指示转换数据集中少数类多数示例的比率。例如:

```py
...
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy=0.5)
```

对于二进制分类问题，这将确保少数类被过采样为多数类的一半数量的例子。这意味着，如果多数类有 1，000 个示例，少数类有 100 个示例，则转换后的数据集将有 500 个少数类示例。

该类类似于 Sklearn 转换对象，因为它适合数据集，然后用于生成新的或转换的数据集。与 Sklearn 转换不同，它将改变数据集中的示例数量，而不仅仅是值(如缩放器)或特征数量(如投影)。

例如，可以通过调用 *fit_sample()* 函数一步拟合应用:

```py
...
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
```

我们可以在一个简单的综合二分类问题上演示这一点，该问题具有 1:100 的类别不平衡。

```py
...
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
```

下面列出了定义数据集和执行随机过采样以平衡类分布的完整示例。

```py
# example of random oversampling to balance the class distribution
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# summarize class distribution
print(Counter(y))
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_over))
```

运行该示例首先创建数据集，然后总结类分布。我们可以看到，多数派有近 100 个 10K 的例子，少数派有 100 个。

然后定义随机过采样变换来平衡少数类，然后拟合并应用于数据集。报告了转换数据集的类分布，显示现在少数类的示例数与多数类相同。

```py
Counter({0: 9900, 1: 100})
Counter({0: 9900, 1: 9900})
```

该变换可用作*管道*的一部分，以确保它仅作为 k 倍交叉验证中每个分割的一部分应用于训练数据集。

不能使用传统的 Sklearn [管道](https://Sklearn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)；相反，可以使用来自不平衡学习库的[管道。例如:](https://imbalanced-learn.org/stable/generated/imblearn.pipeline.Pipeline.html)

```py
...
# pipeline
steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
```

下面的示例提供了一个完整的示例，用于在 1:100 类分布的不平衡数据集上评估决策树。

使用三次重复的重复 10 倍交叉验证对模型进行评估，并在每次重复中对训练数据集分别执行过采样，确保不会出现交叉验证前执行过采样时可能出现的数据泄漏。

```py
# example of evaluating a decision tree with random oversampling
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# define pipeline
steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
score = mean(scores)
print('F1 Score: %.3f' % score)
```

运行该示例使用过采样在不平衡数据集上评估决策树模型。

所选模型和重采样配置是任意的，旨在提供一个模板，您可以使用该模板来测试数据集和学习算法的欠采样，而不是最佳地求解合成数据集。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

使用默认的过采样策略，平衡少数类和多数类。报告每次折叠和每次重复的 F1 平均得分。

```py
F1 Score: 0.990
```

现在我们已经熟悉了过采样，让我们来看看欠采样。

## 随机欠采样不平衡数据集

随机欠采样包括从多数类中随机选择要从训练数据集中删除的示例。

这具有减少训练数据集的转换版本中多数类中的示例数量的效果。这个过程可以重复进行，直到达到期望的类别分布，例如每个类别的示例数量相等。

这种方法可能更适合那些存在类不平衡的数据集，尽管少数类中有足够数量的例子，这样一个有用的模型可以被拟合。

欠采样的一个限制是，多数类中可能有用、重要或对拟合稳健决策边界至关重要的示例被删除。假设示例是随机删除的，那么就没有办法从多数类中检测或保留“*好的*”或更多信息丰富的示例。

> ……在随机欠采样(可能)中，大量数据被丢弃。[……]这可能很成问题，因为这种数据的丢失会使少数和多数实例之间的决策边界更难学习，从而导致分类表现的损失。

—第 45 页，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013

随机欠采样技术可以使用[随机欠采样不平衡学习类](https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)来实现。

该类可以像上一节中的*randomoversacompler*类一样使用，除了策略影响多数类而不是少数类。例如，将 *sampling_strategy* 参数设置为“*多数*”将对由具有最大数量示例的类确定的多数类进行欠采样。

```py
...
# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')
```

例如，多数类中有 1，000 个示例，少数类中有 100 个示例的数据集将被欠采样，这样两个类在转换后的训练数据集中都会有 100 个示例。

我们还可以将 *sampling_strategy* 参数设置为一个浮点值，该值是相对于少数类的百分比，具体来说就是少数类中的示例数除以多数类中的示例数。例如，如果我们将不平衡数据集中的*采样策略*设置为 0.5，多数类中有 1000 个示例，少数类中有 100 个示例，那么转换后的数据集中多数类将有 200 个示例(或者 100/200 = 0.5)。

```py
...
# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy=0.5)
```

这可能是为了确保生成的数据集足够大以适合合理的模型，并且不会丢弃太多来自多数类的有用信息。

> 在随机欠采样中，可以通过随机选择 90 个多数类实例来创建平衡的类分布。生成的数据集将由 20 个实例组成:10 个(随机剩余的)多数类实例和(原始的)10 个少数类实例。

—第 45 页，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013

然后，通过调用 *fit_resample()* 函数并将未转换的数据集作为参数传递，可以一步完成转换并将其应用于数据集。

```py
...
# fit and apply the transform
X_over, y_over = undersample.fit_resample(X, y)
```

我们可以在具有 1:100 类别不平衡的数据集上演示这一点。

下面列出了完整的示例。

```py
# example of random undersampling to balance the class distribution
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# summarize class distribution
print(Counter(y))
# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
X_over, y_over = undersample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_over))
```

运行该示例首先创建数据集，并报告不平衡的类分布。

对数据集进行拟合和应用变换，并报告新的类分布。我们可以看到，多数类被欠采样，与少数类有相同数量的例子。

必须使用判断和经验结果来判断只有 200 个示例的训练数据集是否足以训练模型。

```py
Counter({0: 9900, 1: 100})
Counter({0: 100, 1: 100})
```

这种欠采样变换也可以用在流水线中，就像上一节中的过采样变换一样。

这允许仅使用评估方案(如 k-fold 交叉验证)将转换应用于训练数据集，从而避免模型评估中的任何数据泄漏。

```py
...
# define pipeline
steps = [('under', RandomUnderSampler()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
```

我们可以定义一个在不平衡类别数据集上拟合决策树的示例，在重复的 10 倍交叉验证的每个分割上对训练数据集应用欠采样变换。

下面列出了完整的示例。

```py
# example of evaluating a decision tree with random undersampling
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# define pipeline
steps = [('under', RandomUnderSampler()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
score = mean(scores)
print('F1 Score: %.3f' % score)
```

运行该示例在欠采样的不平衡数据集上评估决策树模型。

所选模型和重采样配置是任意的，旨在提供一个模板，您可以使用该模板来测试数据集和学习算法的欠采样，而不是最佳地求解合成数据集。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

使用默认的欠采样策略，平衡多数类和少数类。报告每次折叠和每次重复的 F1 平均得分。

```py
F1 Score: 0.889
```

## 结合随机过采样和欠采样

结合随机过采样和欠采样可以获得有趣的结果。

例如，可以对少数类应用适度的过采样，以改善对这些示例的偏差，同时也可以对多数类应用适度的欠采样，以减少对该类的偏差。

与单独执行一种或另一种技术相比，这可以提高整体表现。

例如，如果我们有一个类分布为 1:100 的数据集，我们可能首先应用过采样，通过复制少数类中的示例将比率提高到 1:10，然后应用欠采样，通过删除多数类中的示例将比率进一步提高到 1:2。

这可以通过使用不平衡学习来实现，方法是将*采样策略*设置为 0.1 (10%)，然后使用*随机欠采样器*，将*采样策略*设置为 0.5 (50%)。例如:

```py
...
# define oversampling strategy
over = RandomOverSampler(sampling_strategy=0.1)
# fit and apply the transform
X, y = over.fit_resample(X, y)
# define undersampling strategy
under = RandomUnderSampler(sampling_strategy=0.5)
# fit and apply the transform
X, y = under.fit_resample(X, y)
```

我们可以在 1:100 类分布的合成数据集上演示这一点。完整的示例如下所示:

```py
# example of combining random oversampling and undersampling for imbalanced data
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# summarize class distribution
print(Counter(y))
# define oversampling strategy
over = RandomOverSampler(sampling_strategy=0.1)
# fit and apply the transform
X, y = over.fit_resample(X, y)
# summarize class distribution
print(Counter(y))
# define undersampling strategy
under = RandomUnderSampler(sampling_strategy=0.5)
# fit and apply the transform
X, y = under.fit_resample(X, y)
# summarize class distribution
print(Counter(y))
```

运行该示例首先创建合成数据集并总结类分布，显示大约 1:100 的类分布。

然后应用过采样，将分布从大约 1:100 增加到大约 1:10。最后，应用欠采样，进一步将类别分布从 1:10 改善到大约 1:2

```py
Counter({0: 9900, 1: 100})
Counter({0: 9900, 1: 990})
Counter({0: 1980, 1: 990})
```

当使用 k-fold 交叉验证评估一个模型时，我们可能还想应用同样的混合方法。

这可以通过使用具有一系列转换的管道并以正在评估的模型结束来实现；例如:

```py
...
# define pipeline
over = RandomOverSampler(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under), ('m', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
```

我们可以用同一个合成数据集上的决策树模型来演示这一点。

下面列出了完整的示例。

```py
# example of evaluating a model with random oversampling and undersampling
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# define pipeline
over = RandomOverSampler(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under), ('m', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
score = mean(scores)
print('F1 Score: %.3f' % score)
```

运行该示例使用重复的 K 折交叉验证来评估决策树模型，其中对训练数据集进行变换，首先使用过采样，然后对执行的每个分割和重复进行欠采样。报告每次折叠和每次重复的 F1 平均得分。

**注**:考虑到算法或评估程序的随机性，或数值准确率的差异，您的[结果可能会有所不同](https://machinelearningmastery.com/different-results-each-time-in-machine-learning/)。考虑运行该示例几次，并比较平均结果。

所选模型和重采样配置是任意的，旨在提供一个模板，您可以使用该模板来测试数据集和学习算法的欠采样，而不是最佳地求解合成数据集。

```py
F1 Score: 0.985
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   第五章数据级预处理方法，[从不平衡数据集](https://amzn.to/307Xlva)中学习，2018。
*   第三章不平衡数据集:从采样到分类器，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

### 报纸

*   [平衡机器学习训练数据的几种方法的行为研究](https://dl.acm.org/citation.cfm?id=1007735)，2004。
*   [不平衡分布下的预测建模综述](https://arxiv.org/abs/1505.01658)，2015。

### 应用程序接口

*   [不平衡-学习文档](https://imbalanced-learn.org)。
*   [不平衡学习，GitHub](https://github.com/Sklearn-contrib/imbalanced-learn) 。
*   [imblearn.over_sampling。randomoversacompler API](https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.RandomOverSampler.html)。
*   [imb learn . pipeline . pipeline API](https://imbalanced-learn.org/stable/generated/imblearn.pipeline.Pipeline.html)。
*   [imblearn.under_sampling。随机欠采样应用编程接口](https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)。

### 文章

*   [数据分析中的过采样和欠采样，维基百科](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)。

## 摘要

在本教程中，您发现了用于不平衡分类的随机过采样和欠采样

具体来说，您了解到:

*   随机重采样为不平衡数据集的类分布重新平衡提供了一种简单的技术。
*   随机过采样会复制训练数据集中少数类的示例，并可能导致某些模型的过拟合。
*   随机欠采样会从多数类中删除示例，并可能导致丢失对模型来说非常宝贵的信息。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。*