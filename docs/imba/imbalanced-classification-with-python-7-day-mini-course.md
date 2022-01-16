# Python 不平衡分类（7 天迷你课程）

> 原文：<https://machinelearningmastery.com/imbalanced-classification-with-python-7-day-mini-course/>

最后更新于 2021 年 1 月 5 日

#### 不平衡分类速成班。
7 天内登上不平衡分类榜首。

分类预测建模是给一个例子分配一个标签的任务。

不平衡分类是那些分类任务，其中例子在类之间的分布不相等。

实际的不平衡分类需要使用一套专门的技术、数据准备技术、学习算法和表现指标。

在这个速成课程中，您将发现如何在七天内用 Python 开始并自信地完成一个不平衡的分类项目。

这是一个又大又重要的岗位。你可能想把它做成书签。

我们开始吧。

*   **2021 年 1 月更新**:更新了 API 文档的链接。

![Imbalanced Classification With Python (7-Day Mini-Course)](img/f7b02d710e0b3ccb20c5406164995b2b.png)

巨蟒不平衡分类(7 天迷你课程)
图片由[拱门国家公园](https://flickr.com/photos/archesnps/8406457380/)提供，保留部分权利。

## 这个速成班是给谁的？

在我们开始之前，让我们确保你在正确的地方。

本课程面向可能了解一些应用机器学习的开发人员。也许你知道如何使用流行的工具来端到端地解决预测建模问题，或者至少解决大部分主要步骤。

本课程中的课程假设了您的一些情况，例如:

*   你对编程的基本 Python 很熟悉。
*   您可能知道一些用于数组操作的基本 NumPy。
*   你可能知道一些基本的 sci kit-学习建模。

你不需要:

*   数学天才！
*   机器学习专家！

这门速成课程将把你从一个懂一点机器学习的开发人员带到一个能驾驭不平衡分类项目的开发人员。

注意:本速成课程假设您有一个至少安装了 NumPy 的工作 Python 3 SciPy 环境。如果您需要环境方面的帮助，可以遵循这里的逐步教程:

*   [如何用 Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 设置机器学习的 Python 环境

## 速成班概述

这门速成课分为七节课。

你可以每天完成一节课(*推荐的*)或者一天完成所有的课(*铁杆*)。这真的取决于你有多少时间和你的热情程度。

下面是 Python 中不平衡分类的七个入门和有效经验:

*   **第 01 课**:不平衡分类的挑战
*   **第 02 课**:不平衡数据的直觉
*   **第 03 课**:评估不平衡分类模型
*   **第 04 课**:对多数班进行欠采样
*   **第 05 课**:对少数民族班级进行过采样
*   **第 06 课**:结合数据欠采样和过采样
*   **第 07 课**:成本敏感算法

每节课可能需要你 60 秒或 30 分钟。慢慢来，按照自己的节奏完成课程。提问，甚至在下面的评论中发布结果。

这些课程可能期望你去发现如何做事。我会给你一些提示，但是每节课的部分要点是迫使你学习去哪里寻找关于 Python 中的算法和最佳工具的帮助。( ***提示*** :所有答案我都直接在这个博客上了；使用搜索框。)

在评论中发布您的结果；我会为你加油的！

坚持住。不要放弃。

**注**:这只是速成班。关于更多的细节和充实的教程，请参阅我的书，题目是“Python 的[不平衡分类”](https://machinelearningmastery.com/imbalanced-classification-with-python/)

## 第一课:不平衡分类的挑战

在本课中，您将发现不平衡分类问题的挑战。

不平衡的分类问题对预测建模提出了挑战，因为大多数用于分类的机器学习算法都是围绕每个类具有相同数量的例子的假设而设计的。

这导致模型的预测表现很差，特别是对于少数群体。这是一个问题，因为通常情况下，少数类更重要，因此该问题对少数类的分类错误比多数类更敏感。

*   **多数类**:半数以上的例子都属于这一类，往往是否定或正常的情况。
*   **少数民族类**:属于这一类的例子不到一半，往往是阳性或异常情况。

一个分类问题可能会有点歪斜，比如说是不是有轻微的不平衡。或者，对于给定的训练数据集，分类问题可能具有严重的不平衡，其中一个类中可能有数百或数千个示例，而另一个类中可能有数十个示例。

*   **轻微不平衡**。其中示例的分布在训练数据集中不均匀少量(例如 4:6)。
*   **严重失衡**。其中示例的分布在训练数据集中大量不均匀(例如 1:100 或更多)。

我们在实践中感兴趣解决的许多分类预测建模问题是不平衡的。

因此，不平衡分类没有得到比它更多的关注是令人惊讶的。

### 你的任务

在这节课中，你必须列出五个固有的阶级不平衡问题的一般例子。

一个例子可能是欺诈检测，另一个可能是入侵检测。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，你将发现如何培养对倾斜类分布的直觉。

## 第二课:不平衡数据的直觉

在本课中，您将发现如何为不平衡的类别数据集开发实用的直觉。

对于处理不平衡分类问题的初学者来说，一个挑战是特定的倾斜类分布意味着什么。例如，1:10 与 1:100 的班级比例的区别和含义是什么？

[make _ classification()](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)sci kit-learn 函数可用于定义具有所需类别不平衡的合成数据集。“*权重*参数指定了负类中示例的比例，例如[0.99，0.01]表示 99%的示例属于多数类，其余 1%属于少数类。

```py
...
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0)
```

一旦定义，我们就可以使用 [Counter](https://docs.python.org/3/library/collections.html#collections.Counter) 对象来总结类分布，以了解每个类到底有多少个示例。

```py
...
# summarize class distribution
counter = Counter(y)
print(counter)
```

我们还可以创建数据集的散点图，因为只有两个输入变量。然后，这些点可以被每个类着色。这个图提供了一个直观的直觉，说明了 99%对 1%的多数/少数阶级不平衡实际上是什么样子的。

下面列出了创建和总结不平衡类别数据集的完整示例。

```py
# plot imbalanced classification problem
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99, 0.01], flip_y=0)
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

### 你的任务

在本课中，您必须运行示例并复习情节。

对于奖励积分，您可以测试不同的班级比例并查看结果。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何评估不平衡分类的模型。

## 第 03 课:评估不平衡分类模型

在本课中，您将发现如何评估不平衡分类问题的模型。

预测准确率是分类任务最常见的度量标准，尽管当用于不平衡的分类任务时，它是不合适的，并且有潜在的危险误导。

这样做的原因是，如果 98%的数据属于负类，那么只需一直预测负类，就可以达到平均 98%的准确率，达到一个天真地看起来不错，但实际上没有技巧的分数。

相反，必须采用替代表现指标。

流行的替代方法是准确率和召回分数，这使得模型的表现可以通过关注被称为正类的少数类来考虑。

准确率计算正确预测的正例数除以预测的正例总数的比率。最大化准确率将最小化误报。

*   准确率=真阳性/(真阳性+假阳性)

回忆预测正确预测的正面例子总数除以本来可以预测的正面例子总数的比率。最大限度地召回将最大限度地减少假阴性。

*   回忆=真阳性/(真阳性+假阴性)

一个模型的表现可以用一个平均准确率和召回率的分数来概括，称为 F-Measure。最大化 F-Measure 将同时最大化精确度和召回率。

*   F-measure = (2 *准确率*召回)/(准确率+召回)

以下示例将逻辑回归模型应用于不平衡分类问题，并计算准确率，然后可以将其与[准确率](https://Sklearn.org/stable/modules/generated/sklearn.metrics.precision_score.html)、[召回](https://Sklearn.org/stable/modules/generated/sklearn.metrics.recall_score.html)和 [F-measure](https://Sklearn.org/stable/modules/generated/sklearn.metrics.f1_score.html) 进行比较。

```py
# evaluate imbalanced classification model with different metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0)
# split into train/test sets with same class ratio
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, stratify=y)
# define model
model = LogisticRegression(solver='liblinear')
# fit model
model.fit(trainX, trainy)
# predict on test set
yhat = model.predict(testX)
# evaluate predictions
print('Accuracy: %.3f' % accuracy_score(testy, yhat))
print('Precision: %.3f' % precision_score(testy, yhat))
print('Recall: %.3f' % recall_score(testy, yhat))
print('F-measure: %.3f' % f1_score(testy, yhat))
```

### 你的任务

在本课中，您必须运行该示例，并将分类准确率与其他指标(如准确率、召回率和 F-measure)进行比较。

对于奖励积分，尝试其他指标，如 Fbeta-measure 和 ROC AUC 分数。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，你将发现如何对大多数班级进行欠采样。

## 第 04 课:对大多数班级采样不足

在本课中，您将了解如何在训练数据集中对多数类进行欠采样。

在不平衡的数据集上使用标准机器学习算法的一个简单方法是改变训练数据集以具有更平衡的类分布。

这可以通过从多数类中删除示例来实现，多数类被称为“*欠采样*”一个可能的缺点是，多数类中在建模过程中有用的例子可能会被删除。

[不平衡学习库](https://imbalanced-learn.org)提供了许多欠采样算法的例子。可以使用 pip 轻松安装此库；例如:

```py
pip install imbalanced-learn
```

一种快速可靠的方法是从多数类中随机删除示例，以将不平衡降低到不太严重的比例，甚至使类均匀。

下面的例子创建了一个合成的不平衡类别数据，然后使用[随机欠采样](https://imbalanced-learn.org/stable/generated/imblearn.under_sampling.RandomUnderSampler.html)类将类分布从 1:100 的少数类更改为多数类，再更改为不太严重的 1:2。

```py
# example of undersampling the majority class
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99, 0.01], flip_y=0)
# summarize class distribution
print(Counter(y))
# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy=0.5)
# fit and apply the transform
X_under, y_under = undersample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_under))
```

### 你的任务

对于本课，您必须运行示例，并注意对多数类进行欠采样前后类分布的变化。

对于加分，尝试其他欠采样比率，甚至尝试不平衡学习库提供的其他欠采样技术。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何对少数族裔进行过采样。

## 第 05 课:对少数族裔进行过采样

在本课中，您将了解如何对训练数据集中的少数类进行过采样。

从多数类中删除示例的替代方法是从少数类中添加新的示例。

这可以通过简单地复制少数民族类中的例子来实现，但是这些例子没有添加任何新的信息。相反，可以使用训练数据集中的现有示例来合成少数民族的新示例。这些新示例将与特征空间中的现有示例“接近”*，但在微小但随机的方式上有所不同。*

 *SMOTE 算法是对少数类进行过采样的一种流行方法。这种技术可以用来减少不平衡或使阶级分布均匀。

下面的例子演示了在合成数据集上使用不平衡学习库提供的 [SMOTE](https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.SMOTE.html) 类。初始类别分布为 1:100，少数类别被过采样为 1:2 分布。

```py
# example of oversampling the minority class
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99, 0.01], flip_y=0)
# summarize class distribution
print(Counter(y))
# define oversample strategy
oversample = SMOTE(sampling_strategy=0.5)
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_over))
```

### 你的任务

对于本课，您必须运行该示例，并注意对少数类进行过采样前后类分布的变化。

要获得加分，可以尝试其他过采样率，甚至可以尝试不平衡学习库提供的其他过采样技术。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将了解如何将欠采样和过采样技术结合起来。

## 第 6 课:结合数据欠采样和过采样

在本课中，您将了解如何在训练数据集中组合数据欠采样和过采样。

数据欠采样将从多数类中删除示例，而数据过采样将向少数类添加示例。这两种方法可以结合起来，并在单个训练数据集上使用。

考虑到有这么多不同的数据采样技术可供选择，合并哪种方法可能会令人困惑。令人欣慰的是，有一些常见的组合在实践中表现良好；一些例子包括:

*   使用 SMOTE 过采样的随机欠采样。
*   Tomek 将欠采样与 SMOTE 过采样相结合。
*   使用 SMOTE 过采样编辑最近邻欠采样。

通过首先应用一种采样算法，然后应用另一种算法，可以将这些组合手动应用于给定的训练数据集。谢天谢地，不平衡学习库提供了常见的组合数据采样技术的实现。

下面的例子演示了如何使用 [SMOTEENN](https://imbalanced-learn.org/stable/generated/imblearn.combine.SMOTEENN.html) ，它结合了少数类的 SMOTE 过采样和多数类的编辑最近邻欠采样。

```py
# example of both undersampling and oversampling
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.combine import SMOTEENN
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99, 0.01], flip_y=0)
# summarize class distribution
print(Counter(y))
# define sampling strategy
sample = SMOTEENN(sampling_strategy=0.5)
# fit and apply the transform
X_over, y_over = sample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_over))
```

### 你的任务

对于本课，您必须运行该示例，并注意数据采样前后类分布的变化。

对于加分，尝试其他组合数据采样技术，甚至尝试手动应用过采样，然后对数据集进行欠采样。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何使用成本敏感算法进行不平衡分类。

## 第七课:成本敏感算法

在本课中，您将发现如何使用成本敏感算法进行不平衡分类。

大多数机器学习算法都假设模型产生的所有错误分类错误都是相等的。不平衡分类问题通常不是这种情况，在不平衡分类问题中，遗漏一个正类或少数类比错误地将一个例子从负类或多数类中分类出来更糟糕。

成本敏感学习是机器学习的一个子领域，它在训练机器学习模型时考虑了预测误差的成本(以及潜在的其他成本)。许多机器学习算法可以被更新为对成本敏感，其中模型因一个类的错误分类错误而比另一个类(例如少数类)受到更多惩罚。

Sklearn 库通过定义模型时指定的 *class_weight* 属性为一系列算法提供了这种能力。可以指定与类别分布成反比的权重。

如果多数类和少数类的类分布为 0.99 到 0.01，那么*类权重*参数可以被定义为字典，该字典定义对多数类的错误的惩罚为 0.01，对少数类的错误的惩罚为 0.99，例如{0:0.01，1:0.99}。

这是一种有用的启发式方法，可以通过将 *class_weight* 参数设置为字符串“ *balanced* 来自动配置。

下面的示例演示了如何在不平衡的类别数据集上定义和拟合成本敏感的逻辑回归模型。

```py
# example of cost sensitive logistic regression for imbalanced classification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0)
# split into train/test sets with same class ratio
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, stratify=y)
# define model
model = LogisticRegression(solver='liblinear', class_weight='balanced')
# fit model
model.fit(trainX, trainy)
# predict on test set
yhat = model.predict(testX)
# evaluate predictions
print('F-Measure: %.3f' % f1_score(testy, yhat))
```

### 你的任务

在本课中，您必须运行示例并查看成本敏感型模型的表现。

对于加分，将表现与成本不敏感的逻辑回归版本进行比较。

在下面的评论中发表你的答案。我想看看你有什么想法。

这是迷你课程的最后一课。

## 末日！
( *看你走了多远*

你成功了。干得好！

花一点时间，回头看看你已经走了多远。

你发现了:

*   不平衡分类的挑战是缺乏少数群体的例子，以及不同类别分类错误的重要性不同。
*   如何为不平衡的类别数据集开发空间直觉，为数据准备和算法选择提供信息。
*   分类准确率的失败，以及像准确率、召回率和 F-测度这样的替代度量如何更好地总结不平衡数据集上的模型表现。
*   如何从训练数据集中的多数类中删除示例，称为数据欠采样。
*   如何在训练数据集中的少数类中合成新的示例，称为数据过采样。
*   如何在训练数据集中组合数据过采样和欠采样技术，以及产生良好表现的常见组合。
*   如何使用代价敏感的机器学习算法的改进版本来提高不平衡类别数据集的表现。

下一步，看看我用 Python 写的关于[不平衡分类的书](https://machinelearningmastery.com/imbalanced-classification-with-python/)。

## 摘要

**你觉得迷你课程怎么样？**
你喜欢这个速成班吗？

**你有什么问题吗？**有什么症结吗？
让我知道。请在下面留言。*