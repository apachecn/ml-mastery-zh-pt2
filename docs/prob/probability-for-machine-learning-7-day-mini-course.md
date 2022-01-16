# 面向机器学习的概率（7 天迷你课程）

> 原文：<https://machinelearningmastery.com/probability-for-machine-learning-7-day-mini-course/>

最后更新于 2020 年 1 月 10 日

#### 机器学习速成班的概率。
*7 天内获得机器学习所用概率榜首。*

概率是一个数学领域，被普遍认为是机器学习的基础。

虽然概率是一个大领域，有许多深奥的理论和发现，但机器学习从业者需要该领域的具体细节、工具和符号。有了概率是什么的坚实基础，就有可能只关注好的或相关的部分。

在这个速成课程中，您将发现如何在七天内开始并自信地理解和实现 Python 机器学习中使用的概率方法。

这是一个又大又重要的岗位。你可能想把它做成书签。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 1 月更新**:针对 Sklearn v0.22 API 的变化进行了更新。

![Probability for Machine Learning (7-Day Mini-Course)](img/08dde3a29a8c162a4a9777f93f62c491.png)

机器学习概率(7 天迷你课程)
图片由[珀西塔](https://www.flickr.com/photos/dittmars/35837926013/)提供，保留部分权利。

## 这个速成班是给谁的？

在我们开始之前，让我们确保你在正确的地方。

本课程面向可能了解一些应用机器学习的开发人员。也许你知道如何使用流行的工具来端到端地解决预测建模问题，或者至少解决大部分主要步骤。

本课程中的课程假设了您的一些情况，例如:

*   你对编程的基本 Python 很熟悉。
*   您可能知道一些用于数组操作的基本 NumPy。
*   你想学习概率来加深你对机器学习的理解和应用。

你不需要:

*   数学天才！
*   机器学习专家！

这门速成课程将把你从一个懂一点机器学习的开发人员带到一个能掌握概率方法基础知识的开发人员。

**注**:本速成课假设您有一个至少安装了 NumPy 的 Python3 SciPy 工作环境。如果您需要环境方面的帮助，可以遵循这里的逐步教程:

*   [如何用 Anaconda](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/) 设置机器学习的 Python 环境

## 速成班概述

这门速成课分为七节课。

您可以每天完成一节课(推荐)或一天内完成所有课程(硬核)。这真的取决于你有多少时间和你的热情程度。

下面列出了七堂课，这些课将帮助您开始使用 Python 进行机器学习，并提高学习效率:

*   **第 01 课**:概率与机器学习
*   **第 02 课**:三种概率
*   **第 03 课**:概率分布
*   **第 04 课**:朴素贝叶斯分类器
*   **第 05 课**:熵和交叉熵
*   **第 06 课**:幼稚量词
*   **第 07 课**:概率分数

每节课可能需要你 60 秒或 30 分钟。慢慢来，按照自己的节奏完成课程。提问，甚至在下面的评论中发布结果。

这些课程期望你去发现如何做事。我会给你一些提示，但是每节课的部分要点是迫使你学习去哪里寻找关于统计方法、NumPy API 和 Python 中最好的工具的帮助。(**提示**:所有答案我都直接在这个博客上了；使用搜索框。)

在评论中发布您的结果；我会为你加油的！

坚持住。不要放弃。

**注**:这只是速成班。关于更多的细节和充实的教程，请参阅我的书，题目是“机器学习的[概率](https://machinelearningmastery.com/probability-for-machine-learning/)”

## 第一课:概率和机器学习

在本课中，您将发现为什么机器学习实践者应该学习概率来提高他们的技能和能力。

概率是一个量化不确定性的数学领域。

机器学习是从不确定的数据中开发预测建模。不确定性意味着处理不完善或不完整的信息。

不确定性是机器学习领域的基础，然而它是给初学者造成最大困难的方面之一，尤其是那些来自开发人员背景的人。

机器学习中的不确定性主要有三个来源；它们是:

*   **观测中的噪声**，如测量误差和随机噪声。
*   **域的不完全覆盖**，比如你永远不可能观察到所有的数据。
*   **问题的模型不完善**，比如所有的模型都有错误，有些是有用的。

应用机器学习中的不确定性是用概率来管理的。

*   概率和统计帮助我们理解和量化我们从领域观察到的变量的期望值和可变性。
*   概率有助于理解和量化域中观测值的预期分布和密度。
*   概率有助于理解和量化我们的预测模型在应用于新数据时的预期能力和表现差异。

这是机器学习的基础。除此之外，我们可能需要模型来预测概率，我们可能使用概率来开发预测模型(例如朴素贝叶斯)，我们可能使用概率框架来训练预测模型(例如最大似然估计)。

### 你的任务

在这节课中，你必须列出你想在机器学习的背景下学习概率的三个原因。

这些可能与上面的一些原因有关，也可能是你自己的个人动机。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，你将发现三种不同类型的概率以及如何计算它们。

## 第二课:三种概率

在本课中，您将发现随机变量之间的联合概率、边缘概率和条件概率的温和介绍。

概率量化了事件发生的可能性。

具体来说，它量化了一个随机变量出现特定结果的可能性，例如掷硬币、掷骰子或从一副牌中抽一张扑克牌。

我们可以只讨论两个事件的概率:变量 *X* 的事件 *A* 的概率和变量 *Y* 的事件 *B* 的概率，简写就是 *X=A* 和 *Y=B* ，这两个变量在某种程度上是相关或相依的。

因此，我们可能要考虑三种主要类型的概率。

### 联合概率

我们可能对两个同时发生的事件的概率感兴趣，比如两个不同随机变量的结果。

例如，事件 *A* 和事件 *B* 的联合概率正式写成:

*   警队(甲及乙)

事件 *A* 和 *B* 的联合概率计算为事件 *A* 给定事件 *B* 乘以事件 *B* 的概率。

这可以正式表述如下:

*   P(A 和 B) = P(给定的 B) * P(B)

### 边缘概率

我们可能对一个随机变量的事件概率感兴趣，而不考虑另一个随机变量的结果。

边缘概率没有特别的符号；它只是第二个变量的所有事件的所有概率与第一个变量的给定固定事件的所有概率的和或并。

*   所有 Y 的和 P(X=A，Y=yi)

### 条件概率

考虑到另一个事件的发生，我们可能对一个事件的概率感兴趣。

例如，事件 *A* 给定事件 *B* 的条件概率正式写成:

*   给定的

事件 *A* 给定事件 *B* 的条件概率可以使用事件的联合概率计算如下:

*   P(给定的 B) = P(A 和 B) / P(B)

### 你的任务

在本课中，您必须练习计算联合概率、边缘概率和条件概率。

比如一个家庭有两个孩子，最大的是男孩，这个家庭有两个儿子的概率是多少？这被称为“[男孩或女孩问题](https://en.wikipedia.org/wiki/Boy_or_Girl_paradox)”，是练习概率的许多常见玩具问题之一。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，你将发现随机变量的概率分布。

## 第三课:概率分布

在本课中，您将发现概率分布的温和介绍。

在概率上，随机变量可以采用许多可能值中的一个，例如来自状态空间的事件。随机变量的一个特定值或一组值可以被赋予一个概率。

随机变量主要有两类。

*   **离散随机变量**。值是从一组有限的状态中得出的。
*   **连续随机变量**。数值是从一系列实数值中得出的。

离散随机变量有一组有限的状态；例如，汽车的颜色。连续随机变量有一个数值范围；比如人类的身高。

概率分布是随机变量的值的概率的汇总。

### 离散概率分布

离散概率分布概括了离散随机变量的概率。

众所周知的离散概率分布的一些例子包括:

*   泊松分布。
*   伯努利分布和二项分布。
*   多项式和多项式分布。

### 连续概率分布

连续概率分布概括了连续随机变量的概率。

众所周知的连续概率分布的一些例子包括:

*   正态或高斯分布。
*   指数分布。
*   帕累托分布。

### 随机采样高斯分布

我们可以定义一个均值为 50、标准差为 5 的分布，并从这个分布中抽取随机数。我们可以使用[正常()NumPy 功能](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html)来实现。

下面的示例从这个分布中采样并打印了 10 个数字。

```py
# sample a normal distribution
from numpy.random import normal
# define the distribution
mu = 50
sigma = 5
n = 10
# generate the sample
sample = normal(mu, sigma, n)
print(sample)
```

运行该示例将打印从定义的正态分布中随机采样的 10 个数字。

### 你的任务

在本课中，您必须开发一个示例，从不同的连续或离散概率分布函数中进行采样。

对于给定的分布，您可以绘制 x 轴上的值和 y 轴上的概率，以显示您选择的概率分布函数的密度。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现朴素贝叶斯分类器。

## 第 04 课:朴素贝叶斯分类器

在本课中，您将发现用于分类预测建模的朴素贝叶斯算法。

在机器学习中，我们经常对一个预测建模问题感兴趣，在这个问题中，我们希望为给定的观察预测一个类标签。

解决这个问题的一种方法是开发一个概率模型。从概率的角度来看，我们感兴趣的是估计给定观察的类标签的条件概率，或者给定输入数据的类*y*X 的概率。

*   P(y | X)

贝叶斯定理提供了一种替代的和有原则的方法来使用期望的条件概率的逆来计算条件概率，这通常更容易计算。

贝叶斯定理的简单计算形式如下:

*   P(A|B) = P(B|A) * P(A) / P(B)

其中我们感兴趣的概率计算 *P(A|B)* 称为后验概率，事件的边缘概率 *P(A)* 称为先验概率。

贝叶斯定理在分类中的直接应用变得棘手，尤其是当变量或特征的数量增加时。相反，我们可以简化计算，并假设每个输入变量都是独立的。尽管很引人注目，但这种更简单的计算通常会带来非常好的表现，即使输入变量高度相关。

我们可以从零开始实现这一点，方法是假设每个单独输入变量的概率分布，并计算每个特定输入值属于每个类的概率，然后将结果相乘，得到用于选择最有可能的类的分数。

*   P(yi | x1，x2，…，xn)= P(x1 | y1)* P(x2 | y1)*…P(xn | y1)* P(yi)

如果我们假设每个输入变量都是高斯分布，那么 Sklearn 库就提供了一个高效的算法实现。

为了使用 Sklearn 朴素贝叶斯模型，首先定义模型，然后将其拟合到训练数据集上。一旦拟合，概率可以通过 *predict_proba()* 函数预测，类标签可以通过 *predict()* 函数直接预测。

下面列出了将高斯朴素贝叶斯模型([高斯年](https://Sklearn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html))拟合到测试数据集的完整示例。

```py
# example of gaussian naive bayes
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# define the model
model = GaussianNB()
# fit the model
model.fit(X, y)
# select a single sample
Xsample, ysample = [X[0]], y[0]
# make a probabilistic prediction
yhat_prob = model.predict_proba(Xsample)
print('Predicted Probabilities: ', yhat_prob)
# make a classification prediction
yhat_class = model.predict(Xsample)
print('Predicted Class: ', yhat_class)
print('Truth: y=%d' % ysample)
```

运行该示例使模型适合训练数据集，然后对我们在前面示例中使用的第一个示例进行预测。

### 你的任务

对于本课，您必须运行示例并报告结果。

作为奖励，在真实的类别数据集上尝试该算法，例如流行的玩具分类问题，即基于花的测量来分类[鸢尾花种类](https://github.com/jbrownlee/Datasets/blob/master/iris.csv)。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，你将发现熵和交叉熵分数。

## 第五课:熵和交叉熵

在本课中，您将发现机器学习的交叉熵。

信息论是一个研究领域，涉及量化信息用于交流。

量化信息背后的直觉是衡量一个事件中有多少惊喜的想法。那些罕见的事件(低概率)更令人惊讶，因此比那些常见的事件(高概率)拥有更多的信息。

*   **低概率事件**:高信息(惊人)。
*   **高概率事件**:低信息(不出所料)。

我们可以利用事件发生的概率来计算事件中的信息量。

*   信息(x)=-对数(p(x))

我们也可以量化一个随机变量中有多少信息。

这称为熵，它概括了平均表示事件所需的信息量。

对于具有 K 个离散状态的随机变量 X，熵可以计算如下:

*   熵(X)=-和(i=1 至 K p(K) *对数(p(K)))

[交叉熵](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)是给定随机变量或事件集的两个概率分布之间差异的度量。在优化分类模型时，它被广泛用作损失函数。

它建立在熵的基础上，计算一个分布与另一个分布相比，表示或传输一个事件所需的平均位数。

*   交叉熵(P，Q)=–X 中的和 X P(x) *对数(Q(x))

我们可以用一个小例子来具体计算交叉熵。

考虑一个随机变量，它有三个不同颜色的事件。对于这个变量，我们可能有两种不同的概率分布。我们可以计算这两个分布之间的交叉熵。

下面列出了完整的示例。

```py
# example of calculating cross entropy
from math import log2

# calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i]*log2(q[i]) for i in range(len(p))])

# define data
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate cross entropy H(P, Q)
ce_pq = cross_entropy(p, q)
print('H(P, Q): %.3f bits' % ce_pq)
# calculate cross entropy H(Q, P)
ce_qp = cross_entropy(q, p)
print('H(Q, P): %.3f bits' % ce_qp)
```

运行该示例首先从 P 计算 Q 的交叉熵，然后从 Q 计算 P。

### 你的任务

对于本课，您必须运行示例并描述结果及其含义。比如交叉熵的计算是否对称？

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现如何开发和评估一个简单的分类器模型。

## 第 06 课:幼稚分类器

在本课中，您将发现如何为机器学习开发和评估幼稚的分类策略。

分类预测建模问题涉及预测给定模型输入的类别标签。

给定一个分类模型，如何知道模型有没有技巧？

这是每个分类预测建模项目中常见的问题。答案是将给定分类器模型的结果与基线或原始分类器模型进行比较。

考虑一个简单的两类分类问题，其中每个类的观察数不相等(例如不平衡)，0 类有 25 个例子，1 类有 75 个例子。这个问题可以用来考虑不同的朴素分类器模型。

例如，考虑一个以相等概率随机预测 0 类或 1 类的模型。它的表现如何？

我们可以使用一个简单的概率模型来计算预期表现。

*   p(yhat = y)= p(yhat = 0)* p(y = 0)+p(yhat = 1)* p(y = 1)

我们可以插入每个类的出现(0.25 和 0.75)和每个类的预测概率(0.5 和 0.5)，并估计模型的表现。

*   P(yhat = y) = 0.5 * 0.25 + 0.5 * 0.75
*   P(yhat = y) = 0.5

事实证明，这个分类器相当差。

现在，如果我们考虑每次预测多数班(1 班)会怎么样？同样，我们可以插入预测概率(0.0 和 1.0)并估计模型的表现。

*   P(yhat = y) = 0.0 * 0.25 + 1.0 * 0.75
*   P(yhat = y) = 0.75

事实证明，这一简单的改变产生了一个更好的朴素分类模型，并且可能是当类不平衡时使用的最好的朴素分类器。

Sklearn 机器学习库提供了一种称为 [DummyClassifier](https://Sklearn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) 的多数类朴素分类算法的实现，您可以在下一个分类预测建模项目中使用该算法。

下面列出了完整的示例。

```py
# example of the majority class naive classifier in Sklearn
from numpy import asarray
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
# define dataset
X = asarray([0 for _ in range(100)])
class0 = [0 for _ in range(25)]
class1 = [1 for _ in range(75)]
y = asarray(class0 + class1)
# reshape data for sklearn
X = X.reshape((len(X), 1))
# define model
model = DummyClassifier(strategy='most_frequent')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
# calculate accuracy
accuracy = accuracy_score(y, yhat)
print('Accuracy: %.3f' % accuracy)
```

运行该示例准备数据集，然后使用多数类策略在数据集上定义并拟合*DummyCollector*。

### 你的任务

在本课中，您必须运行示例并报告结果，确认模型是否按照我们的计算预期运行。

另外，计算一个朴素分类器模型的预期概率，该模型在每次进行预测时从训练数据集中随机选择一个类别标签。

在下面的评论中发表你的答案。我想看看你有什么想法。

在下一课中，您将发现预测概率的评分模型的指标。

## 第 07 课:概率分数

在本课中，您将发现两种评分方法，可用于评估分类预测建模问题的预测概率。

为分类问题预测概率而不是类别标签可以为预测提供额外的细微差别和不确定性。

增加的细微差别允许使用更复杂的度量来解释和评估预测的概率。

让我们仔细看看评估预测概率的两种流行评分方法。

### 日志丢失分数

逻辑损失，简称对数损失，计算预测概率和观察概率之间的对数似然性。

虽然它是为训练像逻辑回归这样的二分类模型而开发的，但它可以用于评估多类问题，并且在功能上等同于计算从信息论导出的交叉熵。

一个技能完美的模型的日志丢失分数为 0.0。日志丢失可以使用 Sklearn 中的 [log_loss()函数](https://Sklearn.org/stable/modules/generated/sklearn.metrics.log_loss.html)在 Python 中实现。

例如:

```py
# example of log loss
from numpy import asarray
from sklearn.metrics import log_loss
# define data
y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
# define data as expected, e.g. probability for each event {0, 1}
y_true = asarray([[v, 1-v] for v in y_true])
y_pred = asarray([[v, 1-v] for v in y_pred])
# calculate log loss
loss = log_loss(y_true, y_pred)
print(loss)
```

### 布瑞尔分数

布瑞尔分数以格伦·布瑞尔命名，计算预测概率和期望值之间的均方误差。

分数总结了概率预测中误差的大小。

错误分数总是在 0.0 到 1.0 之间，其中技能完美的模型分数为 0.0。

Brier 分数可以使用 Sklearn 中的 [brier_score_loss()函数在 Python 中计算。](https://Sklearn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)

例如:

```py
# example of brier loss
from sklearn.metrics import brier_score_loss
# define data
y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
# calculate brier score
score = brier_score_loss(y_true, y_pred, pos_label=1)
print(score)
```

### 你的任务

在本课中，您必须运行每个示例并报告结果。

作为奖励，改变模拟预测，使其更好或更差，并比较结果得分。

在下面的评论中发表你的答案。我想看看你有什么想法。

这是最后一课。

## 末日！
( *看你走了多远*

你成功了。干得好！

花一点时间，回头看看你已经走了多远。

你发现了:

*   概率在应用机器学习中的重要性。
*   概率的三种主要类型及其计算方法。
*   随机变量的概率分布及如何从中抽取随机样本？
*   如何使用贝叶斯定理计算条件概率，以及如何在分类模型中使用它。
*   如何计算信息、熵和交叉熵分数以及它们的含义。
*   如何开发和评估朴素分类模型的预期表现？
*   如何评估预测分类问题概率值的模型的技能？

下一步，看看我写的关于[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)的书。

## 摘要

**你觉得迷你课程怎么样？**
你喜欢这个速成班吗？

**你有什么问题吗？有什么症结吗？**
让我知道。请在下面留言。