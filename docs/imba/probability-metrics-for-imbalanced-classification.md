# 不平衡分类概率度量的温和介绍

> 原文：<https://machinelearningmastery.com/probability-metrics-for-imbalanced-classification/>

最后更新于 2020 年 1 月 14 日

分类预测建模包括预测示例的类别标签，尽管有些问题需要预测类别成员的概率。

对于这些问题，不需要清晰的类标签，而是需要属于每个类的每个示例的可能性，并在以后进行解释。因此，小的相对概率具有很大的意义，需要专门的度量来量化预测的概率。

在本教程中，您将发现评估不平衡分类的概率预测的指标。

完成本教程后，您将知道:

*   一些分类预测建模问题需要概率预测。
*   对数损失量化了预测和预期概率分布之间的平均差异。
*   Brier 评分量化了预测概率和预期概率之间的平均差异。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![A Gentle Introduction to Probability Metrics for Imbalanced Classification](img/52f01a61f25ecdbbb681d1f1dc6e78c5.png)

不平衡分类概率度量的温和介绍
图片由 [a4gpa](https://flickr.com/photos/a4gpa/195354385/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  概率度量
2.  不平衡分类的日志丢失
3.  不平衡分类的 Brier 评分

## 概率度量

分类预测建模包括预测一个例子的类别标签。

在某些问题上，不需要清晰的类标签，而更喜欢类成员的概率。概率总结了属于每个类别标签的示例的可能性(或不确定性)。概率更为微妙，可以由人类操作员或决策系统进行解释。

概率度量是那些专门设计来使用预测概率而不是简单的类标签来量化分类器模型技能的度量。它们通常是提供单个值的分数，该值可用于根据预测概率与预期类别概率的匹配程度来比较不同的模型。

实际上，数据集没有目标概率。相反，它将有类标签。

例如，一个两类(二进制)分类问题的类标签为 0 表示负的情况，1 表示正的情况。当一个例子的类标签为 0 时，那么类标签为 0 和 1 的概率分别为 1 和 0。当一个例子的类标签为 1 时，那么类标签为 0 和 1 的概率分别为 0 和 1。

*   **Class = 0 的例子** : P(class=0) = 1，P(class=1) = 0
*   **Class = 1 的例子** : P(class=0) = 0，P(class=1) = 1

我们可以看到这将如何扩展到三个级别或更多；例如:

*   **Class = 0 的例子** : P(class=0) = 1，P(class=1) = 0，P(class=2) = 0
*   **Class = 1 的例子** : P(class=0) = 0，P(class=1) = 1，P(class=2) = 0
*   **Class = 2**示例:P(class=0) = 0，P(class=1) = 0，P(class=2) = 1

在二分类问题的情况下，这种表示可以简化为只关注正类。

也就是说，我们只需要一个属于类 1 的例子的概率来表示二进制分类的概率(所谓[伯努利分布](https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/))；例如:

*   【Class = 0 的例子 : P(class=1) = 0
*   **Class = 1 的例子** : P(class=1) = 1

概率度量将总结类成员的预测分布与已知类概率分布的匹配程度。

这种对预测概率的关注可能意味着模型预测的清晰类标签被忽略。这种关注可能意味着，一个预测概率的模型在根据其清晰的类别标签进行评估时，可能会表现得很糟糕，比如使用准确性或类似的分数。这是因为尽管预测的概率可能显示出技巧，但是在被转换成清晰的类标签之前，它们必须用适当的阈值来解释。

此外，对预测概率的关注可能还要求在使用或评估之前校准由一些非线性模型预测的概率。一些模型将学习校准的概率作为训练过程的一部分(例如逻辑回归)，但许多模型不会也需要校准(例如支持向量机、决策树和神经网络)。

通常为每个示例计算给定的概率度量，然后对训练数据集中的所有示例进行平均。

有两种流行的评估预测概率的度量标准；它们是:

*   日志丢失
*   布瑞尔分数

让我们依次仔细看看每一个。

## 不平衡分类的日志丢失

对数损失或简称对数损失是一个已知的损失函数，用于训练逻辑回归分类算法。

对数损失函数计算二分类模型进行概率预测的负对数似然。最值得注意的是，这是逻辑回归，但是这个函数可以被其他模型使用，例如神经网络，并且被其他名称所知，例如[交叉熵](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)。

通常，可以使用每个类别的预期概率和每个类别的预测概率的自然对数来计算对数损失；例如:

*   log loss =-(P(class = 0)* log(P(class = 0))+(P(class = 1))* log(P(class = 1))

最佳可能的日志丢失为 0.0，对于逐渐恶化的分数，值为正到无穷大。

如果只是预测正类的概率，那么可以计算一个二分类预测的对数损失函数( *yhat* )与预期概率( *y* )的比较如下:

*   log loss =-((1–y)* log(1–y hat)+y * log(y hat))

例如，如果预期概率为 1.0，而模型预测为 0.8，则对数损失将为:

*   log loss =-((1–y)* log(1–y hat)+y * log(y hat))
*   log loss =-((1–1.0)* log(1–0.8)+1.0 * log(0.8))
*   LogLoss = -(-0.0 + -0.223)
*   LogLoss = 0.223

通过添加额外的项，可以针对多个类别扩大该计算；例如:

*   log loss =-(C y _ C * log(yhat _ C)中的和 C)

这种一般化也称为交叉熵，计算两个概率分布不同的位数(如果使用 log base-2)或 nats(如果使用 log base-e)。

具体来说，它建立在信息论中[熵的概念之上，并计算一个分布与另一个分布相比，表示或传输一个事件所需的平均位数。](https://machinelearningmastery.com/what-is-information-entropy/)

> …交叉熵是当我们使用模型 q 时，对来自分布为 p 的源的数据进行编码所需的平均位数…

—第 57 页，[机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。

如果我们考虑一个目标或潜在的概率分布 *P* 和目标分布的近似 *Q* ，那么 *Q* 从 *P* 的交叉熵就是用 *Q* 代替 *P* 来表示一个事件的附加比特数。

目前我们将坚持使用对数损失，因为这是在使用这种计算作为分类器模型的评估指标时最常用的术语。

当计算与测试数据集中的一组预期概率相比的一组预测的对数损失时，计算并报告所有样本的对数损失平均值；例如:

*   AverageLogLoss = 1/N * N 中的和 I-((1–y)* log(1–yhat)+y * log(yhat))

训练数据集中一组预测的平均对数损失通常简称为对数损失。

我们可以用一个实例来演示如何计算对数损失。

首先，让我们定义一个合成的二进制类别数据集。我们将使用 [make_classification()函数](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)创建 1000 个示例，两个类的分割比例为 99%/1%。下面列出了创建和汇总数据集的完整示例。

```py
# create an imbalanced dataset
from numpy import unique
from sklearn.datasets import make_classification
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
# summarize dataset
classes = unique(y)
total = len(y)
for c in classes:
	n_examples = len(y[y==c])
	percent = n_examples / total * 100
	print('> Class=%d : %d/%d (%.1f%%)' % (c, n_examples, total, percent))
```

运行该示例会创建数据集，并报告每个类中示例的分布。

```py
> Class=0 : 990/1000 (99.0%)
> Class=1 : 10/1000 (1.0%)
```

接下来，我们将开发一种直觉，对概率进行天真的预测。

一个天真的预测策略是预测多数类的确定性，或者 P(类=0) = 1。另一种策略是预测少数民族，或 P(类=1) = 1。

可以使用 [log_loss() Sklearn 功能](https://Sklearn.org/stable/modules/generated/sklearn.metrics.log_loss.html)计算 Log loss。它将每个类的概率作为输入，并返回平均日志丢失。具体来说，每个示例必须有一个预测，每个类有一个概率，这意味着一个二进制分类问题示例的预测必须有一个类 0 和类 1 的概率。

因此，为所有示例预测类别 0 的某些概率将按如下方式实现:

```py
...
# no skill prediction 0
probabilities = [[1, 0] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
print('P(class0=1): Log Loss=%.3f' % (avg_logloss))
```

我们可以对 P(类 1)=1 做同样的事情。

这两种策略预计会表现糟糕。

一个更好的简单策略是预测每个例子的类分布。例如，因为我们的数据集对于多数和少数类具有 99%/1%的类分布，所以对于每个示例，该分布可以是“*预测的*，以给出概率预测的基线。

```py
...
# baseline probabilities
probabilities = [[0.99, 0.01] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
print('Baseline: Log Loss=%.3f' % (avg_logloss))
```

最后，我们还可以通过将测试集的目标值作为预测来计算完美预测概率的对数损失。

```py
...
# perfect probabilities
avg_logloss = log_loss(testy, testy)
print('Perfect: Log Loss=%.3f' % (avg_logloss))
```

将这些结合在一起，完整的示例如下所示。

```py
# log loss for naive probability predictions.
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
# split into train/test sets with same class ratio
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# no skill prediction 0
probabilities = [[1, 0] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
print('P(class0=1): Log Loss=%.3f' % (avg_logloss))
# no skill prediction 1
probabilities = [[0, 1] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
print('P(class1=1): Log Loss=%.3f' % (avg_logloss))
# baseline probabilities
probabilities = [[0.99, 0.01] for _ in range(len(testy))]
avg_logloss = log_loss(testy, probabilities)
print('Baseline: Log Loss=%.3f' % (avg_logloss))
# perfect probabilities
avg_logloss = log_loss(testy, testy)
print('Perfect: Log Loss=%.3f' % (avg_logloss))
```

运行该示例会报告每个简单策略的日志丢失。

正如预期的那样，预测每个类别标签的确定性会受到较大的日志丢失分数的惩罚，在所有情况下少数类别都是确定的情况下，分数会大得多。

我们可以看到，将数据集中的示例分布预测为基线比其他任何一种简单的度量都会得到更好的分数。该基线代表无技能分类器，低于该策略的日志丢失分数代表具有某种技能的模型。

最后，我们可以看到完美预测概率的对数损失为 0.0，表明实际和预测概率分布之间没有差异。

```py
P(class0=1): Log Loss=0.345
P(class1=1): Log Loss=34.193
Baseline: Log Loss=0.056
Perfect: Log Loss=0.000
```

现在我们已经熟悉了日志丢失，让我们来看看 Brier 评分。

## 不平衡分类的 Brier 评分

以格伦·布瑞尔命名的布瑞尔分数计算预测概率和期望值之间的均方误差。

分数总结了概率预测中误差的大小，是为二分类问题设计的。它侧重于评估正类的概率。然而，它可以适用于多类问题。

因此，它是不平衡分类问题的一个合适的概率度量。

> 概率分数的评估通常通过布瑞尔分数进行。基本思想是计算预测概率分数和真实类别指标之间的均方误差(MSE)，其中正类别编码为 1，负类别编码为 0。

—第 57 页，[从不平衡数据集](https://amzn.to/307Xlva)中学习，2018。

错误分数总是在 0.0 到 1.0 之间，其中技能完美的模型分数为 0.0。

与预期概率( *y* )相比，正预测概率( *yhat* )的 Brier 得分可计算如下:

*   briers core = 1/n * I 与 n 之和(yhat _ I–y_i)^2)

例如，如果预测的正类别概率为 0.8，预期概率为 1.0，则布瑞尔分数计算如下:

*   briers core =(yhat _ I–y_i)^2)
*   briers core =(0.8–1.0)^2
*   BrierScore = 0.04

我们可以用一个工作示例来演示如何使用上一节中使用的相同数据集和简单预测模型来计算布瑞尔分数。

Brier 分数可以使用[Brier _ score _ loss()Sklearn 功能](https://Sklearn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)计算。它只接受正类的概率，并返回一个平均分数。

如前一节所述，我们可以评估预测每个类标签的确定性的简单策略。在这种情况下，由于分数只考虑了正类的概率，这将涉及预测 P(类=1)=0 的 0.0 和 P(类=1)=1 的 1.0。例如:

```py
...
# no skill prediction 0
probabilities = [0.0 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
print('P(class1=0): Brier Score=%.4f' % (avg_brier))
# no skill prediction 1
probabilities = [1.0 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
print('P(class1=1): Brier Score=%.4f' % (avg_brier))
```

我们还可以测试无技能分类器，该分类器预测数据集中正面示例的比率，在本例中为 1%或 0.01。

```py
...
# baseline probabilities
probabilities = [0.01 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
print('Baseline: Brier Score=%.4f' % (avg_brier))
```

最后，我们还可以确认完美预测概率的布瑞尔分数。

```py
...
# perfect probabilities
avg_brier = brier_score_loss(testy, testy)
print('Perfect: Brier Score=%.4f' % (avg_brier))
```

将这些联系在一起，完整的示例如下所示。

```py
# brier score for naive probability predictions.
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
# split into train/test sets with same class ratio
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# no skill prediction 0
probabilities = [0.0 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
print('P(class1=0): Brier Score=%.4f' % (avg_brier))
# no skill prediction 1
probabilities = [1.0 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
print('P(class1=1): Brier Score=%.4f' % (avg_brier))
# baseline probabilities
probabilities = [0.01 for _ in range(len(testy))]
avg_brier = brier_score_loss(testy, probabilities)
print('Baseline: Brier Score=%.4f' % (avg_brier))
# perfect probabilities
avg_brier = brier_score_loss(testy, testy)
print('Perfect: Brier Score=%.4f' % (avg_brier))
```

运行这个例子，我们可以看到朴素模型和基线无技能分类器的分数。

正如我们可能预期的那样，我们可以看到，为所有示例预测 0.0 会导致较低的分数，因为测试集中所有 0.0 预测和大多数 0 类之间的均方误差会导致较小的值。相反，1.0 预测和大部分 0 类值之间的误差会导致更大的误差分数。

重要的是，我们可以看到默认的无技能分类器比预测所有 0.0 值的结果得分更低。同样，这代表基线分数，低于该分数的模型将展示技能。

```py
P(class1=0): Brier Score=0.0100
P(class1=1): Brier Score=0.9900
Baseline: Brier Score=0.0099
Perfect: Brier Score=0.0000
```

布瑞尔分数可能会变得非常小，重点将放在远低于小数点的分数上。例如，在上面的例子中，基线分数和完美分数之间的差异在小数点后四位很小。

一种常见的做法是使用参考分数来转换分数，例如无技能分类器。这被称为简单技能分数，计算如下:

*   BrierSkillScore = 1–(BrierScore/BrierScore _ ref)

我们可以看到，如果参考分数被评估，它将导致 0.0 的 BSS。这代表着没有技能的预测。低于此值将为负值，表示比没有技能更糟糕。高于 0.0 的值代表熟练的预测，完美的预测值为 1.0。

我们可以通过开发一个函数来计算下面列出的 Brier 技能分数来演示这一点。

```py
# calculate the brier skill score
def brier_skill_score(y, yhat, brier_ref):
	# calculate the brier score
	bs = brier_score_loss(y, yhat)
	# calculate skill score
	return 1.0 - (bs / brier_ref)
```

然后，我们可以为每个天真的预测和完美的预测计算平衡计分卡。

下面列出了完整的示例。

```py
# brier skill score for naive probability predictions.
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

# calculate the brier skill score
def brier_skill_score(y, yhat, brier_ref):
	# calculate the brier score
	bs = brier_score_loss(y, yhat)
	# calculate skill score
	return 1.0 - (bs / brier_ref)

# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99], flip_y=0, random_state=1)
# split into train/test sets with same class ratio
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# calculate reference
probabilities = [0.01 for _ in range(len(testy))]
brier_ref = brier_score_loss(testy, probabilities)
print('Reference: Brier Score=%.4f' % (brier_ref))
# no skill prediction 0
probabilities = [0.0 for _ in range(len(testy))]
bss = brier_skill_score(testy, probabilities, brier_ref)
print('P(class1=0): BSS=%.4f' % (bss))
# no skill prediction 1
probabilities = [1.0 for _ in range(len(testy))]
bss = brier_skill_score(testy, probabilities, brier_ref)
print('P(class1=1): BSS=%.4f' % (bss))
# baseline probabilities
probabilities = [0.01 for _ in range(len(testy))]
bss = brier_skill_score(testy, probabilities, brier_ref)
print('Baseline: BSS=%.4f' % (bss))
# perfect probabilities
bss = brier_skill_score(testy, testy, brier_ref)
print('Perfect: BSS=%.4f' % (bss))
```

运行该示例首先计算 BSS 计算中使用的参考 Brier 分数。

然后我们可以看到，预测每个职业的确定性分数会导致负的 BSS 分数，这表明他们比没有技能更糟糕。最后，我们可以看到评估参考预测本身的结果为 0.0，表明没有技能，评估真实值作为预测的结果为 1.0 的满分。

因此，布瑞尔技能评分是评估概率预测的最佳实践，广泛用于例行评估概率分类预测的地方，例如天气预报(例如，是否下雨)。

```py
Reference: Brier Score=0.0099
P(class1=0): BSS=-0.0101
P(class1=1): BSS=-99.0000
Baseline: BSS=0.0000
Perfect: BSS=1.0000
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [Python 中概率评分方法的温和介绍](https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/)
*   [机器学习交叉熵的温和介绍](https://machinelearningmastery.com/what-is-cross-entropy)
*   [带最大似然估计的逻辑回归入门](https://machinelearningmastery.com/logistic-regression-with-maximum-likelihood-estimation)

### 书

*   第 8 章不平衡学习的评估指标，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。
*   第三章绩效衡量，[从不平衡数据集](https://amzn.to/307Xlva)中学习，2018。

### 应用程序接口

*   [sklearn . datasets . make _ classification API](https://Sklearn.org/stable/modules/generated/sklearn.datasets.make_classification.html)。
*   [sklearn . metrics . log _ loss API](https://Sklearn.org/stable/modules/generated/sklearn.metrics.log_loss.html)。
*   [sklearn . metrics . brier _ score _ loss API](https://Sklearn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)。

### 文章

*   [布瑞尔评分，维基百科](https://en.wikipedia.org/wiki/Brier_score)。
*   [交叉熵，维基百科](https://en.wikipedia.org/wiki/Cross_entropy)。
*   [预测验证研究联合工作组](https://www.cawcr.gov.au/projects/verification/)

## 摘要

在本教程中，您发现了评估不平衡分类的概率预测的指标。

具体来说，您了解到:

*   一些分类预测建模问题需要概率预测。
*   对数损失量化了预测和预期概率分布之间的平均差异。
*   Brier 评分量化了预测概率和预期概率之间的平均差异。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。