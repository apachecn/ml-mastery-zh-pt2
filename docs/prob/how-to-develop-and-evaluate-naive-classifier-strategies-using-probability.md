# 如何利用概率开发和评估朴素分类器策略

> 原文：<https://machinelearningmastery.com/how-to-develop-and-evaluate-naive-classifier-strategies-using-probability/>

最后更新于 2019 年 9 月 25 日

朴素分类器是一个简单的分类模型，它对问题的假设很少甚至没有，它的表现提供了一个基线，通过这个基线可以比较数据集上评估的所有其他模型。

有不同的策略可以用于朴素分类器，有些策略比其他策略更好，这取决于数据集和表现度量的选择。最常见的表现度量是分类准确率和常见的朴素分类策略，包括随机猜测类标签、从训练数据集中随机选择标签以及使用多数类标签。

开发一个小概率框架来计算给定朴素分类策略的预期表现，并进行实验来证实理论预期是有用的。这些练习为一般的朴素分类算法的行为以及为分类任务建立表现基线的重要性提供了直觉。

在本教程中，您将发现如何为机器学习开发和评估朴素分类策略。

完成本教程后，您将知道:

*   朴素分类模型的表现提供了一个基线，通过这个基线，所有其他模型都可以被认为是熟练的或不熟练的。
*   多数类分类器比其他朴素分类器模型(如随机猜测和预测随机选择的观察类标签)获得更好的准确性。
*   朴素分类器策略可以通过 Sklearn 库中的 DummyClassifier 类用于预测建模项目。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![How to Develop and Evaluate Naive Classifier Strategies Using Probability](img/164c070a698e458377ea36980a30ea6a.png)

如何使用概率开发和评估朴素分类器策略
图片由[理查德·伦纳德](https://www.flickr.com/photos/richardleonard/104837076/)提供，版权所有。

## 教程概述

本教程分为五个部分；它们是:

1.  朴素分类器
2.  预测随机猜测
3.  预测随机选择的班级
4.  预测多数阶级
5.  Sklearn 中的朴素分类器

## 朴素分类器

分类预测建模问题涉及预测给定模型输入的类别标签。

分类模型适用于训练数据集，并在测试数据集上进行评估，表现通常被报告为正确预测数与所做预测总数之比的一小部分，称为准确性。

给定一个分类模型，如何知道模型有没有技巧？

这是每个分类预测建模项目中常见的问题。答案是将给定分类器模型的结果与基线或原始分类器模型进行比较。

一个简单的分类器模型是一个不使用任何复杂程度来进行预测的模型，通常进行随机或恒定的预测。这种模型是幼稚的，因为它们没有使用任何关于该领域的知识或任何学习来进行预测。

分类任务中基线分类器的表现提供了问题中所有其他模型的预期表现的下限。例如，如果一个分类模型比一个简单的分类器表现得更好，那么它就有一些技巧。如果一个分类器模型的表现比朴素分类器差，那么它就没有任何技能。

幼稚分类器应该用什么分类器？

这是初学者常见的困惑领域，采用了不同的幼稚分类器。

一些常见的选择包括:

*   预测一个随机类。
*   从训练数据集中预测一个随机选择的类。
*   从训练数据集中预测多数类。

问题是，并不是所有的幼稚分类器都是平等的，有些分类器的表现比其他分类器更好。因此，我们应该在所有的分类预测建模项目中使用表现最好的朴素分类器。

我们可以使用简单概率来评估不同朴素分类器模型的表现，并确定应该始终用作原生分类器的策略。

在我们开始评估不同的策略之前，让我们定义一个人为的两类分类问题。为了使其有趣，我们将假设每个类的观察数不相等(例如[问题是不平衡的](https://machinelearningmastery.com/ufaqs/how-do-i-handle-an-imbalance-in-classes/))，0 类有 25 个例子，1 类有 75 个例子。

我们可以用下面列出的 Python 中的一个小例子来具体说明这一点。

```py
# summarize a test dataset
# define dataset
class0 = [0 for _ in range(25)]
class1 = [1 for _ in range(75)]
y = class0 + class1
# summarize distribution
print('Class 0: %.3f' % (len(class0) / len(y) * 100))
print('Class 1: %.3f' % (len(class1) / len(y) * 100))
```

运行该示例会创建数据集，并汇总属于每个类的示例的分数，显示类 0 和类 1 的 25%和 75%，正如我们可能直观预期的那样。

```py
Class 0: 25.000
Class 1: 75.000
```

最后，我们可以定义一个评估朴素分类策略的概率模型。

在这种情况下，我们感兴趣的是计算给定二分类模型的分类准确率。

*   P(yhat = y)

这可以计算为模型预测每个类别值的概率乘以观察每个类别出现的概率。

*   p(yhat = y)= p(yhat = 0)* p(y = 0)+p(yhat = 1)* p(y = 1)

这将计算模型在数据集上的预期表现。它提供了一个非常简单的概率模型，我们可以用它来计算一个简单分类器模型的预期表现。

接下来，我们将使用这个人为的预测问题来探索一个朴素分类器的不同策略。

## 预测随机猜测

也许最简单的策略是随机猜测每个所需预测的可用类别之一。

我们称之为随机猜测策略。

使用我们的概率模型，我们可以计算出这个模型在我们设计的数据集上的平均表现。

每个类的随机猜测是每个可能的类标签上的均匀概率分布，或者在两类问题的情况下，每个类的概率为 0.5。同样，我们知道数据集的类 0 和类 1 的值的预期概率，因为我们设计了这个问题；分别为 0.25 和 0.75。因此，我们计算该策略的平均表现如下:

*   p(yhat = y)= p(yhat = 0)* p(y = 0)+p(yhat = 1)* p(y = 1)
*   P(yhat = y) = 0.5 * 0.25 + 0.5 * 0.75
*   P(yhat = y) = 0.125 + 0.375
*   P(yhat = y) = 0.5

该计算表明，在我们设计的问题上预测一致随机类标签的表现是 0.5%或 50%的分类准确率。

这可能会令人惊讶，但这很好，因为它强调了系统地计算一个幼稚策略的预期表现的好处。

我们可以用一个小实验来证实这个估计是正确的。

该策略可以实现为一个函数，为每个所需的预测随机选择 0 或 1。

```py
# guess random class
def random_guess():
	if random() < 0.5:
		return 0
	return 1
```

然后可以为数据集中所需的每个预测调用这个函数，并可以评估准确率

```py
...
yhat = [random_guess() for _ in range(len(y))]
acc = accuracy_score(y, yhat)
```

这只是一次试验，但每次使用该策略的准确性都会有所不同。

为了解决这个问题，我们可以重复实验 1000 次，并报告策略的平均表现。我们希望平均表现与上面计算的预期表现相匹配。

下面列出了完整的示例。

```py
# example of a random guess naive classifier
from numpy import mean
from numpy.random import random
from sklearn.metrics import accuracy_score

# guess random class
def random_guess():
	if random() < 0.5:
		return 0
	return 1

# define dataset
class0 = [0 for _ in range(25)]
class1 = [1 for _ in range(75)]
y = class0 + class1
# average performance over many repeats
results = list()
for _ in range(1000):
	yhat = [random_guess() for _ in range(len(y))]
	acc = accuracy_score(y, yhat)
	results.append(acc)
print('Mean: %.3f' % mean(results))
```

运行该示例执行了我们实验的 1000 次试验，并报告了策略的平均准确性。

鉴于算法的随机性，您的具体结果会有所不同。

在这种情况下，我们可以看到预期表现与计算表现非常接近。假设[大数定律](https://machinelearningmastery.com/a-gentle-introduction-to-the-law-of-large-numbers-in-machine-learning/)，我们进行的实验越多，我们的估计就越接近我们计算的理论值。

```py
Mean: 0.499
```

这是一个很好的开始，但是如果我们在策略中使用一些关于训练数据集组成的基本信息呢。接下来我们将探讨这一点。

## 预测随机选择的班级

另一种简单的分类方法是以某种方式利用训练数据集。

也许最简单的方法是将训练数据集中的观察结果用作预测。具体来说，我们可以在训练集中随机选择观察值，并为每个请求的预测返回它们。

这是有意义的，我们可能期望训练数据集的这种原始使用会产生比随机猜测稍好的原始准确性。

我们可以通过使用我们的概率框架计算该方法的预期表现来找出答案。

如果我们从具有均匀概率分布的训练数据集中选择示例，我们将从每个类中抽取示例，它们在训练数据集中出现的概率相同。也就是说，我们将绘制概率为 25%的类-0 和概率为 75%的类-1 的示例。这也是模型独立预测的概率。

有了这些知识，我们可以将这些值插入概率模型。

*   p(yhat = y)= p(yhat = 0)* p(y = 0)+p(yhat = 1)* p(y = 1)
*   P(yhat = y) = 0.25 * 0.25 + 0.75 * 0.75
*   P(yhat = y) = 0.0625 + 0.5625
*   P(yhat = y) = 0.625

结果表明，使用从训练数据集中均匀随机选择的类作为预测比简单地在该数据集中预测均匀随机类产生更好的朴素分类器，显示 62.5%而不是 50%，或者 12.2%的提升。

还不错！

让我们用一个小模拟再次确认我们的计算。

下面的 *random_class()* 函数通过从训练数据集中选择并返回一个随机的类标签来实现这个朴素的分类器策略。

```py
# predict a randomly selected class
def random_class(y):
	return y[randint(len(y))]
```

然后，我们可以使用上一节中的相同框架来评估模型 1000 次，并报告这些试验的平均分类准确率。我们希望这个经验估计值与我们的期望值相匹配，或者非常接近。

下面列出了完整的示例。

```py
# example of selecting a random class naive classifier
from numpy import mean
from numpy.random import randint
from sklearn.metrics import accuracy_score

# predict a randomly selected class
def random_class(y):
	return y[randint(len(y))]

# define dataset
class0 = [0 for _ in range(25)]
class1 = [1 for _ in range(75)]
y = class0 + class1
# average over many repeats
results = list()
for _ in range(1000):
	yhat = [random_class(y) for _ in range(len(y))]
	acc = accuracy_score(y, yhat)
	results.append(acc)
print('Mean: %.3f' % mean(results))
```

运行该示例执行了我们实验的 1000 次试验，并报告了策略的平均准确性。

鉴于算法的随机性，您的具体结果会有所不同。

在这种情况下，我们可以看到预期表现再次与计算表现非常接近:模拟中为 62.4%，而我们上面计算的是 62.5%。

```py
Mean: 0.624
```

在预测类别标签时，也许我们可以做得比均匀分布更好。我们将在下一节探讨这一点。

## 预测多数阶级

在前一节中，我们探索了一种策略，该策略基于训练数据集中观察到的标签的均匀概率分布来选择类别标签。

这使得预测的概率分布与观察到的每个类的概率分布相匹配，并且比类标签的均匀分布有所改进。尤其是这种不平衡数据集的一个缺点是，一个类在更大程度上被期望高于另一个类，并且随机预测类，甚至以有偏见的方式，导致太多不正确的预测。

相反，我们可以预测多数类，并确保达到至少与训练数据集中多数类的组成一样高的准确率。

也就是说，如果训练集中 75%的示例是类 1，并且我们为所有示例预测了类 1，那么我们知道我们至少会达到 75%的准确率，这比我们在上一节中随机选择的类有所提高。

我们可以通过使用我们的概率模型计算该方法的预期表现来证实这一点。

这种天真的分类策略预测类别 0 的概率是 0.0(不可能)，预测类别 1 的概率是 1.0(确定)。因此:

*   p(yhat = y)= p(yhat = 0)* p(y = 0)+p(yhat = 1)* p(y = 1)
*   P(yhat = y) = 0.0 * 0.25 + 1.0 * 0.75
*   P(yhat = y) = 0.0 + 0.75
*   P(yhat = y) = 0.75

这证实了我们的预期，并表明在此特定数据集上，该策略将比以前的策略进一步提升 12.5%。

同样，我们可以通过模拟来证实这种方法。

多数类可以使用模式进行统计计算；也就是分布中最常见的观察。

可以使用[模式()SciPy 功能](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html)。它返回两个值，第一个是我们可以返回的模式。下面的*more _ class()*函数实现了这个朴素的分类器。

```py
# predict the majority class
def majority_class(y):
	return mode(y)[0]
```

然后，我们可以在设计好的数据集上评估策略。我们不需要重复多次实验，因为策略中没有随机成分，算法每次在相同的数据集上会给出相同的表现。

下面列出了完整的示例。

```py
# example of a majority class naive classifier
from scipy.stats import mode
from sklearn.metrics import accuracy_score

# predict the majority class
def majority_class(y):
	return mode(y)[0]

# define dataset
class0 = [0 for _ in range(25)]
class1 = [1 for _ in range(75)]
y = class0 + class1
# make predictions
yhat = [majority_class(y) for _ in range(len(y))]
# calculate accuracy
accuracy = accuracy_score(y, yhat)
print('Accuracy: %.3f' % accuracy)
```

运行该示例会报告数据集上多数类朴素分类器的准确性。

准确性与 75%的概率框架计算的期望值和训练数据集的组成相匹配。

```py
Accuracy: 0.750
```

这个多数类朴素分类器是应该用来计算分类预测建模问题的基线表现的方法。

它同样适用于具有相同数量的类别标签的数据集，以及具有两个以上类别标签的问题，例如多类别分类问题。

现在我们已经发现了表现最好的朴素分类器模型，我们可以看到如何在下一个项目中使用它。

## Sklearn 中的朴素分类器

Sklearn 机器学习库提供了多数类朴素分类算法的实现，您可以在下一个分类预测建模项目中使用它。

它是作为[dummy 分类器类](https://Sklearn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)的一部分提供的。

要使用朴素分类器，必须定义类，并将“*策略*”参数设置为“*最频繁*，以确保预测多数类。然后，该类可以适合训练数据集，并用于对测试数据集或其他重采样模型评估策略进行预测。

```py
...
# define model
model = DummyClassifier(strategy='most_frequent')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
```

事实上，dummy 分类器是灵活的，允许使用另外两个朴素的分类器。

具体来说，将“*策略*”设置为“*制服*”将执行我们首先测试的随机猜测策略，将“*策略*”设置为“*分层*”将执行我们其次测试的随机选择类策略。

*   **随机猜测**:将“*战略*论证设置为“*制服*”。
*   **选择随机类**:将“*策略*参数设置为“*分层*”。
*   **多数类**:将“*战略*论证设置为“*最 _ 频繁*”。

通过在我们设计的数据集上进行测试，我们可以确认 DummyClassifier 在多数类朴素分类策略下的表现符合预期。

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

运行该示例准备数据集，然后使用多数类策略在数据集上定义并拟合 DummyClassifier。

评估来自模型的预测的分类准确性证实了模型的表现如预期的那样，获得了 75%的分数。

```py
Accuracy: 0.750
```

这个例子提供了一个起点，用于计算将来您自己的分类预测建模项目的朴素分类器基线表现。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [不平衡分类问题中的机会水平准确率是多少？](https://stats.stackexchange.com/questions/148149/what-is-the-chance-level-accuracy-in-unbalanced-classification-problems)
*   [不要使用随机猜测作为基线分类器](https://machinelearningmastery.com/dont-use-random-guessing-as-your-baseline-classifier/)
*   [硬化. dummy . dummy class ification API](https://Sklearn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)

## 摘要

在本教程中，您发现了如何为机器学习开发和评估幼稚的分类策略。

具体来说，您了解到:

*   朴素分类模型的表现提供了一个基线，通过这个基线，所有其他模型都可以被认为是熟练的或不熟练的。
*   多数类分类器获得了比其他朴素分类器模型更好的准确性，例如随机猜测和预测随机选择的观察类标签。
*   朴素分类器策略可以通过 Sklearn 库中的 DummyClassifier 类用于预测建模项目。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。