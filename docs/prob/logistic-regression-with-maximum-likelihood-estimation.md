# 使用最大似然估计的逻辑回归入门

> 原文：<https://machinelearningmastery.com/logistic-regression-with-maximum-likelihood-estimation/>

最后更新于 2019 年 10 月 28 日

逻辑回归是用于二分类预测建模的模型。

逻辑回归模型的参数可以通过称为[最大似然估计](https://machinelearningmastery.com/what-is-maximum-likelihood-estimation-in-machine-learning/)的概率框架来估计。在这个框架下，必须假设目标变量(类标签)的概率分布，然后定义一个似然函数，计算给定输入数据和模型时观察结果的概率。然后可以优化该函数，以找到在训练数据集上产生最大和似然的参数集。

拟合逻辑回归模型的最大似然方法既有助于更好地理解逻辑回归模型的形式，又提供了一个可用于更一般地拟合分类模型的模板。这是特别正确的，因为在该过程中使用的对数似然函数的负值可以被显示为等同于交叉熵损失函数。

在这篇文章中，你将发现最大似然估计的逻辑回归。

看完这篇文章，你会知道:

*   逻辑回归是二分类预测建模的线性模型。
*   模型的线性部分预测属于类别 1 的实例的对数赔率，通过逻辑函数将其转换为概率。
*   模型的参数可以通过最大化似然函数来估计，该似然函数预测每个例子的伯努利分布的平均值。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![A Gentle Introduction to Logistic Regression With Maximum Likelihood Estimation](img/188bdd45d5b468d3d1667cdfb7cf9a7e.png)

带最大似然估计的逻辑回归的温和介绍
摄影:塞缪尔·约翰，版权所有。

## 概观

本教程分为四个部分；它们是:

1.  逻辑回归
2.  逻辑回归和对数优势
3.  最大似然估计
4.  作为最大似然的逻辑回归

## 逻辑回归

逻辑回归是二分类的经典线性方法。

分类预测建模问题是那些需要为给定的一组输入变量预测一个类标签的问题(例如“*红色*”、“*绿色*”、“*蓝色*”)。二分类是指那些有两个类别标签的分类问题，例如真/假或 0/1。

逻辑回归与线性回归有很多共同之处，尽管线性回归是一种预测数值的技术，而不是用于分类问题。这两种技术都用一条线(或超平面，取决于输入的维数)来建模目标变量。线性回归将直线拟合到数据，数据可用于预测新的数量，而逻辑回归拟合直线以最好地将两个类别分开。

输入数据用 n 个例子表示为 *X* ，输出用 *y* 表示，每个输入一个输出。给定输入的模型预测表示为 *yhat* 。

*   yhat =型号(X)

该模型是根据称为系数(*β*)的参数定义的，其中每个输入有一个系数，还有一个提供截距或偏差的附加系数。

例如，具有 m 个变量 *x1，x2，…，xm* 的输入 *X* 的问题将具有系数*β1，β2，…，βm*和*β0*。给定的输入被预测为示例输入和系数的加权和。

*   yhat = beta 0+beta 1 * x1+beta 2 * x2++betam * XM

该模型也可以使用线性代数来描述，其中系数向量(*β*)和输入数据矩阵( *X* )以及输出向量( *y* )。

*   y = X *贝塔

到目前为止，这与线性回归相同，并且是不够的，因为输出将是实值，而不是类标签。

相反，该模型使用非线性函数来压缩这个加权和的输出，以确保输出是 0 到 1 之间的值。

使用逻辑函数(也称为 sigmoid)，其定义为:

*   f(x) = 1 / (1 + exp(-x))

其中 x 是函数的输入值。在逻辑回归的情况下，x 用加权和代替。

例如:

*   yhat = 1/(1+exp((x * beta)))

如果问题中的两个类别被标记为 0 和 1，则输出被解释为来自标记为 1 的类别的二项式概率分布函数的概率。

> 请注意，输出是 0 到 1 之间的数字，可以解释为属于标记为 1 的类的概率。

—第 726 页，[人工智能:现代方法](https://amzn.to/2Y7yCpO)，第 3 版，2009 年。

训练数据集中的示例来自更广泛的人群，因此，已知该样本是不完整的。此外，预计在观测中会有测量误差或统计噪声。

模型的参数(*β*)必须根据从域中提取的观测样本来估计。

估计参数的方法有很多。最常见的有两种框架；它们是:

*   最小二乘优化([迭代重加权最小二乘](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares))。
*   [最大似然估计](https://machinelearningmastery.com/what-is-maximum-likelihood-estimation-in-machine-learning/)。

两者都是涉及搜索不同模型参数的优化过程。

最大似然估计是一个频繁概率框架，它为模型寻找一组参数，使似然函数最大化。我们将在接下来的章节中仔细研究第二种方法。

## 逻辑回归和对数优势

在我们深入研究如何从数据中估计模型的参数之前，我们需要了解逻辑回归到底在计算什么。

这可能是逻辑回归中最令人困惑的部分，所以我们将慢慢来看。

模型的线性部分(输入的加权和)计算成功事件的对数赔率，特别是样本属于类别 1 的对数赔率。

*   对数赔率=β0+β1 * x1+β2 * x2+…+βm * XM

实际上，该模型估计了每一级输入变量(所有观测值)的 1 类对数赔率。

什么是赔率和对数赔率？

赌博领域的赔率可能很熟悉。赔率通常表示为赢与输(赢:输)，例如，1 比 10 的胜率或胜率表示为 1 : 10。

给定逻辑回归模型预测的成功概率( *p* ，我们可以将其转换为[成功几率](https://en.wikipedia.org/wiki/Odds)，即成功概率除以不成功概率:

*   成功几率= p/(1–p)

几率的对数是计算出来的，特别是对数基数-e 或自然对数。这个量称为对数优势，也可以称为 logit(逻辑单位)，一种度量单位。

*   对数优势=对数(p/(1–p)

回想一下，这就是逻辑回归的线性部分所计算的:

*   对数赔率=β0+β1 * x1+β2 * x2+…+βm * XM

通过计算对数赔率的指数，成功的对数赔率可以转换回成功的赔率。

*   赔率= exp(对数赔率)

或者

*   赔率= exp(beta 0+beta 1 * x1+beta 2 * x2+…+beta m * XM)

成功的几率可以转换回成功的概率，如下所示:

*   p =赔率/(赔率+ 1)

这接近于我们的逻辑回归模型的形式，除了我们想要将对数优势转换为优势作为计算的一部分。

我们可以这样做，并将计算简化如下:

*   p = 1/(1+exp(-对数优势))

这显示了我们如何从对数赔率到赔率，用逻辑回归模型到 1 类概率，并且这个最终函数形式匹配逻辑函数，确保概率在 0 和 1 之间。

我们可以用 Python 中的一些小例子来具体计算概率、赔率和对数赔率之间的转换。

首先，让我们将成功的概率定义为 80%，即 0.8，并将其转换为赔率，然后再次转换回概率。

下面列出了完整的示例。

```py
# example of converting between probability and odds
from math import log
from math import exp
# define our probability of success
prob = 0.8
print('Probability %.1f' % prob)
# convert probability to odds
odds = prob / (1 - prob)
print('Odds %.1f' % odds)
# convert back to probability
prob = odds / (odds + 1)
print('Probability %.1f' % prob)
```

运行该示例显示，0.8 被转换为成功概率 4，并再次回到正确的概率。

```py
Probability 0.8
Odds 4.0
Probability 0.8
```

让我们扩展这个例子，将赔率转换为对数赔率，然后将对数赔率转换回原始概率。这种最终转换实际上是逻辑回归模型或逻辑函数的形式。

下面列出了完整的示例。

```py
# example of converting between probability and log-odds
from math import log
from math import exp
# define our probability of success
prob = 0.8
print('Probability %.1f' % prob)
# convert probability to odds
odds = prob / (1 - prob)
print('Odds %.1f' % odds)
# convert odds to log-odds
logodds = log(odds)
print('Log-Odds %.1f' % logodds)
# convert log-odds to a probability
prob = 1 / (1 + exp(-logodds))
print('Probability %.1f' % prob)
```

运行该示例，我们可以看到我们的赔率被转换为大约 1.4 的对数赔率，然后被正确地转换回 0.8 的成功概率。

```py
Probability 0.8
Odds 4.0
Log-Odds 1.4
Probability 0.8
```

现在我们已经掌握了逻辑回归计算的概率，让我们看看最大似然估计。

## 最大似然估计

[最大似然估计](https://machinelearningmastery.com/what-is-maximum-likelihood-estimation-in-machine-learning/)，简称 MLE，是估计模型参数的概率框架。

在最大似然估计中，我们希望在给定特定概率分布及其参数(*θ*)的情况下，最大化观察数据的条件概率( *X* )，正式表述为:

*   p(X；θ)

其中 *X* 实际上是从问题域 1 到 *n* 的所有观测值的联合概率分布。

*   P(x1，x2，x3，…，xn；θ)

这个结果条件概率被称为观察给定模型参数的数据的可能性，并使用符号 *L()* 来表示[可能性函数](https://en.wikipedia.org/wiki/Likelihood_function)。例如:

*   l(X；θ)

给定分布参数，联合概率分布可以重申为观察每个例子的条件概率的乘积。将许多小概率相乘可能不稳定；因此，通常将这个问题重述为对数条件概率的总和。

*   对数和(Xi；θ))

鉴于对数在似然函数中的频繁使用，它被称为对数似然函数。在优化问题中，通常倾向于最小化成本函数，而不是最大化成本函数。因此，使用对数似然函数的负值，通常称为负对数似然函数。

*   最小化对数和(Xi；θ))

最大似然估计框架可以用作估计回归和分类预测建模的许多不同机器学习模型的参数的基础。这包括逻辑回归模型。

## 作为最大似然的逻辑回归

我们可以将拟合机器学习模型的问题框架为概率密度估计问题。

具体来说，模型和模型参数的选择被称为建模假设 *h* ，问题涉及到寻找最能解释数据 *X* 的 *h* 。因此，我们可以找到最大化似然函数的建模假设。

*   最大化对数和(Xi；h))

监督学习可以被构造成一个条件概率问题，在给定输入的情况下预测输出的概率:

*   P(y | X)

因此，我们可以为监督机器学习定义条件最大似然估计如下:

*   最大化对数和(Xi；h))

现在我们可以用我们的逻辑回归模型来代替 *h* 。

为了使用最大似然，我们需要假设一个概率分布。在逻辑回归的情况下，假设数据样本为二项式概率分布，其中每个例子都是伯努利试验的一个结果。伯努利分布只有一个参数:成功结果的概率( *p* )。

*   P(y=1) = p
*   p(y = 0)= 1–p

> 当有两个类别时，最常用的概率分布是二项式分布。5 这个分布有一个参数 p，即事件或特定类别的概率。

—第 283 页，[应用预测建模](https://amzn.to/2yXgeBT)，2013 年。

伯努利分布的期望值(平均值)可以计算如下:

*   平均值= P(y=1) * 1 + P(y=0) * 0

或者，给定 p:

*   平均值= p * 1+(1–p)* 0

这种计算看似多余，但它为特定输入的似然函数提供了基础，其中概率由模型( *yhat* )给出，实际标签由数据集给出。

*   可能性= yhat * y+(1–yhat)*(1–y)

对于 *y=0* 和 *y=1* 两种情况，该函数总是在模型接近匹配类值时返回大概率，在模型较远时返回小概率。

我们可以用一个小的例子来证明这一点，这个例子既可以预测结果，也可以预测每个结果的大小概率。

下面列出了完整的示例。

```py
# test of Bernoulli likelihood function

# likelihood function for Bernoulli distribution
def likelihood(y, yhat):
	return yhat * y + (1 - yhat) * (1 - y)

# test for y=1
y, yhat = 1, 0.9
print('y=%.1f, yhat=%.1f, likelihood: %.3f' % (y, yhat, likelihood(y, yhat)))
y, yhat = 1, 0.1
print('y=%.1f, yhat=%.1f, likelihood: %.3f' % (y, yhat, likelihood(y, yhat)))
# test for y=0
y, yhat = 0, 0.1
print('y=%.1f, yhat=%.1f, likelihood: %.3f' % (y, yhat, likelihood(y, yhat)))
y, yhat = 0, 0.9
print('y=%.1f, yhat=%.1f, likelihood: %.3f' % (y, yhat, likelihood(y, yhat)))
```

运行该示例将打印类标签( *y* )和预测概率( *yhat* )用于具有每个案例的远近概率的案例。

我们可以看到，似然函数在返回模型达到预期结果的概率方面是一致的。

```py
y=1.0, yhat=0.9, likelihood: 0.900
y=1.0, yhat=0.1, likelihood: 0.100
y=0.0, yhat=0.1, likelihood: 0.900
y=0.0, yhat=0.9, likelihood: 0.100
```

我们可以使用日志更新似然函数，将其转换为对数似然函数:

*   对数似然=对数(yhat) * y +对数(1–yhat)*(1–y)

最后，我们可以对数据集中所有示例的似然函数求和，以最大化似然性:

*   最大化 I 与 n 之和 log(yhat _ I)* y _ I+log(1–yhat _ I)*(1–y _ I)

通常的做法是最小化优化问题的成本函数；因此，我们可以反转函数，使负对数似然最小化:

*   将总和 I 最小化为 n-(log(yhat _ I)* y _ I+log(1–yhat _ I)*(1–y _ I))

计算伯努利分布的对数似然函数的负值相当于计算伯努利分布的[交叉熵](https://en.wikipedia.org/wiki/Cross_entropy)函数，其中 *p()* 代表 0 类或 1 类的概率， *q()* 代表概率分布的估计，在这种情况下通过我们的逻辑回归模型。

*   交叉熵=-(log(q(class 0))* p(class 0)+log(q(class 1))* p(class 1))

与线性回归不同，没有解决这个优化问题的解析解。因此，必须使用迭代优化算法。

> 与线性回归不同，我们不能再以封闭形式记下最大似然估计。相反，我们需要使用优化算法来计算它。为此，我们需要导出梯度和黑森。

—第 246 页，[机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。

该函数确实提供了一些有助于优化的信息(具体来说，可以计算一个黑森矩阵)，这意味着可以使用利用这些信息的有效搜索过程，例如 [BFGS 算法](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)(及其变体)。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 邮件

*   [机器学习最大似然估计的温和介绍](https://machinelearningmastery.com/what-is-maximum-likelihood-estimation-in-machine-learning/)
*   [如何在 Python 中从零开始实现逻辑回归](https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/)
*   [机器学习逻辑回归教程](https://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/)
*   [机器学习的逻辑回归](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)

### 书

*   第 4.4.1 节拟合逻辑回归模型，[统计学习的要素](https://amzn.to/2YVqu8s)，2016。
*   第 4.3.2 节逻辑回归，[模式识别和机器学习](https://amzn.to/2JwHE7I)，2006。
*   第八章逻辑回归，[机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。
*   第 4 章算法:基本方法，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2lnW5S7)，第 4 版，2016。
*   第 18.6.4 节逻辑回归线性分类，[人工智能:现代方法](https://amzn.to/2Y7yCpO)，第 3 版，2009。
*   第 12.2 节逻辑回归，[应用预测建模](https://amzn.to/2yXgeBT)，2013。
*   第 4.3 节逻辑回归，[R](https://amzn.to/31DTbbO)中应用的统计学习导论，2017。

### 文章

*   [最大似然估计，维基百科](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)。
*   [似然函数，维基百科](https://en.wikipedia.org/wiki/Likelihood_function)。
*   [逻辑回归，维基百科](https://en.wikipedia.org/wiki/Logistic_regression)。
*   [逻辑功能，维基百科](https://en.wikipedia.org/wiki/Logistic_function)。
*   [Odds，维基百科](https://en.wikipedia.org/wiki/Odds)。

## 摘要

在这篇文章中，你发现了最大似然估计的逻辑回归。

具体来说，您了解到:

*   逻辑回归是二分类预测建模的线性模型。
*   模型的线性部分预测属于类别 1 的实例的对数赔率，通过逻辑函数将其转换为概率。
*   模型的参数可以通过最大化似然函数来估计，该似然函数预测每个例子的伯努利分布的平均值。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。