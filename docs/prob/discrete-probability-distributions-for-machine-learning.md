# 机器学习的离散概率分布

> 原文：<https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/>

最后更新于 2020 年 10 月 6 日

离散随机变量的概率可以用离散[概率分布](https://machinelearningmastery.com/what-are-probability-distributions)来概括。

离散概率分布用于机器学习，最显著的是用于二进制和多类分类问题的建模，但也用于评估二进制分类模型的表现，例如置信区间的计算，以及用于[自然语言处理](https://machinelearningmastery.com/natural-language-processing/)的文本中单词分布的建模。

在选择用于分类任务的深度学习神经网络的输出层中的激活函数和选择适当的损失函数时，也需要离散概率分布的知识。

离散概率分布在应用机器学习中起着重要的作用，从业者必须了解一些分布。

在本教程中，您将发现机器学习中使用的离散概率分布。

完成本教程后，您将知道:

*   离散随机变量的结果概率可以用离散概率分布来概括。
*   单个二元结果具有伯努利分布，一系列二元结果具有二项式分布。
*   单个分类结果具有多项分布，一系列分类结果具有多项分布。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 10 月更新**:修正了二项式分布描述中的错别字。

![Discrete Probability Distributions for Machine Learning](img/477137cfcd7a1ec33d3bdeeb37271874.png)

机器学习的离散概率分布
约翰·福勒摄，版权所有。

## 教程概述

本教程分为五个部分；它们是:

1.  离散概率分布
2.  二项分布
3.  二项分布
4.  多欧拉分布
5.  多项分布

## 离散概率分布

随机变量是随机过程产生的量。

离散随机变量是一个随机变量，它可以有一组有限的特定结果之一。机器学习中最常用的两种离散随机变量是二元和分类变量。

*   **二进制随机变量** : x 在{0，1}中
*   **分类随机变量** : x 在{1，2，…，K}中。

二进制随机变量是一个离散的随机变量，其中有限的结果集在{0，1}中。分类随机变量是一个离散的随机变量，其中有限的结果集在{1，2，…，K}中，其中 *K* 是唯一结果的总数。

离散随机变量的每个结果或事件都有一个概率。

离散随机变量的事件和它们的概率之间的关系被称为离散概率分布，由概率质量函数(简称 PMF)来概括。

对于可排序的结果，事件等于或小于给定值的概率由累积分布函数(简称 CDF)定义。CDF 的倒数称为百分点函数，它将给出小于或等于概率的离散结果。

*   **PMF** :概率质量函数，返回给定结果的概率。
*   **CDF** :累积分布函数，返回小于等于给定结果的概率值。
*   **PPF** :百分点函数，返回小于等于给定概率的离散值。

有许多常见的离散概率分布。

最常见的是分别针对二元和分类离散随机变量的伯努利分布和多项式分布，以及将每个分布推广到多个独立试验的二项式分布和多项式分布。

*   **二元随机变量**:伯努利分布
*   **二元随机变量的序列**:二项式分布
*   **分类随机变量**:多元线性分布
*   **分类随机变量序列**:多项式分布

在接下来的几节中，我们将依次仔细研究这些分布。

您可能还想探索其他离散概率分布，包括泊松分布和离散均匀分布。

## 二项分布

[伯努利分布](https://en.wikipedia.org/wiki/Bernoulli_distribution)是一个离散的概率分布，它涵盖了一个事件具有二进制结果为 0 或 1 的情况。

*   {0，1}中的 x

“[伯努利试验](https://en.wikipedia.org/wiki/Bernoulli_trial)”是一个实验或案例，其结果遵循伯努利分布。该分布和试验以瑞士数学家[雅各布·伯努利](https://en.wikipedia.org/wiki/Jacob_Bernoulli)的名字命名。

伯努利试验的一些常见例子包括:

*   硬币的单次翻转，可能有正面(0)或反面(1)的结果。
*   男孩(0)或女孩(1)的单胎。

机器学习中伯努利试验的一个常见示例可能是将单个示例二进制分类为第一类(0)或第二类(1)。

这个分布可以用一个单一的变量 *p* 来概括，这个变量定义了结果 1 的概率。给定此参数，每个事件的概率可以计算如下:

*   P(x=1) = p
*   p(x = 0)= 1–p

在抛公平硬币的情况下， *p* 的值将是 0.5，给出每个结果的 50%概率。

## 二项分布

重复多次独立的伯努利试验被称为[伯努利过程](https://en.wikipedia.org/wiki/Bernoulli_process)。

伯努利过程的结果将遵循二项分布。因此，伯努利分布是一个二项分布，只有一次试验。

伯努利过程的一些常见例子包括:

*   一系列独立的硬币翻转。
*   一系列独立的出生。

机器学习算法在二进制分类问题上的表现可以被分析为伯努利过程，其中模型对来自测试集的示例的预测是伯努利试验(正确或不正确)。

二项式分布总结了在给定数量的伯努利试验 *k* 中的成功次数，每个试验都有给定的成功概率 *p* 。

我们可以用伯努利过程来证明这一点，其中成功的概率为 30%或 P(x=1) = 0.3，试验总数为 100 (k=100)。

我们可以用随机生成的案例模拟伯努利过程，并计算给定试验次数的成功次数。这可以通过[二项式()NumPy 功能](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.binomial.html)来实现。该函数将试验总数和成功概率作为参数，并返回一次模拟试验中成功结果的数量。

```py
# example of simulating a binomial process and counting success
from numpy.random import binomial
# define the parameters of the distribution
p = 0.3
k = 100
# run a single simulation
success = binomial(k, p)
print('Total Success: %d' % success)
```

我们期望在给定所选参数(k * p 或 100 * 0.3)的情况下，100 个案例中有 30 个会成功。

每次运行代码时都会产生 100 次试验的不同随机序列，因此您的具体结果会有所不同。试着运行这个例子几次。

在这种情况下，我们可以看到我们获得的成功试验略少于预期的 30 个。

```py
Total Success: 28
```

我们可以使用 [binom.stats() SciPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)计算这个分布的矩，特别是期望值或平均值和方差。

```py
# calculate moments of a binomial distribution
from scipy.stats import binom
# define the parameters of the distribution
p = 0.3
k = 100
# calculate moments
mean, var, _, _ = binom.stats(k, p, moments='mvsk')
print('Mean=%.3f, Variance=%.3f' % (mean, var))
```

运行该示例报告了分布的期望值，正如我们所期望的，是 30，以及方差 21，如果我们计算平方根，得到的标准偏差约为 4.5。

```py
Mean=30.000, Variance=21.000
```

我们可以使用概率质量函数来计算一系列试验的不同成功结果数的可能性，例如 10、20、30 到 100。

我们预计 30 个成功结果的概率最高。

```py
# example of using the pmf for the binomial distribution
from scipy.stats import binom
# define the parameters of the distribution
p = 0.3
k = 100
# define the distribution
dist = binom(k, p)
# calculate the probability of n successes
for n in range(10, 110, 10):
	print('P of %d success: %.3f%%' % (n, dist.pmf(n)*100))
```

运行该示例定义了二项式分布，并计算了[10，100]中每组 10 个成功结果的概率。

概率乘以 100 给出百分比，我们可以看到 30 个成功结果的概率最高，约为 8.6%。

```py
P of 10 success: 0.000%
P of 20 success: 0.758%
P of 30 success: 8.678%
P of 40 success: 0.849%
P of 50 success: 0.001%
P of 60 success: 0.000%
P of 70 success: 0.000%
P of 80 success: 0.000%
P of 90 success: 0.000%
P of 100 success: 0.000%
```

鉴于一项试验的成功概率为 30%，我们预计 100 项试验中 50 项或更少成功的概率接近 100%。我们可以用累积分布函数来计算，如下所示。

```py
# example of using the cdf for the binomial distribution
from scipy.stats import binom
# define the parameters of the distribution
p = 0.3
k = 100
# define the distribution
dist = binom(k, p)
# calculate the probability of <=n successes
for n in range(10, 110, 10):
	print('P of %d success: %.3f%%' % (n, dist.cdf(n)*100))
```

运行该示例会以 10 为一组打印[10，100]中的每一个成功次数，以及在 100 次试验中达到该成功次数或更少的概率。

不出所料，在 50 次或更少的成功之后，涵盖了该分布中 99.999%的预期成功。

```py
P of 10 success: 0.000%
P of 20 success: 1.646%
P of 30 success: 54.912%
P of 40 success: 98.750%
P of 50 success: 99.999%
P of 60 success: 100.000%
P of 70 success: 100.000%
P of 80 success: 100.000%
P of 90 success: 100.000%
P of 100 success: 100.000%
```

## 多欧拉分布

多项式分布，也称为[分类分布](https://en.wikipedia.org/wiki/Categorical_distribution)，涵盖了一个事件具有 K 种可能结果之一的情况。

*   {1，2，3，…，K}中的 x

它是伯努利分布从二元变量到分类变量的推广，其中伯努利分布的例数 *K* 设置为 2， *K=2* 。

遵循多项式分布的一个常见例子是:

*   一卷骰子的结果为{1，2，3，4，5，6}，例如 K=6。

机器学习中多类分布的一个常见示例可能是将单个示例多类分类为 *K* 类之一，例如鸢尾花的三个不同种类之一。

分布可以用从 *p1* 到 *pK* 的 *K* 变量来概括，每个变量定义从 1 到 *K* 的给定分类结果的概率，其中所有概率的总和为 1.0。

*   P(x=1) = p1
*   P(x=2) = p1
*   P(x=3) = p3
*   …
*   P(x=K) = pK

在单卷模具的情况下，每个值的概率将是 1/6，或大约 0.166 或大约 16.6%。

## 多项分布

多个独立的多项试验的重复将遵循多项式分布。

多项式分布是具有 *K* 结果的离散变量的二项式分布的推广。

多项式过程的一个例子包括一系列独立的骰子滚动。

多项式分布的一个常见例子是自然语言处理领域的文本文档中单词的出现计数。

多项式分布由具有 *K* 结果的离散随机变量、从 *p1* 到 *pK* 的每个结果的概率和 *k* 连续试验来概括。

我们可以用一个小例子来证明这一点，3 个类别( *K=3* )具有相等的概率( *p=33.33%* )和 100 个试验。

首先，我们可以使用[多项式()NumPy 函数](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html)来模拟 100 个独立试验，并总结事件导致每个给定类别的次数。该函数将每个类别的试验次数和概率作为一个列表。

下面列出了完整的示例。

```py
# example of simulating a multinomial process
from numpy.random import multinomial
# define the parameters of the distribution
p = [1.0/3.0, 1.0/3.0, 1.0/3.0]
k = 100
# run a single simulation
cases = multinomial(k, p)
# summarize cases
for i in range(len(cases)):
	print('Case %d: %d' % (i+1, cases[i]))
```

我们预计每个类别大约有 33 个事件。

运行该示例会报告每个案例和事件的数量。

每次运行代码时都会产生 100 次试验的不同随机序列，因此您的具体结果会有所不同。试着运行这个例子几次。

在这种情况下，我们看到病例的分布高达 37 例，低至 30 例。

```py
Case 1: 37
Case 2: 33
Case 3: 30
```

我们可能期望 100 个试验的理想化情况分别导致事件 1、2 和 3 的 33、33 和 34 个病例。

我们可以使用概率质量函数或[多项式 pmf() SciPy 函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html)来计算这种特定组合在实践中发生的概率。

下面列出了完整的示例。

```py
# calculate the probability for a given number of events of each type
from scipy.stats import multinomial
# define the parameters of the distribution
p = [1.0/3.0, 1.0/3.0, 1.0/3.0]
k = 100
# define the distribution
dist = multinomial(k, p)
# define a specific number of outcomes from 100 trials
cases = [33, 33, 34]
# calculate the probability for the case
pr = dist.pmf(cases)
# print as a percentage
print('Case=%s, Probability: %.3f%%' % (cases, pr*100))
```

运行该示例报告了每个事件类型的理想化案例数[33，33，34]小于 1%的概率。

```py
Case=[33, 33, 34], Probability: 0.813%
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   第二章:概率分布，[模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。
*   第 3.9 节:常见概率分布，[深度学习](https://amzn.to/2lnc3vL)，2016。
*   第 2.3 节:一些常见的离散分布，[机器学习:概率观点](https://amzn.to/2xKSTCP)，2012。

### 应用程序接口

*   [离散统计分布，SciPy](https://docs.scipy.org/doc/scipy/reference/tutorial/stats/discrete.html) 。
*   [scipy.stats.bernoulli API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html) 的缩写形式。
*   [scipy . stat . binom API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)的缩写形式。
*   [scipy.stats .多项式 API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html) 。

### 文章

*   [伯努利分布，维基百科](https://en.wikipedia.org/wiki/Bernoulli_distribution)。
*   [伯努利过程，维基百科](https://en.wikipedia.org/wiki/Bernoulli_process)。
*   [伯努利试验，维基百科](https://en.wikipedia.org/wiki/Bernoulli_trial)。
*   [二项分布，维基百科](https://en.wikipedia.org/wiki/Binomial_distribution)。
*   [分类分布，维基百科](https://en.wikipedia.org/wiki/Categorical_distribution)。
*   [多项式分布，维基百科](https://en.wikipedia.org/wiki/Multinomial_distribution)。

## 摘要

在本教程中，您发现了机器学习中使用的离散概率分布。

具体来说，您了解到:

*   离散随机变量的结果概率可以用离散概率分布来概括。
*   单个二元结果具有伯努利分布，一系列二元结果具有二项式分布。
*   单个分类结果具有多项分布，一系列分类结果具有多项分布。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。