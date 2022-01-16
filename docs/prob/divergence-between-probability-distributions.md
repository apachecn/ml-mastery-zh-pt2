# 如何计算机器学习的 KL 散度

> 原文：<https://machinelearningmastery.com/divergence-between-probability-distributions/>

最后更新于 2019 年 11 月 1 日

通常希望量化给定随机变量的概率分布之间的差异。

这在机器学习中经常发生，当我们可能对计算实际和观察到的概率分布之间的差异感兴趣时。

这可以使用信息论中的技术来实现，例如库尔巴克-莱布勒散度(KL 散度)或相对熵，以及詹森-香农散度，它提供了 KL 散度的归一化和对称版本。这些评分方法可以作为计算其他广泛使用的方法的捷径，例如[互信息](https://machinelearningmastery.com/information-gain-and-mutual-information)用于建模前的特征选择，[交叉熵](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)用作许多不同分类器模型的损失函数。

在这篇文章中，你将发现如何计算概率分布之间的散度。

看完这篇文章，你会知道:

*   统计距离是计算统计对象之间差异的一般思想，例如随机变量的不同概率分布。
*   库尔巴克-莱布勒散度计算一个分数，用来衡量一个概率分布与另一个概率分布的散度。
*   詹森-香农散度扩展了 KL 散度来计算一个概率分布与另一个概率分布的对称得分和距离度量。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2019 年 10 月更新**:增加了方程式替代形式的描述(感谢 Ori)。

![How to Calculate the Distance Between Probability Distributions](img/d49858e8fd7ecdc2e856035e2ec491ec.png)

如何计算概率分布之间的距离
图片由[帕克森·沃尔伯](https://www.flickr.com/photos/paxson_woelber/9840645363/)提供，版权所有。

## 概观

本教程分为三个部分；它们是:

1.  统计距离
2.  库尔巴克-莱布勒散度
3.  詹森-香农散度

## 统计距离

在许多情况下，我们可能希望比较两种概率分布。

具体来说，我们可能有一个随机变量和该变量的两个不同的概率分布，例如真实分布和该分布的近似值。

在这种情况下，量化分布之间的差异可能是有用的。通常，这被称为计算两个统计对象之间的[统计距离](https://en.wikipedia.org/wiki/Statistical_distance)的问题，例如概率分布。

一种方法是计算两个分布之间的距离度量。这可能具有挑战性，因为很难解释这项措施。

相反，更常见的是计算两个概率分布之间的[散度](https://en.wikipedia.org/wiki/Divergence_(statistics))。发散就像一种度量，但不是对称的。这意味着散度是一个分布如何不同于另一个分布的评分，其中计算分布 P 和 Q 的散度将给出不同于 Q 和 P 的评分

发散分数是[信息论](https://machinelearningmastery.com/what-is-information-entropy/)以及更一般的机器学习中许多不同计算的重要基础。例如，它们提供了计算分数的快捷方式，如[互信息](https://machinelearningmastery.com/information-gain-and-mutual-information)(信息增益)和交叉熵，用作分类模型的损失函数。

发散分数也直接用作理解复杂建模问题的工具，例如在优化[生成对抗网络(GAN)模型](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)时近似目标概率分布。

信息论中两个常用的散度得分是库尔巴克-莱布勒散度和詹森-香农散度。

在下一节中，我们将仔细研究这两个分数。

## 库尔巴克-莱布勒散度

[库尔巴克-莱布勒散度](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)分数，或 KL 散度分数，量化了一个概率分布与另一个概率分布的不同程度。

两个分布 Q 和 P 之间的 KL 散度通常用下面的符号表示:

*   KL(P || Q)

其中“||”运算符表示“*散度*或与 q 的 Ps 散度。

KL 散度可以计算为 P 中每个事件概率的负和乘以 Q 中事件概率对 P 中事件概率的对数

*   KL(P | | Q)=–X 中 X 的和 P(x) * log(Q(x) / P(x))

总和中的值是给定事件的散度。

这与 P 中每个事件概率的正和乘以 P 中事件概率对 Q 中事件概率的对数相同(例如分数中的项被翻转)。这是实践中使用的更常见的实现。

*   KL(P | | Q)= X 中 X 的和 P(x) * log(P(x) / Q(x))

KL 散度得分的直觉是，当一个事件来自 P 的概率较大，而同一事件在 Q 中的概率较小时，存在较大的散度。当来自 P 的概率小，来自 Q 的概率大时，也有较大的发散，但没有第一种情况大。

它可以用来衡量离散和连续概率分布之间的差异，在后一种情况下，计算的是事件的积分，而不是离散事件的概率之和。

> 测量两个概率分布 p 和 q 的不相似性的一种方法被称为 kulback-Leibler 散度(KL 散度)或相对熵。

—第 57 页，[机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。

对数可以是以 2 为底的，以“*位*为单位，或者是以“*自然对数为底的，以*为单位当得分为 0 时，表明两个分布相同，否则得分为正。

重要的是，KL 散度分数是不对称的，例如:

*   KL(P || Q)！= KL(Q || P)

它以方法的两位作者所罗门·库尔巴克和理查德·莱布勒的名字命名，有时被称为“相对熵”

> 这被称为分布 p(x)和 q(x)之间的相对熵或库尔巴克-莱布勒散度，或 KL 散度。

—第 55 页，[模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。

如果我们试图近似一个未知的概率分布，那么来自数据的目标概率分布是 P，Q 是我们对该分布的近似。

在这种情况下，KL 散度总结了从随机变量表示事件所需的额外位数(即，用以 2 为底的对数计算)。我们的近似越好，需要的额外信息就越少。

> ……KL 散度是对数据进行编码所需的额外位数的平均值，这是因为我们使用分布 q 对数据进行编码，而不是真正的分布 p。

—第 58 页，[机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。

我们可以用一个实例来具体说明 KL 散度。

考虑一个随机变量，它有三个不同颜色的事件。对于这个变量，我们可能有两种不同的概率分布；例如:

```py
...
# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
```

我们可以绘制这些概率的条形图，将它们直接作为概率直方图进行比较。

下面列出了完整的示例。

```py
# plot of distributions
from matplotlib import pyplot
# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
print('P=%.3f Q=%.3f' % (sum(p), sum(q)))
# plot first distribution
pyplot.subplot(2,1,1)
pyplot.bar(events, p)
# plot second distribution
pyplot.subplot(2,1,2)
pyplot.bar(events, q)
# show the plot
pyplot.show()
```

运行该示例会为每个概率分布创建一个直方图，从而可以直接比较每个事件的概率。

我们可以看到，分布确实是不同的。

![Histogram of Two Different Probability Distributions for the Same Random Variable](img/dd6774378e7d3997bc10b97d07075233.png)

同一随机变量两种不同概率分布的直方图

接下来，我们可以开发一个函数来计算两个分布之间的 KL 散度。

我们将使用 log base-2 来确保结果以位为单位。

```py
# calculate the kl divergence
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
```

然后我们可以用这个函数来计算 P 和 Q 的 KL 散度，以及反过来，P 和 Q 的 KL 散度

```py
# calculate (P || Q)
kl_pq = kl_divergence(p, q)
print('KL(P || Q): %.3f bits' % kl_pq)
# calculate (Q || P)
kl_qp = kl_divergence(q, p)
print('KL(Q || P): %.3f bits' % kl_qp)
```

将这些结合在一起，完整的示例如下所示。

```py
# example of calculating the kl divergence between two mass functions
from math import log2

# calculate the kl divergence
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# define distributions
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate (P || Q)
kl_pq = kl_divergence(p, q)
print('KL(P || Q): %.3f bits' % kl_pq)
# calculate (Q || P)
kl_qp = kl_divergence(q, p)
print('KL(Q || P): %.3f bits' % kl_qp)
```

运行该示例首先计算 P 与 Q 的散度，刚好小于 2 位，然后计算 Q 与 P 的散度，刚好大于 2 位。

如果我们考虑当 Q 很小时 P 具有大概率，这是直观的，这使得 P 比 Q 从 P 得到的发散更小，因为当 P 具有大概率时 Q 具有更多的小概率。在第二种情况下，分歧更大。

```py
KL(P || Q): 1.927 bits
KL(Q || P): 2.022 bits
```

如果我们将 *log2()* 改为自然对数 *log()* 函数，结果用 nats 表示，如下所示:

```py
# KL(P || Q): 1.336 nats
# KL(Q || P): 1.401 nats
```

SciPy 库提供了 [kl_div()函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html)来计算 kl 散度，尽管这里定义了不同的定义。它还提供了计算相对熵的 [rel_entr()函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html)，与这里 KL 散度的定义相匹配。这很奇怪，因为“*相对熵*经常被用作“ *KL 散度*的同义词。”

然而，我们可以使用 rel_entr() SciPy 函数计算 KL 散度，并确认我们的手动计算是正确的。

*rel_entr()* 函数将每个概率分布中所有事件的概率列表作为参数，并返回每个事件的离差列表。这些可以相加得到 KL 散度。计算使用自然对数而不是对数基数-2，因此单位是以 nats 而不是以位为单位。

下面列出了使用 SciPy 为上面使用的相同概率分布计算 KL(P || Q)和 KL(Q || P)的完整示例:

```py
# example of calculating the kl divergence (relative entropy) with scipy
from scipy.special import rel_entr
# define distributions
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate (P || Q)
kl_pq = rel_entr(p, q)
print('KL(P || Q): %.3f nats' % sum(kl_pq))
# calculate (Q || P)
kl_qp = rel_entr(q, p)
print('KL(Q || P): %.3f nats' % sum(kl_qp))
```

运行该示例，我们可以看到计算出的偏差与我们手动计算的 KL(P || Q)和 KL(Q || P)分别约为 1.3 nats 和 1.4 nats 相匹配。

```py
KL(P || Q): 1.336 nats
KL(Q || P): 1.401 nats
```

## 詹森-香农散度

詹森-香农散度，简称 JS 散度，是另一种量化两个概率分布之间差异(或相似性)的方法。

它使用 KL 散度来计算对称的归一化分数。这意味着 P 与 Q 的散度与 Q 与 P 的散度相同，或者正式表述为:

*   JS(P || Q) == JS(Q || P)

JS 散度可以计算如下:

*   JS(P | | Q)= 1/2 * KL(P | | M)+1/2 * KL(Q | | M)

其中 M 的计算公式为:

*   M = 1/2 * (P + Q)

并且 *KL()* 被计算为上一节描述的 KL 散度。

当使用以 2 为底的对数时，它作为一种度量更有用，因为它提供了 KL 散度的平滑和归一化版本，分数在 0(相同)和 1(最大差异)之间。

分数的平方根给出了一个量，称为詹森-香农距离，简称 JS 距离。

我们可以用一个工作实例来具体说明 JS 分歧。

首先，我们可以定义一个函数来计算 JS 散度，该函数使用上一节准备的*KL _ diffusion()*函数。

```py
# calculate the kl divergence
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# calculate the js divergence
def js_divergence(p, q):
	m = 0.5 * (p + q)
	return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
```

然后，我们可以使用上一节中使用的相同概率分布来测试这个函数。

首先，我们将计算分布的 JS 散度分数，然后计算分数的平方根，给出分布之间的 JS 距离。例如:

```py
...
# calculate JS(P || Q)
js_pq = js_divergence(p, q)
print('JS(P || Q) divergence: %.3f bits' % js_pq)
print('JS(P || Q) distance: %.3f' % sqrt(js_pq))
```

然后可以对相反的情况重复这一过程，以表明发散是对称的，与 KL 发散不同。

```py
...
# calculate JS(Q || P)
js_qp = js_divergence(q, p)
print('JS(Q || P) divergence: %.3f bits' % js_qp)
print('JS(Q || P) distance: %.3f' % sqrt(js_qp))
```

将这些联系在一起，下面列出了计算 JS 散度和 JS 距离的完整示例。

```py
# example of calculating the js divergence between two mass functions
from math import log2
from math import sqrt
from numpy import asarray

# calculate the kl divergence
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# calculate the js divergence
def js_divergence(p, q):
	m = 0.5 * (p + q)
	return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# define distributions
p = asarray([0.10, 0.40, 0.50])
q = asarray([0.80, 0.15, 0.05])
# calculate JS(P || Q)
js_pq = js_divergence(p, q)
print('JS(P || Q) divergence: %.3f bits' % js_pq)
print('JS(P || Q) distance: %.3f' % sqrt(js_pq))
# calculate JS(Q || P)
js_qp = js_divergence(q, p)
print('JS(Q || P) divergence: %.3f bits' % js_qp)
print('JS(Q || P) distance: %.3f' % sqrt(js_qp))
```

运行该示例表明，分布之间的 JS 散度约为 0.4 位，距离约为 0.6。

我们可以看到计算是对称的，给 JS(P || Q)和 JS(Q || P)相同的分数和距离度量。

```py
JS(P || Q) divergence: 0.420 bits
JS(P || Q) distance: 0.648
JS(Q || P) divergence: 0.420 bits
JS(Q || P) distance: 0.648
```

SciPy 库通过 [jensenshannon()函数](https://scipy.github.io/devdocs/generated/scipy.spatial.distance.jensenshannon.html)提供了 JS 距离的实现。

它将来自每个概率分布的所有事件的概率数组作为参数，并返回 JS 距离分数，而不是散度分数。我们可以使用这个函数来确认我们手动计算的 JS 距离。

下面列出了完整的示例。

```py
# calculate the jensen-shannon distance metric
from scipy.spatial.distance import jensenshannon
from numpy import asarray
# define distributions
p = asarray([0.10, 0.40, 0.50])
q = asarray([0.80, 0.15, 0.05])
# calculate JS(P || Q)
js_pq = jensenshannon(p, q, base=2)
print('JS(P || Q) Distance: %.3f' % js_pq)
# calculate JS(Q || P)
js_qp = jensenshannon(q, p, base=2)
print('JS(Q || P) Distance: %.3f' % js_qp)
```

运行该示例，我们可以确认距离分数与我们手动计算的 0.648 相匹配，并且距离计算如预期的那样对称。

```py
JS(P || Q) Distance: 0.648
JS(Q || P) Distance: 0.648
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。
*   [模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。

### 邮件

*   [训练深度学习神经网络时如何选择损失函数](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)
*   [用于训练深度学习神经网络的损失和损失函数](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/)

### 蜜蜂

*   [scipy.stats.entropy API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html) 。
*   [scipy.special.kl_div API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html) 。
*   [scipy . spatial . distance . jensenshannon API](https://scipy.github.io/devdocs/generated/scipy.spatial.distance.jensenshannon.html)。

### 文章

*   [统计距离，维基百科](https://en.wikipedia.org/wiki/Statistical_distance)。
*   [发散(统计)，维基百科](https://en.wikipedia.org/wiki/Divergence_(statistics))。
*   [kul LBA-leilbler 分歧，维基百科](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)。
*   Jensen-Shannon 分歧，维基百科。

## 摘要

在这篇文章中，你发现了如何计算概率分布之间的散度。

具体来说，您了解到:

*   统计距离是计算统计对象之间差异的一般思想，例如随机变量的不同概率分布。
*   库尔巴克-莱布勒散度计算一个分数，用来衡量一个概率分布与另一个概率分布的散度。
*   詹森-香农散度扩展了 KL 散度来计算一个概率分布与另一个概率分布的对称得分和距离度量。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。