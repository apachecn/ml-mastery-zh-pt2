# 机器学习的信息增益和互信息

> 原文：<https://machinelearningmastery.com/information-gain-and-mutual-information/>

最后更新于 2020 年 12 月 10 日

信息增益计算通过某种方式转换数据集所带来的熵减少或惊喜。

它通常用于从训练数据集中构建决策树，方法是评估每个变量的信息增益，并选择最大化信息增益的变量，从而最小化熵，并最好地将数据集分成组以进行有效分类。

通过在目标变量的上下文中评估每个变量的增益，信息增益也可以用于特征选择。在这种略有不同的用法中，计算被称为两个随机变量之间的相互信息。

在这篇文章中，你将发现机器学习中的信息增益和互信息。

看完这篇文章，你会知道:

*   信息增益是通过转换数据集来减少熵或惊喜，通常用于训练决策树。
*   通过比较变换前后数据集的熵来计算信息增益。
*   互信息计算两个变量之间的统计相关性，是应用于变量选择时信息增益的名称。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2019 年 11 月更新**:改进了信息/熵基础的描述(感谢 HR)。
*   **更新 2020 年 8 月**:方程增加了漏括号(感谢大卫)

![What is Information Gain and Mutual Information for Machine Learning](img/cc37df371122d83a2cc04415a9bf33a1.png)

什么是机器学习的信息增益和互信息
摄影:朱塞佩·米洛，版权所有。

## 概观

本教程分为五个部分；它们是:

1.  什么是信息增益？
2.  计算信息增益的实例
3.  机器学习中的信息获取示例
4.  什么是相互信息？
5.  信息获取和相互信息是如何相关的？

## 什么是信息增益？

信息增益(简称 IG)通过根据随机变量的给定值分割数据集来衡量熵的减少或惊喜。

较大的信息增益意味着较低的熵组或样本组，因此不那么令人惊讶。

你可能还记得**信息**用比特来量化一个事件有多令人惊讶。低概率事件信息多，高概率事件信息少。**熵**量化随机变量中有多少信息，或者更具体地说，它的概率分布。偏斜分布的熵较低，而事件概率相等的分布熵较大。

在信息论中，我们喜欢描述一个事件的“*惊喜*”。低概率事件更令人惊讶，因此具有更大的信息量。而事件具有相同可能性的概率分布更令人惊讶，并且具有更大的熵。

*   **偏斜概率分布** ( *不出所料*):低熵。
*   **均衡概率分布** ( *惊人*):高熵。

有关信息和熵的基础知识的更多信息，请参见教程:

*   [信息熵的温和介绍](https://machinelearningmastery.com/what-is-information-entropy/)

现在，让我们考虑数据集的熵。

我们可以根据属于一个或另一个类别的数据集中的观测值的概率分布来考虑数据集的熵，例如在二进制类别数据集的情况下是两个类别。

> 信息论对熵的一种解释是，它规定了对 S 的任意成员(即以均匀概率随机抽取的 S 成员)的分类进行编码所需的最小信息位数。

—第 58 页，[机器学习](https://amzn.to/2jWd51p)，1997。

例如，在二分类问题(两类)中，我们可以计算数据样本的熵，如下所示:

*   熵= -(p(0) * log(P(0)) + p(1) * log(P(1)))

对于两个类，样本分割为 50/50 的数据集的最大熵(最大惊喜)为 1 位，而分割为 10/90 的不平衡数据集的熵较小，因为从数据集中随机抽取的示例的惊喜较小。

我们可以用 Python 中计算这个不平衡数据集熵的例子来演示这一点。下面列出了完整的示例。

```py
# calculate the entropy for a dataset
from math import log2
# proportion of examples in each class
class0 = 10/100
class1 = 90/100
# calculate entropy
entropy = -(class0 * log2(class0) + class1 * log2(class1))
# print the result
print('entropy: %.3f bits' % entropy)
```

运行该示例，我们可以看到用于二进制分类的数据集的熵小于 1 位。也就是说，对数据集中任意示例的类标签进行编码所需的信息不到一位。

```py
entropy: 0.469 bits
```

以这种方式，熵可以用作数据集纯度的计算，例如，类的分布碰巧有多平衡。

0 位的熵表示数据集包含一个类；1 位或更多位的熵表示平衡数据集的最大熵(取决于类的数量)，介于两者之间的值表示这两个极端之间的级别。

信息增益提供了一种使用熵来计算数据集变化如何影响数据集纯度的方法，例如类的分布。较小的熵意味着更高的纯度或更少的惊喜。

> ……信息增益，简单来说就是根据这个属性对示例进行划分所导致的熵的预期减少。

—第 57 页，[机器学习](https://amzn.to/2jWd51p)，1997。

例如，我们可能希望通过将数据集 *S* 拆分为一个具有一系列值的随机变量来评估对纯度的影响。

这可以计算如下:

*   IG(S，a)= H(S)–H(S | a)

其中 *IG(S，a)* 是数据集的信息 *S* 是随机变量 a 的信息， *H(S)* 是数据集在任何变化之前的熵(如上所述) *H(S | a)* 是给定变量 *a* 的数据集的条件熵。

该计算描述了变量 a 在数据集 *S* 中的增益，是转换数据集时保存的位数。

条件熵的计算方法是，针对 a 的每个观测值将数据集分成多个组，并计算整个数据集的每个组中的示例比率之和乘以每个组的熵。

*   H(S | a)= a Sa(v)中的和 v/S * H(Sa(v))

其中 *Sa(v)/S* 是数据集中样本数与变量 a 取值 *v* 的比值， *H(Sa(v))* 是变量 a 取值 *v* 的样本组熵。

这听起来可能有点混乱。

我们可以用一个实例来具体说明信息增益的计算。

## 计算信息增益的实例

在本节中，我们将通过一个实例来具体说明信息增益的计算。

我们可以定义一个函数，根据属于类别 0 和类别 1 的样本的比率来计算一组样本的熵。

```py
# calculate the entropy for the split in the dataset
def entropy(class0, class1):
	return -(class0 * log2(class0) + class1 * log2(class1))
```

现在，考虑一个包含 20 个示例的数据集，13 个用于类 0，7 个用于类 1。我们可以计算这个数据集的熵，它将少于 1 位。

```py
...
# split of the main dataset
class0 = 13 / 20
class1 = 7 / 20
# calculate entropy before the change
s_entropy = entropy(class0, class1)
print('Dataset Entropy: %.3f bits' % s_entropy)
```

现在考虑数据集中的一个变量有两个唯一的值，比如“*值 1* ”和“*值 2* ”我们感兴趣的是计算这个变量的信息增益。

让我们假设，如果我们按值 1 分割数据集，我们有一组八个样本，七个用于类 0，一个用于类 1。然后我们可以计算这组样本的熵。

```py
...
# split 1 (split via value1)
s1_class0 = 7 / 8
s1_class1 = 1 / 8
# calculate the entropy of the first group
s1_entropy = entropy(s1_class0, s1_class1)
print('Group1 Entropy: %.3f bits' % s1_entropy)
```

现在，让我们假设我们按值 2 分割数据集；我们有一组 12 个样本，每组 6 个。我们希望这个组的熵为 1。

```py
...
# split 2  (split via value2)
s2_class0 = 6 / 12
s2_class1 = 6 / 12
# calculate the entropy of the second group
s2_entropy = entropy(s2_class0, s2_class1)
print('Group2 Entropy: %.3f bits' % s2_entropy)
```

最后，我们可以根据为变量的每个值创建的组和计算出的熵来计算这个变量的信息增益。

第一个变量产生了一组来自数据集的八个示例，第二组具有数据集中剩余的 12 个样本。因此，我们拥有计算信息增益所需的一切。

在这种情况下，信息增益可以计算为:

*   熵(数据集)–(计数(组 1) /计数(数据集)*熵(组 1) +计数(组 2) /计数(数据集)*熵(组 2))

或者:

*   熵(13/20，7/20)–(8/20 *熵(7/8，1/8) + 12/20 *熵(6/12，6/12))

或者用代码:

```py
...
# calculate the information gain
gain = s_entropy - (8/20 * s1_entropy + 12/20 * s2_entropy)
print('Information Gain: %.3f bits' % gain)
```

将这些结合在一起，完整的示例如下所示。

```py
# calculate the information gain
from math import log2

# calculate the entropy for the split in the dataset
def entropy(class0, class1):
	return -(class0 * log2(class0) + class1 * log2(class1))

# split of the main dataset
class0 = 13 / 20
class1 = 7 / 20
# calculate entropy before the change
s_entropy = entropy(class0, class1)
print('Dataset Entropy: %.3f bits' % s_entropy)

# split 1 (split via value1)
s1_class0 = 7 / 8
s1_class1 = 1 / 8
# calculate the entropy of the first group
s1_entropy = entropy(s1_class0, s1_class1)
print('Group1 Entropy: %.3f bits' % s1_entropy)

# split 2  (split via value2)
s2_class0 = 6 / 12
s2_class1 = 6 / 12
# calculate the entropy of the second group
s2_entropy = entropy(s2_class0, s2_class1)
print('Group2 Entropy: %.3f bits' % s2_entropy)

# calculate the information gain
gain = s_entropy - (8/20 * s1_entropy + 12/20 * s2_entropy)
print('Information Gain: %.3f bits' % gain)
```

首先，数据集的熵计算在略低于 1 位。然后，第一和第二组的熵分别在大约 0.5 和 1 比特处计算。

最后，变量的信息增益计算为 0.117 位。也就是说，通过所选变量拆分数据集获得的增益为 0.117 位。

```py
Dataset Entropy: 0.934 bits
Group1 Entropy: 0.544 bits
Group2 Entropy: 1.000 bits
Information Gain: 0.117 bits
```

## 机器学习中的信息获取示例

也许在机器学习中，信息增益最常用的是在[决策树](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)中。

一个例子是[迭代二分器 3 算法](https://en.wikipedia.org/wiki/ID3_algorithm)，简称 ID3，用于构建决策树。

> 信息增益正是 ID3 在树生长的每一步选择最佳属性时使用的度量。

—第 58 页，[机器学习](https://amzn.to/2jWd51p)，1997。

为数据集中的每个变量计算信息增益。选择具有最大信息增益的变量来分割数据集。通常，较大的增益表示较小的熵或较小的惊喜。

> 请注意，最小化熵相当于最大化信息增益…

—第 547 页，[机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。

然后对每个创建的组重复该过程，排除已经选择的变量。一旦到达所需的决策树深度或不可能再进行拆分，此操作就会停止。

> 现在为每个非终端后代节点重复选择新属性和划分训练示例的过程，这次仅使用与该节点相关联的训练示例。排除了树中较高的属性，因此任何给定的属性在树中的任何路径上最多只能出现一次。

—第 60 页，[机器学习](https://amzn.to/2jWd51p)，1997。

在决策树的大多数现代实现中，信息增益可以用作拆分标准，例如在用于分类的[决策树分类器类](https://Sklearn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)中的 Sklearn Python 机器学习库中实现分类和回归树(CART)算法。

这可以通过在配置模型时将标准参数设置为“*熵*”来实现；例如:

```py
# example of a decision tree trained with information gain
from sklearn.tree import DecisionTreeClassifier
model = sklearn.tree.DecisionTreeClassifier(criterion='entropy')
...
```

信息增益也可用于建模前的特征选择。

它包括计算目标变量和训练数据集中每个输入变量之间的信息增益。 [Weka 机器学习工作台](https://waikato.github.io/weka-wiki/)通过[信息增益属性评估](http://weka.sourceforge.net/doc.dev/weka/attributeSelection/InfoGainAttributeEval.html)类为特征选择提供信息增益的实现。

在这个特征选择的上下文中，信息增益可以被称为“*互信息*”，并计算两个变量之间的统计相关性。使用信息增益(互信息)进行特征选择的一个例子是[互信息类函数](https://Sklearn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)。

## 什么是相互信息？

[互信息](https://en.wikipedia.org/wiki/Mutual_information)是在两个变量之间计算出来的，测量一个变量给定另一个变量的已知值时不确定性的减少。

> 一个叫做互信息的量衡量一个人从一个随机变量中获得的信息量。

—第 310 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2lnW5S7)，2016 年第 4 版。

两个随机变量 *X* 和 *Y* 之间的相互信息可以正式表述如下:

*   一(X；Y)= H(X)–H(X | Y)

其中*I(X；Y)* 是 *X* 和 *Y* 的互信息， *H(X)* 是给定 *Y* 的 *X* 和 *H(X | Y)* 的条件熵。结果有位的单位。

互信息是对两个随机变量之间的依赖或“T0”相互依赖的度量。因此，度量是对称的，意味着*我(X；Y)= I(Y；X)* 。

> 它衡量学习 y 的值所导致的 x 的不确定性的平均减少；反之亦然，x 传递的关于 y 的平均信息量。

—第 139 页，[信息论、推理和学习算法](https://amzn.to/2KfDDF7)，2003。

Kullback-Leibler，或 KL，散度是计算两个概率分布之间差异的度量。

互信息也可以计算为联合概率分布和每个变量的边缘概率乘积之间的 KL 散度。

> 如果变量不是独立的，我们可以通过考虑联合分布和边际的乘积之间的 kulback-Leibler 散度来获得关于它们是否“接近”独立的一些想法，这被称为变量之间的互信息

—第 57 页，[模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。

这可以正式表述如下:

*   一(X；Y) = KL(p(X，Y) || p(X) * p(Y))

互信息总是大于或等于零，其中值越大，两个变量之间的关系越大。如果计算结果为零，那么变量是独立的。

互信息通常用作相关系数的一般形式，例如随机变量之间相关性的度量。

它也被用作一些机器学习算法的一个方面。一个常见的例子是[独立成分分析](https://en.wikipedia.org/wiki/Independent_component_analysis)，简称独立成分分析，它提供数据集统计独立成分的投影。

## 信息获取和相互信息是如何关联的？

互信息和信息增益是同一件事，尽管措施的上下文或用法通常会产生不同的名称。

例如:

*   转换到数据集(*决策树*)的效果:信息增益。
*   变量之间的依赖(*特征选择*):互信息。

注意计算互信息的方式和计算信息增益的方式的相似性；它们是等价的:

*   一(X；Y)= H(X)–H(X | Y)

和

*   IG(S，a)= H(S)–H(S | a)

因此，相互信息有时被用作信息增益的同义词。从技术上讲，如果应用于相同的数据，它们计算的数量是相同的。

我们可以把两者之间的关系理解为联合概率分布和边缘概率分布(互信息)的差异越大，信息的增益(信息增益)越大。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [信息论、推理和学习算法](https://amzn.to/2KfDDF7)，2003。
*   [机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。
*   [模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。
*   [机器学习](https://amzn.to/2jWd51p)，1997。
*   [数据挖掘:实用机器学习工具与技术](https://amzn.to/2lnW5S7)，第 4 版，2016。

### 应用程序接口

*   [scipy.stats.entropy API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)

### 文章

*   [熵(信息论)，维基百科](https://en.wikipedia.org/wiki/Entropy_(information_theory))。
*   [决策树中的信息增益，维基百科](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)。
*   [ID3 算法，维基百科](https://en.wikipedia.org/wiki/ID3_algorithm)。
*   [信息增益比，维基百科](https://en.wikipedia.org/wiki/Information_gain_ratio)。
*   [互信息，维基百科](https://en.wikipedia.org/wiki/Mutual_information)。

## 摘要

在这篇文章中，你发现了机器学习中的信息增益和互信息。

具体来说，您了解到:

*   信息增益是通过转换数据集来减少熵或惊喜，通常用于训练决策树。
*   通过比较变换前后数据集的熵来计算信息增益。
*   互信息计算两个变量之间的统计相关性，是应用于变量选择时信息增益的名称。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。