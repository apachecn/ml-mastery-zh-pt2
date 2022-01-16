# 机器学习的 Fbeta 测量的温和介绍

> 原文：<https://machinelearningmastery.com/fbeta-measure-for-machine-learning/>

Fbeta-measure 是一个可配置的单一评分指标，用于基于对正类的预测来评估二进制分类模型。

Fbeta 度量是使用准确率和召回率计算的。

**准确率**是计算正类正确预测百分比的指标。**回忆**计算所有可能做出的积极预测中积极类的正确预测的百分比。最大化准确率将最小化假阳性错误，而最大化召回将最小化假阴性错误。

**F-measure** 被计算为精确度和召回率的调和平均值，给予每个相同的权重。它允许使用单个分数同时考虑精确度和召回率来评估模型，这在描述模型的表现和比较模型时很有帮助。

**Fbeta 测度**是 F 测度的推广，增加了一个配置参数β。默认β值为 1.0，与 F-测度相同。较小的β值(如 0.5)在计算分数时更重视准确率，而较少考虑召回，而较大的β值(如 2.0)在计算分数时更重视准确率，而更多考虑召回。

当精确度和召回率都很重要，但其中一个需要稍微多注意时，例如当假阴性比假阳性更重要时，或者相反，这是一个有用的度量标准。

在本教程中，您将发现用于评估机器学习分类算法的 Fbeta-measure。

完成本教程后，您将知道:

*   在二分类问题中，准确率和召回率提供了两种方法来总结正类的错误。
*   F-measure 提供了一个总结精确度和召回率的单一分数。
*   Fbeta-measure 提供了一个可配置的 F-measure 版本，在计算单个分数时或多或少会关注精确度和召回率。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![A Gentle Introduction to the Fbeta-Measure for Machine Learning](img/71228c1bd2ece24151b3db17c5b3bdf7.png)

机器学习的 Fbeta-Measure 的温和介绍
照片由[马尔科·韦奇](https://flickr.com/photos/30478819@N08/34564940376/)拍摄，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  精确度和召回率
    1.  混淆矩阵
    2.  精确
    3.  回忆
2.  衡量
    1.  最坏情况
    2.  最佳案例
    3.  50%准确率，完美召回
3.  测量
    1.  F1-测量
    2.  F0.5 测量
    3.  F2 测量

## 精确度和召回率

在我们深入研究 Fbeta-measure 之前，我们必须回顾用于评估分类模型所做预测的准确率和召回度量的基础。

### 混淆矩阵

一个[混淆矩阵](https://machinelearningmastery.com/ufaqs/what-is-a-confusion-matrix/)总结了一个模型对每个类所做的预测的数量，以及这些预测实际属于的类。它有助于理解模型产生的预测误差的类型。

最简单的混淆矩阵是针对两类分类问题，有负(0 类)和正(1 类)类。

在这种类型的混淆矩阵中，表中的每个单元格都有一个具体且易于理解的名称，总结如下:

```py
               | Positive Prediction | Negative Prediction
Positive Class | True Positive (TP)  | False Negative (FN)
Negative Class | False Positive (FP) | True Negative (TN)
```

准确率和召回度量是根据混淆矩阵中的单元格来定义的，特别是像真阳性和假阴性这样的术语。

### 精确

准确率是一个度量标准，它量化了正确的积极预测的数量。

它的计算方法是正确预测的正例数除以预测的正例总数。

*   准确率=真阳性/(真阳性+假阳性)

结果是一个介于 0.0(无准确率)和 1.0(完全或完美准确率)之间的值。

准确率的直觉是它不关心假阴性，它**最小化假阳性**。我们可以用下面的一个小例子来演示这一点。

```py
# intuition for precision
from sklearn.metrics import precision_score
# no precision
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
score = precision_score(y_true, y_pred)
print('No Precision: %.3f' % score)
# some false positives
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
score = precision_score(y_true, y_pred)
print('Some False Positives: %.3f' % score)
# some false negatives
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
score = precision_score(y_true, y_pred)
print('Some False Negatives: %.3f' % score)
# perfect precision
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
score = precision_score(y_true, y_pred)
print('Perfect Precision: %.3f' % score)
```

运行该示例演示计算所有不正确和所有正确的预测类标签的准确率，分别显示无准确率和完美准确率。

一个预测一些假阳性的例子显示了精确度的下降，强调了该度量与最小化假阳性有关。

一个预测一些假阴性的例子显示了完美的准确率，强调了该措施与假阴性无关。

```py
No Precision: 0.000
Some False Positives: 0.714
Some False Negatives: 1.000
Perfect Precision: 1.000
```

### 回忆

回忆是一个度量标准，它量化了从所有可能做出的积极预测中做出的正确积极预测的数量。

计算方法是正确预测的正例数除以可预测的正例总数。

*   回忆=真阳性/(真阳性+假阴性)

结果是 0.0(无召回)到 1.0(完全或完美召回)之间的值。

回忆的直觉是它不关心假阳性，它**最小化假阴性**。我们可以用下面的一个小例子来演示这一点。

```py
# intuition for recall
from sklearn.metrics import recall_score
# no recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
score = recall_score(y_true, y_pred)
print('No Recall: %.3f' % score)
# some false positives
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
score = recall_score(y_true, y_pred)
print('Some False Positives: %.3f' % score)
# some false negatives
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
score = recall_score(y_true, y_pred)
print('Some False Negatives: %.3f' % score)
# perfect recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
score = recall_score(y_true, y_pred)
print('Perfect Recall: %.3f' % score)
```

运行该示例演示了计算所有不正确和所有正确的预测类标签的召回，分别显示无召回和完美召回。

一个预测一些假阳性的例子显示了完美的回忆，强调了该措施与假阳性无关。

一个预测一些假阴性的例子显示了召回率的下降，强调了该措施与最小化假阴性有关。

```py
No Recall: 0.000
Some False Positives: 1.000
Some False Negatives: 0.600
Perfect Recall: 1.000
```

现在我们已经熟悉了准确率和召回率，让我们回顾一下 F-measure。

## 衡量

精确度和召回率衡量正类可能出现的两种错误。

最大化准确率可以最小化误报，最大化召回可以最小化误报。

F-Measure 或 F-Score 提供了一种将准确率和召回率结合到一个度量中的方法，该度量可以捕获这两个属性。

*   F-Measure = (2 *准确率*召回)/(准确率+召回)

这是两个分数的[调和平均值](https://en.wikipedia.org/wiki/Harmonic_mean)。

结果是一个介于 0.0 和 1.0 之间的值。

F-measure 的直觉是，这两个度量在重要性上是平衡的，只有好的准确率和好的召回率才能产生好的 F-measure。

### 最坏情况

第一，如果所有的例子都被完美地错误地预测，我们将具有零准确率和零召回，导致零 F 测度；例如:

```py
# worst case f-measure
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# no precision or recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = f1_score(y_true, y_pred)
print('No Precision or Recall: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
```

运行该示例，我们可以看到没有准确率或召回率导致最坏情况下的 F 度量。

```py
No Precision or Recall: p=0.000, r=0.000, f=0.000
```

假设准确率和召回率只与正类相关，我们可以通过预测所有示例的负类来实现相同的最坏情况准确率、召回率和 F-measure:

```py
# another worst case f-measure
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# no precision and recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = f1_score(y_true, y_pred)
print('No Precision or Recall: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
```

鉴于没有预测到阳性病例，我们必须输出零准确率和召回率，进而输出 f 测度。

```py
No Precision or Recall: p=0.000, r=0.000, f=0.000
```

### 最佳案例

相反，完美的预测将产生完美的精确度和召回率，进而产生完美的 F 值，例如:

```py
# best case f-measure
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# perfect precision and recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = f1_score(y_true, y_pred)
print('Perfect Precision and Recall: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
```

运行该示例，我们可以看到完美的准确率和召回率导致了完美的 F 度量。

```py
Perfect Precision and Recall: p=1.000, r=1.000, f=1.000
```

### 50%准确率，完美召回

不可能有完美的精确和不召回，或者没有精确和完美的召回。精确度和召回率都需要预测真正的阳性。

考虑我们预测所有情况的正类的情况。

这将给我们 50%的准确率，因为一半的预测是假阳性。它会给我们完美的回忆，因为我们不会有假阴性。

对于我们在示例中使用的平衡数据集，一半的预测是真阳性，一半是假阳性；因此，精确率将为 0.5%或 50%。将 50%的感知准确率和完美召回率结合起来，将得到一个惩罚的 F-测度，具体来说就是 50%到 100%之间的调和平均值。

下面的例子演示了这一点。

```py
# perfect precision f-measure
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# perfect precision, 50% recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = f1_score(y_true, y_pred)
print('Result: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
```

运行该示例证实，我们确实具有 50 的感知准确率和完美的召回率，并且 F 分数的结果值约为 0.667。

```py
Result: p=0.500, r=1.000, f=0.667
```

## 测量

f 检验平衡了精确度和召回率。

在某些问题上，我们可能会对更注重准确率的 F-测度感兴趣，例如当误报对于最小化更重要，但漏报仍然重要时。

在其他问题上，我们可能会对一个更注重回忆的 F-测度感兴趣，比如什么时候假阴性对最小化更重要，但假阳性仍然重要。

解决办法是 Fbeta-措施。

Fbeta 测度是 F 测度的抽象，其中[谐波均值](https://en.wikipedia.org/wiki/Harmonic_mean)计算中的准确率和召回率的平衡由一个称为*β*的系数控制。

*   Fbeta = ((1 + beta^2) *准确率*召回)/ (beta^2 *准确率+召回)

β参数的选择将用于 Fbeta 测量的名称中。

例如，β值 2 被称为 f 2 度量或 F2 分数。β值 1 被称为 f 1 度量或 F1 分数。

beta 参数的三个常见值如下:

*   **f 0.5-测量**(β= 0.5):准确率权重更大，召回权重更小。
*   **F1-测量**(β= 1.0):在准确率和召回率上平衡权重。
*   **F2-测量**(β= 2.0):准确率权重较小，召回权重较大

起初，不同β值对计算的影响并不直观。

让我们仔细看看这些案例。

### F1-测量

上一节中讨论的 F-测度是一个*β*值为 1 的 Fbeta 测度的例子。

具体来说，F-measure 和 F1-measure 计算的是同一件事；例如:

*   F-Measure = ((1 + 1^2) *准确率*召回)/ (1^2 *准确率+召回)
*   F-Measure = (2 *准确率*召回)/(准确率+召回)

考虑我们有 50 感知准确率和完美回忆的情况。对于这种情况，我们可以手动计算 F1 度量，如下所示:

*   F-Measure = (2 *准确率*召回)/(准确率+召回)
*   f-测量= (2 * 0.5 * 1.0) / (0.5 + 1.0)
*   f-测量= 1.0 / 1.5
*   f-测量= 0.666

我们可以使用 Sklearn 中的 [fbeta_score()函数](https://Sklearn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)来确认这个计算，将“ *beta* 参数设置为 1.0。

下面列出了完整的示例。

```py
# calculate the f1-measure
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# perfect precision, 50% recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = fbeta_score(y_true, y_pred, beta=1.0)
print('Result: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
```

运行该示例证实了完美的准确率和 50%的召回率以及 0.667 的 F1 度量，证实了我们的计算(带舍入)。

0.667 的 F1 度量值与上一节中为相同场景计算的 F 度量值相匹配。

```py
Result: p=0.500, r=1.000, f=0.667
```

### f 0.5-测量

F0.5 测量值是 Fbeta 测量值的一个示例，其*β*值为 0.5。

它具有提高精确度的重要性和降低召回率的重要性的效果。

如果最大化准确率可以最小化假阳性，最大化召回可以最小化假阴性，那么**f 0.5-度量将更多的注意力放在最小化假阳性上**而不是最小化假阴性上。

f 0.5-测量值计算如下:

*   f 0.5-测量= ((1 + 0.5^2) *准确率*召回)/ (0.5^2 *准确率+召回)
*   f 0.5-测量= (1.25 *准确率*召回)/ (0.25 *准确率+召回)

假设我们有 50%的准确率和完美的召回率。对于这种情况，我们可以手动计算 F0.5 测量值，如下所示:

*   f 0.5-测量= (1.25 *准确率*召回)/ (0.25 *准确率+召回)
*   f 0.5-测量= (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0)
*   f 0.5-测量值= 0.625 / 1.125
*   f 0.5-测量值= 0.555

我们预计β值为 0.5 会导致该场景的得分较低，因为准确率得分较低，召回率很高。

这正是我们所看到的，在 F1 分数计算为 0.667 的相同情况下，F0.5 的测量值为 0.555。准确率在计算中发挥了更大的作用。

我们可以证实这个计算；下面列出了完整的示例。

```py
# calculate the f0.5-measure
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# perfect precision, 50% recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = fbeta_score(y_true, y_pred, beta=0.5)
print('Result: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
```

运行该示例确认准确率和召回值，然后报告 0.556(带舍入)的 F0.5 度量，与我们手动计算的值相同。

```py
Result: p=0.500, r=1.000, f=0.556
```

### F2-测量

F2 测量值是 Fbeta 测量值的一个示例，其*β*值为 2.0。

它具有降低精确度的重要性和增加召回率的重要性的效果。

如果最大化准确率可以最小化假阳性，最大化召回可以最小化假阴性，那么 **F2-measure 比最小化假阳性更注重最小化假阴性**。

F2 测量值计算如下:

*   F2-Measure = ((1 + 2^2) *准确率*召回)/ (2^2 *准确率+召回)
*   F2-测量= (5 *准确率*召回)/ (4 *准确率+召回)

假设我们有 50%的准确率和完美的召回率。

对于这种情况，我们可以手动计算 F2 度量，如下所示:

*   F2-测量= (5 *准确率*召回)/ (4 *准确率+召回)
*   F2-测量= (5 * 0.5 * 1.0) / (4 * 0.5 + 1.0)
*   F2-测量= 2.5 / 3.0
*   F2-测量值= 0.833

我们预计 2.0 的 *beta* 值将导致该场景的更高得分，因为召回具有满分，这将比准确率表现差的情况得到提升。

这正是我们所看到的，对于 F1 分数计算为 0.667 的相同场景，F2 测量值为 0.833。回忆在计算中起了更大的作用。

我们可以证实这个计算；下面列出了完整的示例。

```py
# calculate the f2-measure
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# perfect precision, 50% recall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = fbeta_score(y_true, y_pred, beta=2.0)
print('Result: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
```

运行该示例确认准确率和召回值，然后报告 0.883 的 F2 度量值，与我们手动计算的值相同(带舍入)。

```py
Result: p=0.500, r=1.000, f=0.833
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [不平衡分类评估指标之旅](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)
*   [如何计算不平衡分类的准确率、召回率和 F-Measure](https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/)

### 报纸

*   [F-measure 的真相](https://www.toyota-ti.ac.jp/Lab/Denshi/COIN/people/yutaka.sasaki/F-measure-YS-26Oct07.pdf)，2007。

### 蜜蜂

*   硬化. metrics.f1_score API 。
*   [硬化. metrics.fbeta_score API](https://Sklearn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html) 。

### 文章

*   f1 得分，维基百科。
*   [调和的意思，维基百科](https://en.wikipedia.org/wiki/Harmonic_mean)。

## 摘要

在本教程中，您发现了用于评估机器学习分类算法的 Fbeta 度量。

具体来说，您了解到:

*   在二分类问题中，准确率和召回率提供了两种方法来总结正类的错误。
*   F-measure 提供了一个总结精确度和召回率的单一分数。
*   Fbeta-measure 提供了一个可配置的 F-measure 版本，在计算单个分数时或多或少会关注精确度和召回率。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。