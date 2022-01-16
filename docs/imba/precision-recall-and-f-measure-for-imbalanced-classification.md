# 如何计算不平衡分类的准确率、召回率和 F-Measure

> 原文：<https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/>

最后更新于 2020 年 8 月 2 日

分类准确率是正确预测的总数除以数据集的预测总数。

作为一种表现度量，准确性不适用于不平衡的分类问题。

主要原因是，来自多数类(或多个类)的压倒性数量的示例将压倒少数类中的示例数量，这意味着即使是不熟练的模型也可以达到 90%或 99%的准确性分数，这取决于类不平衡碰巧有多严重。

使用分类准确度的替代方法是使用精确度和召回度量。

在本教程中，您将发现如何计算和发展直觉，以获得不平衡分类的准确率和召回率。

完成本教程后，您将知道:

*   准确率量化了实际属于正类的正类预测的数量。
*   Recall 量化了从数据集中所有正面示例中做出的正面类预测的数量。
*   F-Measure 提供了一个单一的分数，在一个数字中平衡了对精确度和召回率的关注。

**用我的新书[Python 不平衡分类](https://machinelearningmastery.com/imbalanced-classification-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

*   **2020 年 1 月更新**:关于精确和召回目标的改进语言。修正了准确率和召回寻求最小化的错别字(谢谢评论！).
*   **2020 年 2 月更新**:修正了召回和 f1 的变量名错别字。

![How to Calculate Precision, Recall, and F-Measure for Imbalanced Classification](img/0a358ca9b00a01f82e8cc6072b662472.png)

如何计算不平衡分类的准确率、召回率和 F-Measure[瓦尔德马尔合并](https://flickr.com/photos/paxx/8161204857/)图片，保留部分权利。

## 教程概述

本教程分为五个部分；它们是:

1.  不平衡分类的混淆矩阵
2.  不平衡分类的准确率
3.  不平衡分类的召回
4.  不平衡分类的准确率与召回率
5.  不平衡分类的 f-测度

## 不平衡分类的混淆矩阵

在我们深入精确和回忆之前，回顾一下[混淆矩阵](https://machinelearningmastery.com/confusion-matrix-machine-learning/)是很重要的。

对于不平衡分类问题，多数类通常被称为阴性结果(例如“*无变化*”或“*阴性检测结果*”)，少数类通常被称为阳性结果(例如“变化”或“阳性检测结果”)。

混淆矩阵不仅能更深入地了解预测模型的表现，还能更深入地了解哪些类被正确预测，哪些被错误预测，以及出现了什么类型的错误。

最简单的混淆矩阵是针对两类分类问题，有负(0 类)和正(1 类)类。

在这种类型的混淆矩阵中，表中的每个单元格都有一个具体且易于理解的名称，总结如下:

```py
               | Positive Prediction | Negative Prediction
Positive Class | True Positive (TP)  | False Negative (FN)
Negative Class | False Positive (FP) | True Negative (TN)
```

准确率和召回度量是根据混淆矩阵中的单元格来定义的，特别是像真阳性和假阴性这样的术语。

现在我们已经复习了混淆矩阵，让我们更仔细地看看准确率度量。

## 不平衡分类的准确率

准确率是一个度量标准，它量化了正确的积极预测的数量。

因此，准确率计算少数民族的准确率。

它的计算方法是正确预测的正例数除以预测的正例总数。

> 准确率评估被分类为阳性的实例中正确分类实例的比例…

—第 52 页，[从不平衡数据集](https://amzn.to/307Xlva)中学习，2018。

### 二分类的准确率

在具有两个类别的不平衡分类问题中，准确率计算为真阳性数除以真阳性和假阳性总数。

*   准确率=真阳性/(真阳性+假阳性)

结果是一个介于 0.0(无准确率)和 1.0(完全或完美准确率)之间的值。

让我们用一些例子来具体说明这个计算。

考虑一个少数与多数比例为 1:100 的数据集，其中有 100 个少数类示例和 10，000 个多数类示例。

一个模型做出预测，预测 120 个属于少数民族的例子，其中 90 个是正确的，30 个是不正确的。

该模型的准确率计算如下:

*   准确率=真阳性/(真阳性+假阳性)
*   准确率= 90 / (90 + 30)
*   准确率= 90 / 120
*   准确率= 0.75

结果是准确率为 0.75，这是一个合理的值，但并不突出。

你可以看到，准确率仅仅是所有正面预测中正确正面预测的比率，或者少数民族预测的准确率。

考虑同一个数据集，其中一个模型预测了 50 个属于少数民族的例子，其中 45 个是真阳性，5 个是假阳性。我们可以按如下方式计算该模型的准确率:

*   准确率=真阳性/(真阳性+假阳性)
*   准确率= 45 / (45 + 5)
*   准确率= 45 / 50
*   准确率= 0.90

在这种情况下，尽管模型预测属于少数群体的例子少得多，但正确的正面例子的比例要好得多。

这突出表明，尽管精确是有用的，但它并不能说明全部情况。它没有评论有多少真正的正类例子被预测为属于负类，即所谓的假阴性。

### 多分类准确率

准确率不限于二分类问题。

在具有两个以上类别的不平衡分类问题中，准确率计算为所有类别的真阳性之和除以所有类别的真阳性和假阳性之和。

*   准确率= C 中的和 C 真阳性 _c/C 中的和 C(真阳性 _c +假阳性 _ C)

例如，我们可能有一个不平衡的多类分类问题，其中多数类是负类，但有两个正少数类:类 1 和类 2。准确率可以量化两个正类中正确预测的比率。

考虑一个少数与多数类比率为 1:1:100 的数据集，即每个正类的比率为 1:1，少数类与多数类的比率为 1:100，每个少数类中有 100 个示例，多数类中有 10，000 个示例。

一个模型预测了第一个少数民族阶层的 70 个例子，其中 50 个是正确的，20 个是不正确的。它预测第二类 150 分，正确 99 分，错误 51 分。该模型的准确率可以计算如下:

*   准确率=(真阳性 _1 +真阳性 _2) /((真阳性 _1 +真阳性 _2) +(假阳性 _1 +假阳性 _2))
*   准确率= (50 + 99) / ((50 + 99) + (20 + 51))
*   准确率= 149 / (149 + 71)
*   准确率= 149 / 220
*   准确率= 0.677

我们可以看到，随着少数民族类别数量的增加，准确率度量计算也在增加。

### 使用 Scikit 计算准确率-学习

可以使用[precision _ score()sci kit-learn 功能](https://Sklearn.org/stable/modules/generated/sklearn.metrics.precision_score.html)计算准确率分数。

例如，我们可以使用这个函数来计算上一节中场景的准确率。

首先，有 100 个阳性到 10，000 个阴性例子的情况，一个模型预测 90 个真阳性和 30 个假阳性。下面列出了完整的示例。

```py
# calculates precision for 1:100 dataset with 90 tp and 30 fp
from sklearn.metrics import precision_score
# define actual
act_pos = [1 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
# define predictions
pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
pred_neg = [1 for _ in range(30)] + [0 for _ in range(9970)]
y_pred = pred_pos + pred_neg
# calculate prediction
precision = precision_score(y_true, y_pred, average='binary')
print('Precision: %.3f' % precision)
```

运行该示例计算准确率，与我们的手动计算相匹配。

```py
Precision: 0.750
```

接下来，我们可以使用相同的函数以 1:1:100 计算多类问题的准确率，每个少数类有 100 个示例，多数类有 10，000 个示例。一个模型预测 1 类有 50 个真阳性和 20 个假阳性，2 类有 99 个真阳性和 51 个假阳性。

当使用 *precision_score()* 函数进行多类分类时，通过“*标签*参数指定少数类并执行将“*平均值*参数设置为“*微*以确保计算如我们预期的那样执行是很重要的。

下面列出了完整的示例。

```py
# calculates precision for 1:1:100 dataset with 50tp,20fp, 99tp,51fp
from sklearn.metrics import precision_score
# define actual
act_pos1 = [1 for _ in range(100)]
act_pos2 = [2 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos1 + act_pos2 + act_neg
# define predictions
pred_pos1 = [0 for _ in range(50)] + [1 for _ in range(50)]
pred_pos2 = [0 for _ in range(1)] + [2 for _ in range(99)]
pred_neg = [1 for _ in range(20)] + [2 for _ in range(51)] + [0 for _ in range(9929)]
y_pred = pred_pos1 + pred_pos2 + pred_neg
# calculate prediction
precision = precision_score(y_true, y_pred, labels=[1,2], average='micro')
print('Precision: %.3f' % precision)
```

同样，运行该示例会计算与我们的手动计算相匹配的多类示例的准确率。

```py
Precision: 0.677
```

## 不平衡分类的召回

回忆是一个度量标准，它量化了从所有可能做出的积极预测中做出的正确积极预测的数量。

与只对所有积极预测中正确的积极预测进行评论的精确度不同，回忆提供了错过积极预测的指示。

通过这种方式，回忆提供了积极类覆盖的一些概念。

> 对于不平衡学习，回忆通常用于衡量少数民族的覆盖范围。

—第 27 页，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

### 二进制分类的召回

在具有两个类别的不平衡分类问题中，召回率的计算方法是真阳性数除以真阳性和假阴性总数。

*   回忆=真阳性/(真阳性+假阴性)

结果是 0.0(无召回)到 1.0(完全或完美召回)之间的值。

让我们用一些例子来具体说明这个计算。

与上一节一样，考虑一个少数与多数比例为 1:100 的数据集，其中有 100 个少数类示例和 10，000 个多数类示例。

一个模型做出预测，并正确预测了 90 个正类预测，错误预测了 10 个。我们可以这样计算这个模型的召回率:

*   回忆=真阳性/(真阳性+假阴性)
*   回忆= 90 / (90 + 10)
*   回忆= 90 / 100
*   回忆= 0.9

这款车型召回率很高。

### 多类分类的召回

召回不仅限于二分类问题。

在有两个以上类别的不平衡分类问题中，召回率的计算方法是所有类别的真阳性之和除以所有类别的真阳性和假阴性之和。

*   回忆= C 中的和 C 真实阳性 _c/C 中的和 C(真实阳性 _c +假阴性 _ C)

与上一节一样，考虑少数与多数类比率为 1:1:100 的数据集，即每个正类的比率为 1:1，少数类与多数类的比率为 1:100，每个少数类中有 100 个示例，多数类中有 10，000 个示例。

一个模型预测 1 类有 77 个正确的例子，23 个错误的例子，2 类有 95 个正确的例子，5 个错误的例子。我们可以如下计算该模型的召回率:

*   回忆=(真阳性 _1 +真阳性 _2) /((真阳性 _1 +真阳性 _2) +(假阴性 _1 +假阴性 _2))
*   回忆= (77 + 95) / ((77 + 95) + (23 + 5))
*   回忆= 172 / (172 + 28)
*   召回= 172 / 200
*   回忆= 0.86

### 用 Scikit 计算召回-学习

可以使用 [recall_score() Sklearn 功能](https://Sklearn.org/stable/modules/generated/sklearn.metrics.recall_score.html)计算召回分数。

例如，我们可以使用这个函数来计算上述场景的召回率。

首先，我们可以考虑 1:100 不平衡的情况，分别有 100 个和 10，000 个例子，一个模型预测 90 个真阳性和 10 个假阴性。

下面列出了完整的示例。

```py
# calculates recall for 1:100 dataset with 90 tp and 10 fn
from sklearn.metrics import recall_score
# define actual
act_pos = [1 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
# define predictions
pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
pred_neg = [0 for _ in range(10000)]
y_pred = pred_pos + pred_neg
# calculate recall
recall = recall_score(y_true, y_pred, average='binary')
print('Recall: %.3f' % recall)
```

运行该示例，我们可以看到分数与上面的手动计算相匹配。

```py
Recall: 0.900
```

对于不平衡的多类分类问题，我们也可以使用 *recall_score()* 。

在这种情况下，数据集具有 1:1:100 的不平衡，每个少数民族类别为 100，多数民族类别为 10，000。一个模型预测 1 类有 77 个真阳性和 23 个假阴性，2 类有 95 个真阳性和 5 个假阴性。

下面列出了完整的示例。

```py
# calculates recall for 1:1:100 dataset with 77tp,23fn and 95tp,5fn
from sklearn.metrics import recall_score
# define actual
act_pos1 = [1 for _ in range(100)]
act_pos2 = [2 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos1 + act_pos2 + act_neg
# define predictions
pred_pos1 = [0 for _ in range(23)] + [1 for _ in range(77)]
pred_pos2 = [0 for _ in range(5)] + [2 for _ in range(95)]
pred_neg = [0 for _ in range(10000)]
y_pred = pred_pos1 + pred_pos2 + pred_neg
# calculate recall
recall = recall_score(y_true, y_pred, labels=[1,2], average='micro')
print('Recall: %.3f' % recall)
```

同样，运行该示例计算与我们的手动计算相匹配的多类示例的召回率。

```py
Recall: 0.860
```

## 不平衡分类的准确率与召回率

你可以决定在不平衡的分类问题上使用精确度或召回率。

最大化准确率将最小化假阳性的数量，而最大化召回将最小化假阴性的数量。

*   **准确率**:当**最大限度地减少误报**是重点时合适。
*   **回忆**:适当的时候**尽量减少假阴性**是重点。

有时候，我们想要积极阶层的优秀预测。我们想要高准确率和高召回率。

这可能很有挑战性，因为召回率的提高往往是以精确度的降低为代价的。

> 在不平衡的数据集上，目标是在不损害准确率的情况下提高召回率。然而，这些目标往往是相互冲突的，因为为了增加少数民族的人口比例，计划生育的数量也经常增加，导致准确率降低。

—第 55 页，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

然而，我们可以选择一个新的度量标准，将精确度和召回率结合成一个分数，而不是选择一个或另一个。

## 不平衡分类的 f-测度

分类准确率被广泛使用，因为它是用于总结模型表现的单一度量。

F-Measure 提供了一种方法，可以将准确率和召回率结合到一个能够同时获取这两种属性的度量中。

单独来看，精确和回忆都不能说明全部情况。我们可以有极好的准确率和可怕的回忆，或者，可怕的准确率和极好的回忆。F-measure 提供了一种用一个分数来表达两个问题的方法。

一旦为二进制或多类分类问题计算了准确率和召回率，这两个分数就可以合并到 F-Measure 的计算中。

传统的 F 值计算如下:

*   F-Measure = (2 *准确率*召回)/(准确率+召回)

这是两个分数的[调和平均值](https://en.wikipedia.org/wiki/Harmonic_mean)。这有时被称为 F 分数或 F1 分数，可能是不平衡分类问题中最常用的度量。

> ……F1-measure 对准确率和召回率进行同等加权，是从不平衡数据中学习时最常使用的变量。

—第 27 页，[不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。

与精确度和召回率一样，较差的 F-Measure 分数为 0.0，最佳或完美的 F-Measure 分数为 1.0

例如，一个完美的准确率和召回分数将产生一个完美的 F-Measure 分数:

*   F-Measure = (2 *准确率*召回)/(准确率+召回)
*   f-测量= (2 * 1.0 * 1.0) / (1.0 + 1.0)
*   F-Measure = (2 * 1.0) / 2.0
*   f-测量= 1.0

让我们用一个实例来具体说明这个计算。

考虑一个少数与多数比例为 1:100 的二元类别数据集，有 100 个少数类示例和 10，000 个多数类示例。

考虑一个模型，该模型预测阳性类有 150 个例子，95 个是正确的(真阳性)，这意味着有 5 个被遗漏(假阴性)，55 个是不正确的(假阳性)。

我们可以按如下方式计算准确率:

*   准确率=真阳性/(真阳性+假阳性)
*   准确率= 95 / (95 + 55)
*   准确率= 0.633

我们可以按如下方式计算召回:

*   回忆=真阳性/(真阳性+假阴性)
*   回忆= 95 / (95 + 5)
*   回忆= 0.95

这说明该模型准确率差，但召回率优秀。

最后，我们可以如下计算 F-Measure:

*   F-Measure = (2 *准确率*召回)/(准确率+召回)
*   f-Measure =(2 * 0.633 * 0.95)/(0.633+0.95)
*   F-Measure = (2 * 0.601) / 1.583
*   F-Measure = 1.202 / 1.583
*   F-Measure = 0.759

我们可以看到，好的召回水平抵消了差的准确率，给出了一个好的或合理的 F-measure 分数。

### 用科学工具包计算 f 测量-学习

可以使用 [f1_score() Sklearn 功能](https://Sklearn.org/stable/modules/generated/sklearn.metrics.f1_score.html)计算 F-measure 分数。

例如，我们使用这个函数来计算上述场景的 F-Measure。

这是 1:100 不平衡的情况，分别有 100 和 10，000 个例子，一个模型预测 95 个真阳性、5 个假阴性和 55 个假阳性。

下面列出了完整的示例。

```py
# calculates f1 for 1:100 dataset with 95tp, 5fn, 55fp
from sklearn.metrics import f1_score
# define actual
act_pos = [1 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
# define predictions
pred_pos = [0 for _ in range(5)] + [1 for _ in range(95)]
pred_neg = [1 for _ in range(55)] + [0 for _ in range(9945)]
y_pred = pred_pos + pred_neg
# calculate score
score = f1_score(y_true, y_pred, average='binary')
print('F-Measure: %.3f' % score)
```

运行该示例计算 F-Measure，匹配我们的手动计算，在一些小的舍入误差内。

```py
F-Measure: 0.760
```

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何计算深度学习模型的准确率、召回率、F1 等](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)
*   [如何在 Python 中使用 ROC 曲线和查准率曲线进行分类](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)

### 报纸

*   [分类任务绩效测量的系统分析](https://www.sciencedirect.com/science/article/abs/pii/S0306457309000259)，2009。

### 书

*   [不平衡学习:基础、算法和应用](https://amzn.to/32K9K6d)，2013。
*   [从不平衡数据集中学习](https://amzn.to/307Xlva)，2018。

### 应用程序接口

*   [sklearn . metrics . precision _ score API](https://Sklearn.org/stable/modules/generated/sklearn.metrics.precision_score.html)。
*   [sklearn . metrics . recall _ score API](https://Sklearn.org/stable/modules/generated/sklearn.metrics.recall_score.html)。
*   硬化. metrics.f1_score API 。

### 文章

*   [混淆矩阵，维基百科](https://en.wikipedia.org/wiki/Confusion_matrix)。
*   [精准与召回，维基百科](https://en.wikipedia.org/wiki/Precision_and_recall)。
*   f1 得分，维基百科。

## 摘要

在本教程中，您发现了如何为不平衡分类计算和开发精确度和召回率的直觉。

具体来说，您了解到:

*   准确率量化了实际属于正类的正类预测的数量。
*   Recall 量化了从数据集中所有正面示例中做出的正面类预测的数量。
*   F-Measure 提供了一个单一的分数，在一个数字中平衡了对精确度和召回率的关注。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。