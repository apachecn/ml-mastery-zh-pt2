# 为机器学习学习概率的 5 个理由

> 原文：<https://machinelearningmastery.com/why-learn-probability-for-machine-learning/>

最后更新于 2019 年 11 月 8 日

概率是一个量化不确定性的数学领域。

不可否认，它是机器学习领域的一个支柱，许多人建议在开始之前将其作为学习的前提科目。这是一个误导性的建议，因为一旦从业者有了应用机器学习过程的背景来解释它，概率对他们来说就更有意义了。

在这篇文章中，你将发现为什么机器学习实践者应该学习概率来提高他们的技能和能力。

看完这篇文章，你会知道:

*   不是每个人都应该学习概率；这取决于你在学习机器学习的旅程中所处的位置。
*   许多算法是使用概率的工具和技术设计的，例如朴素贝叶斯和概率图形模型。
*   作为许多机器学习算法训练基础的最大似然框架来自概率领域。

**用我的新书[机器学习概率](https://machinelearningmastery.com/probability-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![5 Reasons to Learn Probability for Machine Learning](img/315ba5b6a99ee90605f750da73de78f8.png)

学习机器学习概率的 5 个理由
图片由[马可·韦奇](https://www.flickr.com/photos/30478819@N08/35542222010/)提供，保留部分权利。

## 概观

本教程分为七个部分；它们是:

1.  不学概率的理由
2.  类别成员资格需要预测概率
3.  一些算法是用概率设计的
4.  使用概率框架训练模型
5.  模型可以用概率框架来调整
6.  概率测量用于评估模型技能
7.  还有一个原因

## 不学概率的理由

在我们讨论你应该学习概率的原因之前，让我们先来看看你不应该学习概率的原因。

我认为如果你刚刚开始应用机器学习，就不应该学习概率。

*   **不需要**。为了使用机器学习作为解决问题的工具，并不需要理解一些机器学习算法的抽象理论。
*   **很慢**。在开始机器学习之前，花几个月到几年的时间来研究整个相关领域，会延迟你实现能够解决预测建模问题的目标。
*   **这是一个巨大的领域**。不是所有的概率都与理论机器学习相关，更不用说应用机器学习了。

我推荐一种广度优先的方法来开始应用机器学习。

我称之为结果第一的方法。这是您从学习和实践使用工具(如 Sklearn 和 Python 中的 Pandas)端到端解决预测建模问题的步骤(例如如何获得结果)开始的地方。

然后，这个过程为逐步加深你的知识提供了框架和背景，比如算法是如何工作的，以及最终作为基础的数学。

在你知道如何解决预测建模问题后，让我们看看为什么你应该加深对概率的理解。

## 1.类别成员资格需要预测概率

分类预测建模问题是那些给一个例子指定一个给定标签的问题。

您可能熟悉的一个例子是[鸢尾花数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set)，其中我们对一朵花进行了四次测量，目标是将三种不同的已知鸢尾花中的一种分配给观察。

我们可以将问题建模为直接给每个观察分配一个类标签。

*   **输入**:一朵花的尺寸。
*   **输出**:一种鸢尾。

一种更常见的方法是将问题框架化为一个概率类成员关系，其中预测属于每个已知类的观测值的概率。

*   **输入**:一朵花的尺寸。
*   **输出**:每个鸢尾属种的隶属概率。

将问题框架化为类成员的预测简化了建模问题，并使模型更容易学习。它允许模型捕捉数据中的模糊性，这允许下游的过程，例如用户在域的上下文中解释概率。

通过选择概率最大的类，可以将概率转化为清晰的类标签。也可以使用[概率校准过程](https://machinelearningmastery.com/calibrated-classification-model-in-Sklearn/)来缩放或转换概率。

这种选择类成员框架的问题解释模型所做的预测需要对概率的基本理解。

## 2.模型是用概率设计的

有些算法是专门为利用概率的工具和方法而设计的。

这些算法包括单个算法，如朴素贝叶斯算法，它是使用[贝叶斯定理](https://machinelearningmastery.com/bayes-theorem-for-machine-learning/)和一些简化假设构建的。

*   [朴素贝叶斯](https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/)

线性回归算法可被视为最小化预测均方误差的概率模型，逻辑回归算法可被视为最小化预测正类别标签的负对数似然性的概率模型。

*   线性回归
*   逻辑回归

它还扩展到整个研究领域，如概率图形模型，通常称为[图形模型](https://en.wikipedia.org/wiki/Graphical_model)或简称 PGM，围绕贝叶斯定理设计。

一个值得注意的图形模型是贝叶斯信念网络，它能够捕捉变量之间的条件依赖关系。

*   [贝叶斯信念网络](https://machinelearningmastery.com/introduction-to-bayesian-belief-networks/)

## 3.用概率框架训练模型

许多机器学习模型是使用在概率框架下设计的迭代算法来训练的。

一般概率建模框架的一些例子有:

*   [最大似然估计](https://machinelearningmastery.com/what-is-maximum-likelihood-estimation-in-machine-learning/)(经常光顾者)。
*   [最大后验估计](https://machinelearningmastery.com/maximum-a-posteriori-estimation/)(贝叶斯)。

也许最常见的是最大似然估计的框架，有时简称 MLE。这是在给定观测数据的情况下估计模型参数(例如权重)的框架。

这是线性回归模型的普通最小二乘估计和逻辑回归的对数损失估计的基础框架。

[期望最大化](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)算法，简称 EM，是一种最大似然估计方法，通常用于无监督数据聚类，例如估计 k 个聚类的 k 均值，也称为 [k 均值聚类算法](https://en.wikipedia.org/wiki/K-means_clustering)。

对于预测类别成员的模型，最大似然估计提供了一个框架，用于最小化观察到的概率分布和预测到的概率分布之间的差异或分歧。这用于分类算法，如逻辑回归以及深度学习神经网络。

通常在训练期间使用熵，例如通过交叉熵来测量概率分布的这种差异。熵，以及通过 [KL 散度](https://machinelearningmastery.com/divergence-between-probability-distributions/)和[交叉熵](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)测量的分布之间的差异来自信息论领域，该领域直接建立在概率论的基础上。例如，熵直接计算为概率的负对数。

因此，这些来自信息论的工具，如最小化交叉熵损失，可以被视为模型估计的另一个概率框架。

*   最小交叉熵损失估计

## 4.用概率框架调整模型

调整机器学习模型的超参数是很常见的，例如 kNN 的 k 或神经网络中的学习率。

典型的方法包括超参数的网格搜索范围或随机采样超参数组合。

贝叶斯优化是一种更有效的超参数优化，它包括基于最有可能产生更好表现的配置对可能配置的空间进行定向搜索。顾名思义，这种方法是在对可能的构型空间进行采样时，根据并利用[贝叶斯定理](https://machinelearningmastery.com/bayes-theorem-for-machine-learning/)设计出来的。

有关贝叶斯优化的更多信息，请参见教程:

*   [如何在 Python 中从头实现贝叶斯优化](https://machinelearningmastery.com/what-is-bayesian-optimization/)

## 5.用概率方法评估模型

对于那些进行概率预测的算法，需要评估措施来总结模型的表现。

有许多方法可以用来总结基于预测概率的模型的表现。常见的例子包括:

*   对数损失(也称为[交叉熵](https://machinelearningmastery.com/cross-entropy-for-machine-learning/))。
*   更简单的分数和更简单的技能分数

有关评估预测概率的指标的更多信息，请参见教程:

*   [Python 中概率评分方法的温和介绍](https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/)

对于预测单一概率分数的二进制分类任务，可以构建接收器操作特性(ROC)曲线，以探索在解释预测时可以使用的不同临界值，这些临界值反过来会导致不同的权衡。ROC 曲线下的面积，或 ROC AUC，也可以作为一个综合指标来计算。正类上的一个相关方法是准确率-召回曲线和曲线下面积。

*   ROC 曲线和 ROC 曲线面积比
*   准确率-召回曲线和 AUC

有关这些曲线以及何时使用它们的更多信息，请参见教程:

*   [如何在 Python 中使用 ROC 曲线和查准率曲线进行分类](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)

这些评分方法的选择和解释需要对概率论有基本的理解。

## 还有一个原因

如果我能再给一个理由，那就是:因为它很有趣。

说真的。

学习概率，至少是我用实际例子和可执行代码教概率的方式，很有意思。一旦你能看到运算是如何在真实数据上工作的，就很难避免对一个通常不直观的主题产生强烈的直觉。

中级机器学习从业者学习概率很关键，你有更多的理由吗？

请在下面的评论中告诉我。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [模式识别与机器学习](https://amzn.to/2JwHE7I)，2006。
*   [机器学习:概率视角](https://amzn.to/2xKSTCP)，2012。
*   [机器学习](https://amzn.to/2jWd51p)，1997。

### 邮件

*   [Python 中概率评分方法的温和介绍](https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/)
*   [如何以及何时在 Python 中使用 ROC 曲线和准确率-召回曲线进行分类](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)
*   [训练深度学习神经网络时如何选择损失函数](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)

### 文章

*   [图形模型，维基百科](https://en.wikipedia.org/wiki/Graphical_model)。
*   [最大似然估计，维基百科](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)。
*   [期望最大化算法，维基百科](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)。
*   [交叉熵，维基百科](https://en.wikipedia.org/wiki/Cross_entropy)。
*   [kul LBA-leilbler 分歧，维基百科](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)。
*   [贝叶斯优化，维基百科](https://en.wikipedia.org/wiki/Bayesian_optimization)。

## 摘要

在这篇文章中，你发现了为什么作为机器学习的实践者，你应该加深对概率的理解。

具体来说，您了解到:

*   不是每个人都应该学习概率；这取决于你在学习机器学习的旅程中所处的位置。
*   许多算法是使用概率的工具和技术设计的，例如朴素贝叶斯和概率图形模型。
*   作为许多机器学习算法训练基础的最大似然框架来自概率领域。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。