# 如何选择优化算法

> 原文：<https://machinelearningmastery.com/tour-of-optimization-algorithms/>

最后更新于 2021 年 10 月 12 日

**优化**是寻找目标函数的一组输入的问题，该目标函数导致最大或最小的函数求值。

从拟合逻辑回归模型到训练人工神经网络，这是许多机器学习算法背后的挑战性问题。

在流行的科学代码库中，可能有数百种流行的优化算法，也可能有数十种算法可供选择。这使得知道对于给定的优化问题要考虑哪些算法变得很有挑战性。

在本教程中，您将发现不同优化算法的导游。

完成本教程后，您将知道:

*   优化算法可以分为使用导数的算法和不使用导数的算法。
*   经典算法使用目标函数的一阶导数，有时也使用二阶导数。
*   直接搜索和随机算法是为函数导数不可用的目标函数设计的。

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![How to Choose an Optimization Algorithm](img/61c806943d9edd68bfcd472defa313f1.png)

如何选择优化算法
图片由 [Matthewjs007](https://www.flickr.com/photos/mattz27/8676145834/) 提供，保留部分权利。

## 教程概述

本教程分为三个部分；它们是:

1.  优化算法
2.  可微目标函数
3.  非微分目标函数

## 优化算法

优化是指寻找函数的输入参数或自变量的过程，这些参数或自变量导致函数的最小或最大输出。

机器学习中遇到的最常见的优化问题类型是**连续函数优化**，其中函数的输入参数是实值数值，例如浮点值。函数的输出也是输入值的实值计算。

我们可以将这类问题称为连续函数优化，以区别于采用离散变量的函数，并称为组合优化问题。

有许多不同类型的优化算法可以用于连续函数优化问题，也许还有许多方法可以对它们进行分组和总结。

对优化算法进行分组的一种方法是基于正在被优化的目标函数的可用信息量，该信息量又可以被优化算法使用和利用。

通常，关于目标函数的可用信息越多，如果该信息可以有效地用于搜索，该函数就越容易优化。

也许优化算法的主要分歧在于目标函数是否能在某一点上被微分。也就是说，对于给定的候选解，是否可以计算函数的一阶导数(梯度或斜率)。这将算法分为可以利用计算出的梯度信息的算法和不利用计算出的梯度信息的算法。

*   可微目标函数？
    *   使用导数信息的算法。
    *   不使用导数信息的算法。

在本教程中，我们将把它作为分组优化算法的主要部分，并研究可微和不可微目标函数的算法。

**注**:这并不是对连续函数优化算法的详尽介绍，尽管它确实涵盖了你作为一名常规从业者可能遇到的主要方法。

## 可微目标函数

一个[可微函数](https://en.wikipedia.org/wiki/Differentiable_function)是一个可以计算输入空间中任意给定点的导数的函数。

一个函数对一个值的导数是函数在该点的变化率或变化量。它通常被称为斜坡。

*   **一阶导数**:目标函数在给定点的斜率或变化率。

具有一个以上输入变量(例如多元输入)的函数的导数通常称为梯度。

*   **梯度**:多元连续目标函数的导数。

多元目标函数的导数是一个向量，向量中的每个元素都称为偏导数，或者是给定变量在假设所有其他变量保持不变的情况下的变化率。

*   **偏导数**:多元目标函数导数的元素。

我们可以计算目标函数导数的导数，即目标函数变化率的变化率。这叫做二阶导数。

*   **二阶导数**:目标函数导数变化的速率。

对于一个接受多个输入变量的函数，这是一个矩阵，称为黑森矩阵。

*   **黑森矩阵**:具有两个或多个输入变量的函数的二阶导数。

简单的可微函数可以使用微积分进行分析优化。通常，我们感兴趣的目标函数无法解析求解。

如果可以计算出目标函数的梯度，优化就容易得多，因此，对使用导数的优化算法的研究比不使用导数的算法多得多。

使用梯度信息的一些算法组包括:

*   包围算法
*   局部下降算法
*   一阶算法
*   二阶算法

**注**:本分类受 2019 年《优化的[算法](https://amzn.to/39KZSQn)》一书启发

让我们依次仔细看看每一个。

### 包围算法

包围优化算法旨在解决具有一个输入变量的优化问题，其中最优解已知存在于特定范围内。

包围算法能够有效地导航已知范围并定位最优解，尽管它们假设只存在一个最优解(称为单峰目标函数)。

如果导数信息不可用，一些包围算法可以在没有导数信息的情况下使用。

包围算法的例子包括:

*   斐波那契搜索
*   黄金分割搜索
*   等分法

### 局部下降算法

局部下降优化算法适用于具有多个输入变量和单个全局最优值(例如单峰目标函数)的优化问题。

也许局部下降算法最常见的例子是线性搜索算法。

*   行搜索

线性搜索有许多变体(例如布伦特-德克尔算法)，但是该过程通常涉及选择在搜索空间中移动的方向，然后在所选方向上的线或超平面中执行包围式搜索。

重复这个过程，直到没有进一步的改进。

限制在于优化搜索空间中的每个方向移动在计算上是昂贵的。

### 一阶算法

一阶优化算法明确涉及使用一阶导数(梯度)来选择在搜索空间中移动的方向。

该过程包括首先计算函数的梯度，然后使用步长(也称为学习率)沿相反方向(例如下坡至最小化问题的最小值)跟随梯度。

步长是一个超参数，它控制在搜索空间中移动多远，这与“局部下降算法”不同，后者对每个方向的移动执行一个完整的直线性搜索。

步长过小会导致搜索耗时较长并可能被卡住，而步长过大则会导致搜索空间的曲折或跳跃，完全错过最佳值。

一阶算法通常称为梯度下降，更具体的名称是指程序的微小扩展，例如:

*   梯度下降
*   动力
*   阿达格勒
*   RMSProp
*   圣经》和《古兰经》传统中）亚当（人类第一人的名字

梯度下降算法还为流行的随机版本算法提供了模板，该算法被称为随机梯度下降(SGD)，用于训练人工神经网络(深度学习)模型。

重要的区别是梯度是适当的，而不是直接计算的，使用训练数据的预测误差，如一个样本(随机)，所有例子(批次)，或训练数据的小子集(小批次)。

旨在加速梯度下降算法(动量等)的扩展。)可以是并且通常与 SGD 一起使用。

*   随机梯度下降
*   分批梯度下降
*   小批量梯度下降

### 二阶算法

二阶优化算法明确涉及使用二阶导数(Hessian)来选择在搜索空间中移动的方向。

这些算法只适用于那些可以计算或近似黑森矩阵的目标函数。

单变量目标函数的二阶优化算法示例包括:

*   牛顿法
*   割线法

多元目标函数的二阶方法称为拟牛顿法。

*   拟牛顿法

有许多拟牛顿方法，它们通常以算法的开发者命名，例如:

*   戴维森-弗莱彻-鲍威尔
*   布赖登-弗莱彻-戈德法布-尚诺(BFGS)
*   有限记忆 BFGS(左 BFGS)

现在我们已经熟悉了所谓的经典优化算法，让我们看看当目标函数不可微时使用的算法。

## 非微分目标函数

利用目标函数导数的优化算法快速有效。

然而，有些目标函数的导数无法计算，这通常是因为该函数由于各种现实原因而变得复杂。或者导数可以在定义域的某些区域计算，但不是全部，或者不是很好的指导。

上一节描述的经典算法的目标函数的一些困难包括:

*   没有功能的分析描述(例如模拟)。
*   多重全局最优(例如多模态)。
*   随机函数评估(如噪声)。
*   不连续的目标函数(例如，具有无效解的区域)。

因此，有些优化算法不期望一阶或二阶导数可用。

这些算法有时被称为黑盒优化算法，因为它们对目标函数的假设很少或没有(相对于经典方法)。

这些算法的组合包括:

*   直接算法
*   随机算法
*   人口算法

让我们依次仔细看看每一个。

### 直接算法

直接优化算法适用于无法计算导数的目标函数。

算法是确定性的过程，并且通常假设目标函数具有单个全局最优值，例如单峰。

直接搜索方法通常也被称为“*模式搜索*，因为它们可以使用几何形状或决策(例如模式)来导航搜索空间。

梯度信息直接从目标函数比较搜索空间中的点的分数之间的相对差异的结果来近似(因此得名)。然后，这些直接估计被用于选择在搜索空间中移动的方向，并对最优区域进行三角测量。

直接搜索算法的例子包括:

*   循环坐标搜索
*   鲍威尔方法
*   胡克-吉夫斯方法
*   NelderMead 单纯形搜索

### 随机算法

随机优化算法是在目标函数的搜索过程中利用随机性的算法，这些目标函数的导数无法计算。

与确定性直接搜索方法不同，随机算法通常需要对目标函数进行更多的采样，但能够处理具有欺骗性局部最优的问题。

随机优化算法包括:

*   模拟退火
*   进化策略
*   交叉熵方法

### 人口算法

种群优化算法是一种随机优化算法，它维护一个候选解池(种群)，这些解一起用于采样、探索和钻研一个最优解。

这种类型的算法旨在解决更具挑战性的目标问题，这些问题可能具有噪声函数评估和许多全局最优(多模态)，并且使用其他方法找到好的或足够好的解决方案是具有挑战性的或不可行的。

候选解决方案库增加了搜索的健壮性，增加了克服局部最优的可能性。

群体优化算法的例子包括:

*   遗传算法
*   差分进化
*   粒子群优化算法

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 书

*   [优化算法](https://amzn.to/39KZSQn)，2019。
*   [元试探法精要](https://amzn.to/2HxZVn4)，2011。
*   [计算智能:导论](https://amzn.to/2HzjbjV)，2007。
*   [随机搜索与优化导论](https://amzn.to/34JYN7m)，2003。

### 文章

*   [数学优化，维基百科](https://en.wikipedia.org/wiki/Mathematical_optimization)。

## 摘要

在本教程中，您发现了不同优化算法的导游。

具体来说，您了解到:

*   优化算法可以分为使用导数的算法和不使用导数的算法。
*   经典算法使用目标函数的一阶导数，有时也使用二阶导数。
*   直接搜索和随机算法是为函数导数不可用的目标函数设计的。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。