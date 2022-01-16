# 为什么优化在机器学习中很重要

> 原文:[https://machinelearning master . com/为什么优化在机器学习中非常重要/](https://machinelearningmastery.com/why-optimization-is-important-in-machine-learning/)

最后更新于 2021 年 10 月 12 日

机器学习包括使用算法从历史数据中学习和归纳，以便对新数据进行预测。

这个问题可以描述为近似一个函数，该函数将输入的例子映射到输出的例子。逼近一个函数可以通过将问题框架化为函数优化来解决。这是机器学习算法定义参数化映射函数(例如，输入的加权和)的地方，并且优化算法用于为参数(例如，模型系数)的值提供资金，当用于将输入映射到输出时，该参数最小化函数的误差。

这意味着，每次我们在训练数据集上拟合机器学习算法时，我们都在解决一个优化问题。

在本教程中，您将发现优化在机器学习中的核心作用。

完成本教程后，您将知道:

*   机器学习算法执行函数逼近，这是使用函数优化来解决的。
*   函数优化是我们在拟合机器学习算法时最小化误差、成本或损失的原因。
*   在预测建模项目的数据准备、超参数调整和模型选择过程中，也会执行优化。

**用我的新书[机器学习优化](https://machinelearningmastery.com/optimization-for-machine-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

Let’s get started.![Why Optimization Is Important in Machine Learning](img/bcae90c56265dce584b208cc19407af1.png)

为什么优化在机器学习中很重要
图片由[马尔科·韦奇](https://www.flickr.com/photos/160866001@N07/50119137418/)提供，版权所有。

## 教程概述

本教程分为三个部分；它们是:

1.  机器学习与优化
2.  学习即优化
3.  机器学习项目中的优化
    1.  数据准备作为优化
    2.  超参数优化
    3.  作为优化的模型选择

## 机器学习与优化

[函数优化](https://en.wikipedia.org/wiki/Mathematical_optimization)是寻找目标目标函数的输入集的问题，该输入集导致函数的最小值或最大值。

这可能是一个具有挑战性的问题，因为函数可能有几十个、几百个、几千个甚至几百万个输入，并且函数的结构是未知的，并且通常是不可微的和有噪声的。

*   **函数优化**:找到导致目标函数最小或最大的输入集。

机器学习可以描述为[函数逼近](https://en.wikipedia.org/wiki/Function_approximation)。也就是说，近似未知的潜在函数，该函数将输入示例映射到输出，以便对新数据进行预测。

这可能很有挑战性，因为我们可以用来近似函数的例子通常数量有限，并且被近似的函数的结构通常是非线性的、有噪声的，甚至可能包含矛盾。

*   **函数逼近**:从具体的例子中归纳出一个可重用的映射函数，用于对新的例子进行预测。

函数优化通常比函数逼近简单。

重要的是，在机器学习中，我们经常使用函数优化来解决函数逼近的问题。

几乎所有机器学习算法的核心都是优化算法。

此外，处理预测建模问题的过程除了学习模型之外，还涉及多个步骤的优化，包括:

*   模型超参数的选择。
*   在建模之前选择要应用于数据的转换
*   选择要用作最终模型的建模管道。

现在我们知道优化在机器学习中起着核心作用，让我们看看一些学习算法的例子以及它们如何使用优化。

## 学习即优化

预测建模问题涉及到根据输入示例进行预测。

在回归问题的情况下，必须预测数字量，而在分类问题的情况下，必须预测类别标签。

预测建模的问题非常具有挑战性，以至于我们无法编写代码来进行预测。相反，我们必须使用应用于历史数据的学习算法来学习一个名为“预测模型”的“*”程序，我们可以使用该程序对新数据进行预测。*

在统计学习(机器学习的一种统计观点)中，问题被框架化为给定输入数据( *X* )和相关输出数据( *y* 的例子的映射函数(f)的学习。

*   y = f(X)

给定新的输入示例( *Xhat* )，我们必须使用我们学习的函数( *fhat* )将每个示例映射到预期的输出值( *yhat* )。

*   yhat = fhat

所学的映射将是不完美的。没有一个模型是完美的，考虑到问题的难度、观测数据中的噪声以及学习算法的选择，预计会有一些预测误差。

数学上，学习算法通过解决函数优化问题来解决逼近映射函数的问题。

具体来说，给定输入和输出的示例，找到导致最小损失、最小成本或最小预测误差的映射函数的输入集。

映射函数的选择越偏向或受约束，优化就越容易解决。

让我们看一些例子来说明这一点。

线性回归(用于回归问题)是一种高度受限的模型，可以使用线性代数进行解析求解。映射函数的输入是模型的系数。

我们可以使用优化算法，如拟牛顿局部搜索算法，但它的效率几乎总是低于解析解。

*   **线性回归**:函数输入是模型系数，可以解析求解的优化问题。

逻辑回归(用于分类问题)受到的约束稍少，必须作为优化问题来解决，尽管在模型施加的约束下，已知正在解决的优化函数的结构。

这意味着可以使用类似拟牛顿法的局部搜索算法。我们可以使用像随机梯度下降这样的全局搜索，但它几乎总是效率较低。

*   **逻辑回归**:函数输入是模型系数，需要迭代局部搜索算法的优化问题。

神经网络模型是一种非常灵活的学习算法，几乎没有约束。映射函数的输入是网络权重。给定的搜索空间是多模态和高度非线性的，不能使用局部搜索算法；相反，必须使用全局搜索算法。

通常使用全局优化算法，特别是随机梯度下降，并且更新是以知道模型结构的方式进行的(反向传播和链式规则)。我们可以使用一种全局搜索算法，它不考虑模型的结构，就像遗传算法一样，但它几乎总是效率较低。

*   **神经网络**:函数输入是模型权重，需要迭代全局搜索算法的优化问题。

我们可以看到，每个算法对映射函数的形式做出不同的假设，这影响了要解决的优化问题的类型。

我们还可以看到，每个机器学习算法使用的默认优化算法并不是任意的；它代表了用于解决由该算法构成的特定优化问题的最有效算法，例如，神经网络的随机梯度下降而不是遗传算法。偏离这些默认值需要一个很好的理由。

并非所有的机器学习算法都能解决优化问题。一个值得注意的例子是 k 近邻算法，它存储训练数据集，并查找每个新例子的 k 个最佳匹配，以便进行预测。

现在我们已经熟悉了机器学习算法中的学习作为优化，让我们看看机器学习项目中一些相关的优化示例。

## 机器学习项目中的优化

除了在训练数据集上拟合学习算法之外，优化在机器学习项目中起着重要的作用。

在拟合模型之前准备数据的步骤和调整所选模型的步骤也可以被视为优化问题。事实上，整个预测建模项目可以被认为是一个大的优化问题。

让我们依次仔细看看这些案例。

### 数据准备作为优化

数据准备包括将原始数据转换成最适合学习算法的形式。

这可能涉及缩放值、处理缺失值以及改变变量的概率分布。

可以进行转换来改变历史数据的表示，以满足特定学习算法的期望或要求。然而，有时当期望被违背或者当对数据执行不相关的转换时，可以获得好的或者最好的结果。

我们可以把选择应用于训练数据的变换看作是一个搜索或优化问题，最好地将数据的未知底层结构暴露给学习算法。

*   **数据准备**:函数输入是需要迭代全局搜索算法的变换、优化问题的序列。

这个优化问题通常是通过人工试错来完成的。然而，有可能使用全局优化算法来自动化该任务，其中函数的输入是应用于训练数据的变换的类型和顺序。

数据变换的数量和排列通常是非常有限的，并且可以对常用序列进行穷举搜索或网格搜索。

有关此主题的更多信息，请参见教程:

*   [如何网格搜索数据准备技术](https://machinelearningmastery.com/grid-search-data-preparation-techniques/)

### 超参数优化

机器学习算法具有超参数，这些超参数可以被配置为针对特定数据集定制算法。

虽然许多超参数的动态是已知的，但是它们对给定数据集上的结果模型的性能的具体影响是未知的。因此，测试所选机器学习算法的关键算法超参数的一组值是标准做法。

这被称为超参数调谐或[超参数优化](https://en.wikipedia.org/wiki/Hyperparameter_optimization)。

为此，通常使用简单的优化算法，如随机搜索算法或网格搜索算法。

*   **超参数调整**:函数输入是算法超参数，需要迭代全局搜索算法的优化问题。

有关此主题的更多信息，请参见教程:

*   [随机搜索和网格搜索的超参数优化](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)

然而，使用迭代全局搜索算法来解决这个优化问题变得越来越普遍。一种流行的选择是贝叶斯优化算法，它能够在优化目标函数的同时(使用替代函数)逼近正在优化的目标函数。

这是所希望的，因为评估模型超参数的单个组合是昂贵的，需要根据模型评估程序的选择，在整个训练数据集上拟合模型一次或多次(例如[重复 k 倍交叉验证](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/))。

有关此主题的更多信息，请参见教程:

*   [如何在 Python 中从头实现贝叶斯优化](https://machinelearningmastery.com/what-is-bayesian-optimization/)

### 作为优化的模型选择

模型选择包括从许多候选机器学习模型中选择一个用于预测建模问题。

实际上，它包括选择机器学习算法或产生模型的机器学习管道。然后用它来训练一个最终的模型，这个模型可以用在期望的应用中，对新的数据进行预测。

这个模型选择的过程通常是由机器学习实践者执行的手动过程，包括准备数据、评估候选模型、调整表现良好的模型以及最终选择最终模型等任务。

这可以被框架化为一个优化问题，包含部分或整个预测建模项目。

*   **模型选择**:功能输入为数据变换、机器学习算法、算法超参数；需要迭代全局搜索算法的优化问题。

自动机器学习(AutoML)算法越来越多地被用于选择算法、算法和超参数，或者数据准备、算法和超参数，而几乎没有用户干预。

有关 AutoML 的更多信息，请参见教程:

*   [Python 自动机器学习(AutoML)库](https://machinelearningmastery.com/automl-libraries-for-python/)

与超参数调整一样，通常使用也近似目标函数的全局搜索算法，例如贝叶斯优化，因为每个函数评估都很昂贵。

这种自动优化机器学习的方法也是现代机器学习即服务产品的基础，这些产品由谷歌、微软和亚马逊等公司提供。

尽管快速高效，但这种方法仍然无法超越由高技能专家(如参加机器学习竞赛的专家)准备的手工模型。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [作为搜索问题的应用机器学习的温和介绍](https://machinelearningmastery.com/applied-machine-learning-as-a-search-problem/)
*   [如何网格搜索数据准备技术](https://machinelearningmastery.com/grid-search-data-preparation-techniques/)
*   [随机搜索和网格搜索的超参数优化](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)
*   [如何在 Python 中从头实现贝叶斯优化](https://machinelearningmastery.com/what-is-bayesian-optimization/)
*   [Python 自动机器学习(AutoML)库](https://machinelearningmastery.com/automl-libraries-for-python/)

### 文章

*   [数学优化，维基百科](https://en.wikipedia.org/wiki/Mathematical_optimization)。
*   [函数逼近，维基百科](https://en.wikipedia.org/wiki/Function_approximation)。
*   [最小二乘函数逼近，维基百科](https://en.wikipedia.org/wiki/Least-squares_function_approximation)。
*   [超参数优化，维基百科](https://en.wikipedia.org/wiki/Hyperparameter_optimization)。
*   [车型选择，维基百科](https://en.wikipedia.org/wiki/Model_selection)。

## 摘要

在本教程中，您发现了优化在机器学习中的核心作用。

具体来说，您了解到:

*   机器学习算法执行函数逼近，这是使用函数优化来解决的。
*   函数优化是我们在拟合机器学习算法时最小化误差、成本或损失的原因。
*   在预测建模项目的数据准备、超参数调整和模型选择过程中，也会执行优化。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。