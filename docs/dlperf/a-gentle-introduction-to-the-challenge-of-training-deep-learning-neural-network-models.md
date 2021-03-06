# 训练深度学习神经网络模型的挑战的温和介绍

> 原文：<https://machinelearningmastery.com/a-gentle-introduction-to-the-challenge-of-training-deep-learning-neural-network-models/>

最后更新于 2021 年 1 月 18 日

深度学习神经网络学习从输入到输出的映射函数。

这是通过更新网络的权重来实现的，以响应模型在训练数据集上产生的误差。不断进行更新以减少这种错误，直到找到足够好的模型，或者学习过程停滞不前。

一般来说，训练神经网络的过程是使用该技术最具挑战性的部分，并且就配置该过程所需的努力和执行该过程所需的计算复杂性而言，是最耗时的。

在这篇文章中，你将发现为深度学习神经网络寻找模型参数的挑战。

看完这篇文章，你会知道:

*   神经网络学习从输入到输出的映射函数，可以概括为解决函数逼近问题。
*   与其他机器学习算法不同，神经网络的参数必须通过解决具有许多好的解和许多误导性好的解的非凸优化问题来找到。
*   随机梯度下降算法用于解决优化问题，其中使用反向传播算法在每次迭代中更新模型参数。

**用我的新书[更好的深度学习](https://machinelearningmastery.com/better-deep-learning/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![A Gentle Introduction to the Challenge of Training Deep Learning Neural Network Models](img/201dc91937b6295c7ad18d1911265094.png)

温和介绍训练深度学习神经网络模型的挑战
图片由[米盖尔·迪凯特](https://www.flickr.com/photos/miguel_discart_vrac/24331314025/)提供，版权所有。

## 概观

本教程分为四个部分；它们是:

1.  神经网络学习映射函数
2.  学习网络权重很难
3.  导航错误表面
4.  学习算法的组成部分

## 神经网络学习映射函数

深度学习神经网络学习映射函数。

开发模型需要来自域的历史数据，这些数据用作训练数据。该数据由来自域的观察或示例组成，具有描述条件的输入元素和捕获观察含义的输出元素。

例如，输出是一个量的问题通常被描述为回归预测建模问题。而输出是标签的问题通常被描述为分类预测建模问题。

神经网络模型使用示例来学习如何将特定的输入变量集映射到输出变量。它必须以这样一种方式做到这一点，即这种映射对训练数据集很有效，但对模型在训练期间看不到的新示例也很有效。这种在具体例子和新例子中表现良好的能力被称为模型概括的能力。

> 多层感知器只是将一组输入值映射到输出值的数学函数。

—第 5 页，[深度学习](https://amzn.to/2NJW3gE)，2016。

我们可以把输入变量和输出变量之间的关系描述成一个复杂的数学函数。对于给定的模型问题，我们必须相信存在一个真实的映射函数来最好地将输入变量映射到输出变量，并且神经网络模型可以在逼近真实未知的底层映射函数方面做得很好。

> 前馈网络定义了一个映射，并学习导致最佳函数逼近的参数值。

—第 168 页，[深度学习](https://amzn.to/2NJW3gE)，2016。

因此，我们可以将神经网络解决的更广泛的问题描述为“*函数逼近*”他们学习在给定训练数据集的情况下逼近未知的底层映射函数。给定我们设计的特定网络结构，他们通过学习权重和模型参数来做到这一点。

> 最好将前馈网络视为功能近似机器，旨在实现统计泛化，偶尔从我们对大脑的了解中获得一些见解，而不是将其视为大脑功能的模型。

—第 169 页，[深度学习](https://amzn.to/2NJW3gE)，2016。

## 学习网络权重很难

寻找神经网络的参数通常是困难的。

对于许多更简单的机器学习算法，我们可以在给定训练数据集的情况下计算最优模型。

例如，我们可以使用线性代数来计算线性回归模型和训练数据集的特定系数，该训练数据集可以最大限度地减少平方误差。

同样，我们可以使用优化算法，当为非线性算法(如逻辑回归或支持向量机)找到一组最佳模型参数时，这些算法可以提供收敛保证。

为许多机器学习算法寻找参数涉及解决一个凸优化问题:这是一个误差表面，形状像一个碗，只有一个最佳解决方案。

深度学习神经网络不是这样。

我们既不能直接计算模型的最优权值集，也不能得到全局收敛保证来寻找最优权值集。

> 应用于非凸损失函数的随机梯度下降法没有【……】收敛保证，并且对初始参数值敏感。

—第 177 页，[深度学习](https://amzn.to/2NJW3gE)，2016。

事实上，训练神经网络是使用该技术最具挑战性的部分。

> 为了解决神经网络训练问题的一个实例，在数百台机器上投入几天到几个月的时间是很常见的。

—第 274 页，[深度学习](https://amzn.to/2NJW3gE)，2016。

在神经网络中使用非线性激活函数意味着我们为了找到模型参数而必须解决的优化问题不是凸的。

它不是一个简单的碗形，只有一套我们肯定能找到的最佳重量。相反，有一个山峰和山谷的景观，有许多好的和许多误导性的好参数集，我们可能会发现。

解决这个优化问题很有挑战性，尤其是因为误差面包含许多局部最优解、平点和悬崖。

必须使用迭代过程来导航模型的非凸误差表面。一个导航错误的幼稚算法很可能会被误导、丢失，并最终陷入困境，从而导致模型表现不佳。

## 导航非凸误差曲面

神经网络模型可以被认为是通过导航非凸误差表面来学习的。

可以在训练数据集上评估具有特定权重集的模型，并且可以将所有训练数据集的平均误差视为模型的误差。模型权重的改变将导致模型误差的改变。因此，我们寻求一组权重，从而得到一个误差很小的模型。

这包括重复评估模型和更新模型参数的步骤，以便降低误差面。重复这个过程，直到找到一组足够好的参数，或者搜索过程停滞不前。

这是一个搜索或优化过程，我们称以这种方式运行的优化算法为梯度优化算法，因为它们天真地沿着误差梯度运行。它们计算成本高，速度慢，它们的经验行为意味着在实践中使用它们更多的是艺术而不是科学。

最常用于导航误差表面的算法称为随机梯度下降，简称 SGD。

> 几乎所有的深度学习都是由一个非常重要的算法提供动力的:随机梯度下降或 SGD。

—第 151 页，[深度学习](https://amzn.to/2NJW3gE)，2016。

可以使用为非凸优化问题设计的其他全局优化算法，例如遗传算法，但是随机梯度下降更有效，因为它通过称为[反向传播](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)的算法专门使用梯度信息来更新模型权重。

> [反向传播]描述了一种通过巧妙应用导数链规则来计算网络训练误差相对于权重的导数的方法。

—第 49 页，[神经锻造:前馈人工神经网络中的监督学习](https://amzn.to/2S8qRdI)，1999。

反向传播是指从微积分中计算特定模型参数的模型误差的导数(例如斜率或梯度)的技术，允许更新模型权重以向下移动梯度。因此，用于训练神经网络的算法通常也称为简单的反向传播。

> 实际上，反向传播仅指计算梯度的方法，而另一种算法，如随机梯度下降，用于使用该梯度执行学习。

—第 204 页，[深度学习](https://amzn.to/2NJW3gE)，2016。

随机梯度下降可用于为其他机器学习算法(如线性回归)寻找参数，并且在处理非常大的数据集时使用，尽管如果有足够的资源，那么基于凸的优化算法的效率明显更高。

## 学习算法的组成部分

使用具有反向传播的随机梯度下降来训练深度学习神经网络模型包括选择多个组件和超参数。在本节中，我们将依次看一看每一个。

必须选择一个误差函数，通常称为目标函数、成本函数或损失函数。通常，选择一个特定的概率推理框架，称为最大似然。在这个框架下，通常选择的损失函数是分类问题的交叉熵和回归问题的均方误差。

*   **损耗功能**。用于估计模型表现的函数，该模型在来自训练数据集的示例上具有一组特定的权重。

搜索或优化过程需要一个开始模型更新的起点。起点由初始模型参数或权重定义。由于误差面是非凸的，优化算法对初始起点敏感。因此，选择小的随机值作为初始模型权重，尽管可以使用不同的技术来选择这些值的比例和分布。这些技术被称为“*权重初始化*”方法。

*   **重量初始化**。在训练过程开始时，将初始小随机值分配给模型权重的过程。

更新模型时，必须使用训练数据集中的大量示例来计算模型误差，通常简称为“*损失*”可以使用训练数据集中的所有示例，这可能适用于较小的数据集。或者，可以使用单个示例，这可能适用于流传输示例或数据经常变化的问题。可以使用混合方法，其中可以选择来自训练数据集的示例数量，并将其用于估计误差梯度。实例数量的选择被称为批量。

*   **批量**。在更新模型参数之前用于估计误差梯度的示例数。

一旦估计了误差梯度，就可以计算误差的导数并用于更新每个参数。训练数据集中和误差梯度估计中可能存在统计噪声。此外，模型的深度(层数)和模型参数单独更新的事实意味着，很难精确计算每个模型参数的变化量，以便最好地沿着误差梯度向下移动整个模型。

相反，每次迭代都会执行一小部分权重更新。一个名为“*学习率*”的超参数控制模型权重的更新量，进而控制模型在训练数据集上的学习速度。

*   **学习率**:学习算法每个周期每个模型参数更新的量。

训练过程必须重复多次，直到发现一组好的或足够好的模型参数。过程的总迭代次数由训练数据集的完整遍数限定，在此之后训练过程终止。这被称为训练次数“*时代*”

*   **时代**。在训练过程终止之前，通过训练数据集的完整次数。

学习算法有许多扩展，尽管这五个超参数通常控制深度学习神经网络的学习算法。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

*   [深度学习](https://amzn.to/2NJW3gE)，2016 年。
*   [神经锻造:前馈人工神经网络中的监督学习](https://amzn.to/2S8qRdI)，1999。
*   [用于模式识别的神经网络](https://amzn.to/2S8qdwt)，1995。

## 摘要

在这篇文章中，你发现了为深度学习神经网络寻找模型参数的挑战。

具体来说，您了解到:

*   神经网络学习从输入到输出的映射函数，可以概括为解决函数逼近问题。
*   与其他机器学习算法不同，神经网络的参数必须通过解决具有许多好的解和许多误导性好的解的非凸优化问题来找到。
*   随机梯度下降算法用于解决优化问题，其中使用反向传播算法在每次迭代中更新模型参数。

你有什么问题吗？
在下面的评论中提问，我会尽力回答。