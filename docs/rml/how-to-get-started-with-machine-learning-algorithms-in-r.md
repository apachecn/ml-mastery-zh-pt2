# 如何在 R 中入门机器学习算法

> 原文：<https://machinelearningmastery.com/how-to-get-started-with-machine-learning-algorithms-in-r/>

最后更新于 2019 年 8 月 22 日

r 是[最受欢迎的应用机器学习平台](https://machinelearningmastery.com/best-programming-language-for-machine-learning/ "Best Programming Language for Machine Learning")。当你想认真对待应用机器学习的时候，你会发现你的方法[变成了](https://machinelearningmastery.com/what-is-r/ "What is R")。

它非常强大，因为提供了这么多机器学习算法。一个问题是算法都是由第三方提供的，这使得它们的使用非常不一致。这让你慢了很多，因为你必须一次又一次地学习如何对数据建模，以及如何用每个包中的每个算法进行预测。

在这篇文章中，你将发现如何用 R 语言中的机器学习算法克服这个困难，用遵循一致结构的预先准备的秘籍。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

## 算法多，一致性差

R 生态系统是巨大的。开源第三方包提供了这种能力，让学者和专业人士能够将最强大的算法交到美国从业者手中。

我从 R 开始时遇到的一个问题是，每个算法的用法因包而异。这种不一致也延伸到了文档中，一些文档提供了有效的分类示例，但忽略了回归，而另一些文档则根本没有提供示例。

所有这一切意味着，如果你想尝试不同包中的一些不同算法，你必须花时间弄清楚如何依次用每种方法进行拟合和预测。这需要很多时间，尤其是有不稳定的例子和小插曲。

我将这些困难总结如下:

*   **不一致**:算法实现在模型适合数据的方式和模型用于生成预测的方式上有所不同。这意味着你必须研究每一个包和每一个算法实现，仅仅是为了把一个有效的例子放在一起，更不用说让它适应你的问题了。
*   **去中心化**:算法跨不同的包实现，可能很难定位哪个包提供了你需要的算法的实现，更不用说哪个包提供了最流行的实现。此外，一个包的文档可能分布在多个帮助文件、网站和简介中。这意味着你必须做大量的搜索来定位一个算法，更不用说编译一个算法列表供你选择。
*   **不完整**:算法文档几乎总是部分完整的。可能会也可能不会提供示例用法，如果提供了，可能会也可能不会在规范问题上演示。这意味着您没有明显的方法来快速理解如何使用实现。
*   **复杂度**:算法在实现和描述的复杂度上各不相同。当你从一个包裹跳到另一个包裹时，这会让你付出代价。你想专注于如何从算法及其参数中获得最大收益，而不是为了得到一个你好的世界而耗费精力去解析大量的 pdf。

## 构建算法秘籍

如果你有一本算法秘籍，你可以查找并在 R 中找到机器学习算法的例子，你可以复制粘贴并适应你的特定问题，你可以做得更多。

为此，秘籍方法的工作，它将必须确认一些关键原则:

*   **独立**:每个代码示例都必须是独立的、完整的并且可以执行。
*   **只是代码**:每一个配方都必须专注于对机器学习理论阐述最少的代码(这方面有很棒的书，不要把这些顾虑混在一起)。
*   **简单性**:每个菜谱都必须以最常见的用例来呈现，这大概就是你在查找的时候想要做的事情。您想查阅官方文档只是为了查找参数，以便从算法中获得最大收益。
*   **便携**:所有秘籍必须提供在一个单一的参考，可以搜索和打印，浏览和查找(一本秘籍书)。
*   **一致**:所有代码示例都是一致呈现的，遵循相同的代码结构和风格约定(加载数据、拟合模型、进行预测)。

一本算法秘籍会让你有能力运用 R 平台进行机器学习，解决复杂的问题。

*   你可以直接应用算法和特性。
*   你可以找到你需要的代码。
*   你一眼就能明白发生了什么。
*   你可以拥有秘籍，用你想要的方式使用和组织它们。
*   你可以充分利用算法和特性。

## R 中的算法配方

我已经列出了这些秘籍的例子。

我在 R 中提供了示例机器学习方法，按算法类型或相似性分组，如下所示:

*   [线性回归](https://machinelearningmastery.com/linear-regression-in-r/ "Linear Regression in R"):普通最小二乘回归、逐步回归、主成分回归、偏最小二乘回归。
*   [惩罚线性回归](https://machinelearningmastery.com/penalized-regression-in-r/ "Penalized Regression in R"):岭回归、最小绝对收缩和选择算子(LASSO)和弹性
*   [非线性回归](https://machinelearningmastery.com/non-linear-regression-in-r/ "Non-Linear Regression in R"):多元自适应回归 spins(MARS)、支持向量机(SVM)、k 近邻(kNN)和神经网络。
*   [非线性决策树回归](https://machinelearningmastery.com/non-linear-regression-in-r-with-decision-trees/ "Non-Linear Regression in R with Decision Trees"):分类回归树(CART)、条件决策树、模态树、规则系统、装袋 CART、随机森林、梯度提升机(GBM)和立体派。
*   [线性分类](https://machinelearningmastery.com/linear-classification-in-r/ "Linear Classification in R") : Logistic 回归、线性判别分析(LDA)和偏最小二乘判别分析。
*   [非线性分类](https://machinelearningmastery.com/non-linear-classification-in-r/ "Non-Linear Classification in R"):混合判别分析(MDA)、二次判别分析(QDA)、正则化判别分析(RDA)、神经网络、柔性判别分析(FDA)、支持向量机(SVM)、k 近邻(kNN)和朴素贝叶斯。
*   [非线性决策树分类](https://machinelearningmastery.com/non-linear-classification-in-r-with-decision-trees/ "Non-Linear Classification in R with Decision Trees"):分类回归树(CART)、C4.5、PART、装袋 CART、随机森林、梯度提升机(GBM)、增强 C5.0

我认为这些秘籍真的符合这个任务的要求。

## 摘要

在这篇文章中，你发现了机器学习在 R 中的流行和力量，但这种力量的代价是驾驭它所需的时间。

您发现，解决 R 中这一限制的一种方法是设计一个完整的独立机器学习算法的秘籍，您可以根据需要查找并应用于您的特定问题。

最后，您在 R 中看到了各种算法类型的机器学习算法配方示例。

如果你觉得这种方法有用，我很想听听。