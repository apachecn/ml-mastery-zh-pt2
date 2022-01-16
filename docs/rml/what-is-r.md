# 什么是 R

> 原文：<https://machinelearningmastery.com/what-is-r/>

最后更新于 2019 年 8 月 22 日

r 也许是统计编程和应用机器学习最强大和最受欢迎的平台之一。

当你认真对待机器学习时，你会找到进入 r 的方法。

在这篇文章中，你会发现 R 是什么，它来自哪里，以及它的一些最重要的特性。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

[![R Logo](img/77cd5adda522d8b151b1d4fbb40c44b9.png)](https://machinelearningmastery.com/wp-content/uploads/2014/08/R-Logo.jpg)

r 很快

## 什么是 R？

r 是一个用于统计编程和可视化的开源环境。

r 是很多东西，刚开始可能会很混乱。

*   **R 是计算机语言**。它是 Lisp 的变体，你可以用它编写程序。
*   **R 是解释者**。它可以解析和执行直接键入或从扩展名为. R 的文件中加载的 R 脚本(程序)。
*   **R 为平台**。它可以创建图形显示在屏幕上或保存到文件中。它还可以准备可以查询和更新的模型。

您可能希望在文件中编写 R 脚本，并使用 R 解释器以批处理模式运行它们，以获得表格或图形等结果。您可能希望打开 R 解释器并键入命令来加载数据，以特定的方式探索和建模数据。

有图形环境，但最简单和最常见的使用是从 R 控制台(像一个 [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop) )。如果你刚开始学习 R，我建议你在控制台上学习 R。

## R 从何而来

r 是由罗斯·伊哈卡和罗伯特·绅士在新西兰奥克兰大学创建的，作为 T2 编程语言 T3 的实现。发展始于 1993 年。1995 年，在 GNU 图形处理器下发布的文件传输协议上提供了一个版本。更大的核心小组和开源项目成立于 1997 年。

它最初是作者的一个实验，使用类似于 S 中提供的语法在 Lisp 中实现一个统计测试平台。随着它的发展，它采用了更多的 S 的语法和特性，最终在能力和范围上超越了它。

关于 R 的历史有趣而详细的介绍，请查看技术报告 [R:过去和未来的历史](https://www.stat.auckland.ac.nz/~ihaka/downloads/Interface98.pdf) (PDF)。

## R 的主要特征

当您需要分析数据、绘制数据图或为数据建立统计模型时，r 是一个可以使用的工具。它非常适合一次性分析原型和学术工作，但不适合构建要部署在可扩展或操作环境中的模型。

### 重组的好处

R 有三个主要好处:

*   **开源** : R 是自由开源的。您可以立即下载并开始免费使用。您可以阅读源代码，从中学习并修改它以满足您的需求。简直太神奇了。
*   **Packages** : R 之所以受欢迎，是因为它有大量非常强大的算法被实现为称为 Packages 的第三方库。统计领域的学者以 R 包的形式发布他们的方法是很常见的，这意味着你可以直接访问一些最先进的方法。
*   **成熟度** : R 的灵感来源于专有统计语言 S，使用并改进了统计计算中有用的习惯用法和隐喻，比如在矩阵、向量和数据帧中工作。

更多关于 R 包的信息，请查看 [CRAN](https://cran.r-project.org/) (综合 R 档案网。)并通过[包](https://cran.r-project.org/web/packages/available_packages_by_name.html)或[视图](https://cran.r-project.org/web/views/)进行浏览。列出机器学习软件包的[机器学习&统计学习](https://cran.r-project.org/web/views/MachineLearning.html)视图将会非常有趣。

### R 的困难

该平台的三个主要困难是:

*   **不一致性**:每个算法都是用自己的参数、命名约定和参数来实现的。有些人试图坚持粗略的约定(比如预测函数进行预测)，但即使是标准函数名的结果也可能在复杂性上有所不同。这可能非常令人沮丧，需要深入阅读您使用的每个新包的文档。
*   **文档**:文档很多，但一般比较直接简洁。内置的帮助很少对您的需求有足够的帮助，它不断地驱使您到网络上获取完整的工作示例，您必须从这些示例中获得您的用例。
*   **可扩展性** : R 用于可放入一台机器内存的数据。它不适用于流数据、大数据或跨多台机器工作。

这种语言有点迟钝，但作为一名程序员，你会毫不费力地学会它，并根据自己的需要修改例子。许多包利用用 C、C++、FORTRAN 和 Java 编写的数学代码，在 R 环境中提供了一个纵容接口。

## 谁在用 R？

商业公司现在支持 R。例如，Revolution R 是一个商业支持的 R 版本，其扩展对企业有用，例如集成开发环境。甲骨文、IBM、Mathematica、MATLAB、SPSS、SAS 等提供与 R 及其平台的集成。

Revolution Analytics 博客还提供了一长串公开宣布采用该平台的公司名单。

数据科学竞赛的[卡格尔平台](http://blog.kaggle.com/2011/11/27/kagglers-favorite-tools/)和[kd 掘金民调](http://www.kdnuggets.com/polls/2013/languages-analytics-data-mining-data-science.html)都指出 R 是成功实践数据科学家最受欢迎的平台。在文章[机器学习的最佳编程语言](https://machinelearningmastery.com/best-programming-language-for-machine-learning/ "Best Programming Language for Machine Learning")中了解更多信息。

## 摘要

在这篇文章中，你得到了 R 是什么的概述，它的关键特性，它来自哪里，谁在使用它。

这是一个令人兴奋和强大的平台。如果你正在考虑开始使用 R 进行应用机器学习，你可能想看看这个关于 R 进行机器学习的 7 本书的[列表](https://machinelearningmastery.com/books-for-machine-learning-with-r/ "Books for Machine Learning with R")。

更多关于 R 的信息，请查看[统计计算的 R 项目](https://www.r-project.org/)主页。在那里你可以找到下载链接、文档和手册、电子邮件列表等等。