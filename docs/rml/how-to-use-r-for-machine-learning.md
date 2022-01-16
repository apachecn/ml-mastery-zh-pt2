# 如何将 R 用于机器学习

> 原文：<https://machinelearningmastery.com/how-to-use-r-for-machine-learning/>

最后更新于 2019 年 8 月 22 日

r 有一大堆包，哪些最适合你的机器学习项目？

在这篇文章中，你将发现机器学习旅程中为每个子任务推荐的确切的 R 函数和包。

这很有用。为这一页添加书签。我相信你会一次又一次地回来查看。

如果你是 R 用户，知道更好的方法，在评论里分享，我会更新列表。

**用我的新书[用 R](https://machinelearningmastery.com/machine-learning-with-r/) 启动你的项目**，包括*一步一步的教程*和所有例子的 *R 源代码*文件。

我们开始吧。

![How To Use R For Machine Learning](img/a1b370e34760a15f2ecf90ad60c1b712.png)

尼尔·卡明斯摄，版权所有。

## 你应该使用什么样的 R 包？

R 有超过 6000 个第三方软件包。大量可用的软件包是 R 平台的优势之一。这也是挫折。

应该使用哪些软件包？

作为机器学习项目的一部分，您需要执行一些特定的任务。加载数据、评估计法和提高准确性等任务。您可以为每个任务使用多种技术，多个包可以提供这些技术。

考虑到有这么多不同的方法来完成一个给定的子任务，你需要发现那些最能满足你需求的函数和包。

## 将同类最佳的包映射到项目任务

解决这个问题的方法是创建一个机器学习项目期间您可能要处理的所有子任务的映射，并找到您可以使用的最佳包和函数。

首先列出机器学习项目中的所有子任务。你可以仔细看看应用机器学习的[流程](https://machinelearningmastery.com/process-for-working-through-machine-learning-problems/)和[机器学习项目检查表](https://machinelearningmastery.com/machine-learning-checklist/)。

考虑到 R 是一种统计语言，它提供了许多可用于数据分析的工具，以及可用于训练和生成预测的预测模型。

使用您最喜欢的搜索引擎，您可以找到包中的所有包和功能，您可以使用它们来完成每个任务。这可能是详尽无遗的，你可以最终有许多不同的候选解决方案。

你需要将每个选项列表缩减为完成任务的一种首选方式。你可以对每一个进行实验，看看什么对你有用。你也可以仔细检查你的搜索结果，梳理出从业者最常用的功能。

接下来是从 R 包和函数到机器学习项目任务的映射，您可以使用它来开始使用 R 进行机器学习。

## 如何将 R 用于机器学习项目

本节列出了通用机器学习项目的许多主要子任务。每个任务都列出了您可以在 R 中用来完成任务的特定函数和父包。

所选函数的一些属性如下:

*   **最小值**:列表是一个项目中机器学习任务的最低值，也是你可以使用的函数和包名的最低值。实际使用列出的每个功能需要更多的作业。
*   **简单**:选择函数是为了简单地为任务提供直接结果。一个函数优于多个函数调用。
*   **偏好**:功能是根据我的偏好和最佳估计选择的。其他从业者可能有不同的选择(分享在评论里！).

任务分为三大组:

1.  用于为建模准备数据的数据准备任务。
2.  评估赛车的算法任务和评估预测建模算法。
3.  改进结果任务，从表现良好的算法中获得更多。

### 1.数据准备任务

#### 数据加载

从文件中加载数据集。

*   csv:从 utils 包中读取. CSV 函数

#### 数据清理

清理数据集以确保数据合理且一致，为分析和建模做好准备。

*   输入:从 [Hmisc](https://cran.r-project.org/web/packages/Hmisc/index.html) 包输入。
*   异常值:来自[异常值](https://cran.r-project.org/web/packages/outliers/index.html)包的各种函数。
*   重新平衡:从 [DMwR](https://cran.r-project.org/web/packages/DMwR/index.html) 包中删除功能。

#### 数据汇总

使用描述性统计数据汇总数据集。

*   汇总分配:基本包的汇总功能。
*   总结相关性:统计包中的 cor 函数

#### 数据可视化

可视化总结数据集。

*   散点图矩阵:图形包中的成对函数。
*   直方图:来自图形包的 hist 函数。
*   密度图:来自[点阵](https://cran.r-project.org/web/packages/lattice/index.html)包的密度图功能。
*   方块和触须绘图:图形包中的方块绘图功能

荣誉提名:

*   ggpairs 功能来自 [GGally](https://cran.r-project.org/web/packages/GGally/index.html) 包，可以在一个图中完成所有操作
*   [ggplot2](https://cran.r-project.org/web/packages/ggplot2/index.html) 和[点阵](https://cran.r-project.org/web/packages/lattice/index.html)包通常非常适合绘图

#### 特征选择

选择数据集中与构建预测模型最相关的要素。

*   RFE:来自[Caret](https://cran.r-project.org/web/packages/caret/index.html)包的 rfe 函数
*   相关:从[Caret](https://cran.r-project.org/web/packages/caret/index.html)包中找到相关函数

Caret 包提供了一套功能选择方法，请参见评估计法任务。

荣誉提名:

*   [FSelector](https://cran.r-project.org/web/packages/FSelector/index.html) 包。

#### 数据转换

创建数据集的转换，以便向学习算法最好地展示问题的结构。

*   规范化:自定义编写的函数
*   标准化:从基础包扩展功能。

Caret 包作为测试工具的一部分提供数据转换，请参见下一节。

### 2.评估计法任务

caret 包中的函数应该用于评估数据集上的模型。

caret 包支持各种表现度量和测试选项，如数据拆分和交叉验证。预处理也可以配置为测试线束的一部分。

#### 模型评估

*   模型评估:来自[Caret](https://cran.r-project.org/web/packages/caret/index.html)包的训练功能。
*   测试选项:来自[Caret](https://cran.r-project.org/web/packages/caret/index.html)包的列车控制功能。
*   预处理选项:从[Caret](https://cran.r-project.org/web/packages/caret/index.html)包中的预处理功能。

请注意，许多现代预测模型(如高级决策树的风格)提供了某种形式的特征选择、参数调整和内置集成。

#### 预测模型

Caret 包提供了对所有最佳预测建模算法的访问。

### 3.改进结果任务

从表现良好的模型中获取最大利益的技术，以服务于做出准确的预测。

#### 算法调整

caret 包提供算法调整，作为测试工具的一部分，并包括随机、网格和自适应搜索等技术。

#### 模型集合

许多现代预测建模算法提供内置的集成。caret 包中提供了一套装袋和增强功能。

*   混合:来自[carestenmble](https://cran.r-project.org/web/packages/caretEnsemble/index.html)包的 carestenmble。
*   堆叠:从[carestensemble](https://cran.r-project.org/web/packages/caretEnsemble/index.html)包装中取出。
*   装袋:从[到](https://cran.r-project.org/web/packages/ipred/index.html)包装的装袋功能。

## 摘要

在这篇文章中，你发现使用 R 进行机器学习的最好方法是将特定的 R 函数和包映射到机器学习项目的任务上。

您发现了可用于机器学习项目最常见任务的特定包和函数，包括指向进一步文档的链接。

## 你的下一步

开始使用 R 进行机器学习。在你当前或下一个机器学习项目中使用以上建议。

我错过了一个重要的包裹吗？我错过了机器学习项目中的一个关键任务吗？留下评论，让我知道我错过了什么。

你有问题吗？给我发邮件或留言。