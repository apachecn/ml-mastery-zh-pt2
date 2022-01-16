# Weka 中解决机器学习问题的模板

> 原文：<https://machinelearningmastery.com/template-for-working-through-machine-learning-problems-in-weka/>

最后更新于 2019 年 8 月 22 日

当你在 Weka 开始时，你可能会感到不知所措。

有如此多的数据集、如此多的过滤器和如此多的算法可供选择。

选择太多了。你可以做的事情太多了。

[![Too much Choice](img/14c0e33d22ea16c8bf1a218040f3bdf5.png)](https://machinelearningmastery.com/wp-content/uploads/2014/03/choice.jpg)

太多选择
图片由[埃米利奥拉布拉多](https://www.flickr.com/photos/3059349393/3786855827/sizes/l/)提供，保留部分权利。

结构化流程是关键。

我已经[谈到了过程](https://machinelearningmastery.com/4-steps-to-get-started-in-machine-learning/ "4-Steps to Get Started in Machine Learning: The Top-Down Strategy for Beginners to Start and Practice")和需要像[抽查算法](https://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/ "Why you should be Spot-Checking Algorithms on your Machine Learning Problems")这样的任务来克服压倒性的优势，并开始学习关于你的问题的有用的东西。在这篇文章中，我想给你一个这个过程的简化版本，你可以用来练习应用机器学习。

**用我的新书[用 Weka](https://machinelearningmastery.com/machine-learning-mastery-weka/) 启动你的项目**，包括*的分步教程*和清晰的*截图*所有示例。

## 问题解决模板

这个模板是一个简化的过程，它专注于学习问题，一个好的解决方案，并且非常快地完成。

它被组织成应用机器学习的六个步骤。通过使用[Weka 探索者](https://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/ "How to Run Your First Classifier in Weka")和[Weka 实验者](https://machinelearningmastery.com/design-and-run-your-first-experiment-in-weka/ "Design and Run your First Experiment in Weka")图形用户界面，每一步都被分解成特定的问题供您回答。

该流程的六个步骤及其目标如下:

1.  问题定义
2.  数据分析
3.  数据准备
4.  评估计法
5.  改善结果
6.  呈现结果

在接下来的部分中，我将总结流程中每一步的关键问题和答案。您可能希望打印出这些问题或将其复制到文档中，以创建自己的模板文档。

## 1.问题定义

问题定义的目的是理解并清楚地描述正在解决的问题。

### 问题描述

1.  这个问题的非正式描述是什么？
2.  问题的正式描述是什么？
3.  你对这个问题有什么假设？

### 提供的数据

1.  对选择数据施加了哪些限制？
2.  在提供的数据集中定义每个属性。

## 2.数据分析

数据分析的目的是了解可用于开发模型的信息。

[![Attribute Histograms](img/035f27316a68e09ce1194833b5bac2ad.png)](https://machinelearningmastery.com/wp-content/uploads/2014/02/histograms.png)

显示类别值的属性直方图

1.  属性是什么数据类型？
2.  是否有丢失或损坏的值？
3.  回顾属性的分布，你注意到了什么？
4.  回顾类值的分布，你注意到了什么？
5.  查看直方图中带有类值的属性分布，您会注意到什么？
6.  回顾属性的成对散点图，你会注意到什么？

## 3.数据准备

数据准备的目的是发现和公开数据集中的结构。

1.  规范化数据集
2.  标准化数据集
3.  将数据集平方
4.  离散化属性(如果是整数)
5.  移除和/或替换缺失的值(如果存在)
6.  创建数据集的转换，以测试问题定义中提出的假设

## 4.评估计法

评估计法的目的是开发测试工具和基线准确率，并以此为基础进行改进。

[![Algorithm ranking when analyzing results in the Weka Experimenter](img/cdab3456f0be64680cff89d02b8213f7.png)](https://machinelearningmastery.com/wp-content/uploads/2014/02/Screen-Shot-2014-02-25-at-5.30.54-AM.png)

在 Weka 实验器中分析结果时的算法排名

1.  探索不同的分类算法
2.  设计并运行抽查实验
3.  回顾并解释算法排名
4.  检查并解释算法的准确性
5.  根据需要重复该过程

## 5.改善结果

改进结果的目的是利用结果来开发更精确的模型。

### 算法调整

1.  探索不同的算法配置
2.  设计并运行算法调优实验
3.  回顾并解释算法排名
4.  检查并解释算法的准确性
5.  根据需要重复该过程

### 集成方法

1.  探索不同的集成方法
2.  设计并运行算法集成实验
3.  回顾和解读集成排名
4.  回顾和解释集合的准确性
5.  根据需要重复该过程
6.  你能用其他元算法来改善结果吗，比如阈值？
7.  你能通过使用与表现良好的算法在同一家族中的其他算法来提高结果吗？

## 6.呈现结果

呈现结果的目的是描述问题和解决方案，以便第三方能够理解。

完成以下部分，总结问题和解决方案。

1.  有什么问题？
2.  解决方案是什么？
3.  有什么发现？
4.  有哪些限制？
5.  结论是什么？

## 如何使用

在 Weka 安装的“*目录中有许多有趣的数据集。 [UCI 机器学习资源库](https://archive.ics.uci.edu/ml/) 上也有很多数据集可以下载使用。*

 *选择一个问题，并使用此模板解决它。你会惊讶于你学到了多少，像这样的结构化过程能帮助你保持专注。

## 摘要

在这篇文章中，你了解了一个用于应用机器学习过程的结构化模板。该模板可以打印出来，并逐步用于解决 Weka 机器学习工作台中的一个问题。

随着问题的展开，回答模板每一步中的具体问题将快速建立对问题和您的解决方案的更深入理解。这是无价的，就像实验室里的科学家笔记本。*