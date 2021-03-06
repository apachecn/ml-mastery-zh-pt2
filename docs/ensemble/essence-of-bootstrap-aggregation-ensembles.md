# 自举聚合集成的本质

> 原文：<https://machinelearningmastery.com/essence-of-bootstrap-aggregation-ensembles/>

最后更新于 2021 年 10 月 22 日

**Bootstrap 聚合**，或 bagging，是一种流行的集成方法，它在训练数据集的不同 Bootstrap 样本上拟合决策树。

实现起来很简单，对各种各样的问题都很有效，重要的是，该技术的适度扩展产生了集成方法，这些集成方法是一些最强大的技术之一，如随机森林，在各种各样的预测建模问题上表现良好。

因此，我们可以将 bagging 方法推广到集成学习的框架中，并且比较和对比属于方法的“***【bagging 家族】*** ”的一套常见集成方法。我们还可以使用这个框架来探索进一步的扩展，以及如何根据项目数据集或选择的预测模型进一步定制该方法。

在本教程中，您将发现机器学习集成的引导聚合方法的本质。

完成本教程后，您将知道:

*   基于自举样本和决策树的机器学习 bagging 集成方法。
*   如何从装袋方法中提取基本元素，以及像随机森林这样的流行扩展如何与装袋直接相关。
*   如何通过为方法的基本要素选择新的程序来设计装袋的新扩展？

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Essence of Bootstrap Aggregation Ensembles](img/a1a4b23f84851b7cda3e679cbd6863cf.png)

自举聚合集成精华
图片由 [GPA 图片档案](https://www.flickr.com/photos/iip-photo-archive/49912883128/)提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  自举聚合
2.  装袋集成的本质
3.  装袋集成家族
    1.  随机子空间集成
    2.  随机森林集合
    3.  额外树木组合
4.  定制装袋集成

## 自举聚合

Bootstrap Aggregation，简称[装袋](https://machinelearningmastery.com/bagging-ensemble-with-python/)，是一种集成机器学习算法。

该技术包括为每个集成成员创建训练数据集的自举样本，并在每个样本上训练决策树模型，然后使用类似预测平均值的统计量直接组合预测。

> Breiman 的 bagging(Bootstrap Aggregation 的缩写)算法是最早、最简单、但有效的基于集成的算法之一。

—第 12 页，[集成机器学习](https://amzn.to/2C7syo5)，2012。

训练数据集的样本是使用[自举方法](https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/)创建的，该方法包括随机选择样本并进行替换。

替换意味着相同的示例被隐喻性地返回到候选行池，并且可以在训练数据集的任何单个样本中被再次或多次选择。对于一些引导样本，训练数据集中的一些示例也可能根本没有被选择。

> 一些原始示例出现不止一次，而一些原始示例不在示例中。

—第 48 页，[集合方法](https://amzn.to/2XZzrjG)，2012。

bootstrap 方法的预期效果是使数据集的每个样本都非常不同，或者对于创建集合来说非常不同。

然后对每个数据样本进行决策树拟合。考虑到训练数据集中的差异，每棵树都会有所不同。通常，决策树被配置为可能具有增加的深度或者不使用修剪。这可以使每棵树对训练数据集更加专门化，进而进一步增加树之间的差异。

树的差异是可取的，因为它们将增加集成的“*多样性”*，这意味着产生在它们的预测或预测误差中具有较低相关性的集成成员。人们普遍认为，由技巧性和多样性(以不同方式技巧或犯不同错误)的集成成员组成的集成表现更好。

> 通过在其上训练每个分类器的自举副本内的变化，以及通过使用相对较弱的分类器来确保集合中的多样性，所述相对较弱的分类器的决策边界相对于训练数据中相对较小的扰动而可测量地变化。

—第 12 页，[集成机器学习](https://amzn.to/2C7syo5)，2012。

装袋的一个好处是，它通常不会过度填充训练数据集，并且集成成员的数量可以继续增加，直到保持数据集的表现停止提高。

这是 bagging 集成方法的高级总结，但是我们可以概括该方法并提取基本元素。

## 装袋集成的本质

装袋的本质是利用独立的模型。

这样的话，可能是最接近大众*智慧*比喻的实现，尤其是考虑到随着独立贡献者的加入，表现不断提升。

不幸的是，我们不能开发真正独立的模型，因为我们只有一个训练数据集。相反，bagging 方法使用随机性逼近独立模型。具体来说，通过在用于训练每个模型的数据集的采样中使用随机性，迫使模型之间具有某种半独立性。

> 虽然实际上不可能获得真正独立的基础学习器，因为它们是从相同的训练数据集生成的，但是通过在学习过程中引入随机性，可以获得依赖性较小的基础学习器，并且集成可以期望良好的泛化能力。

—第 48 页，[集合方法](https://amzn.to/2XZzrjG)，2012。

装袋程序的结构可分为三个基本要素；它们是:

*   **不同的训练数据集**:为每个集成模型创建不同的训练数据集样本。
*   **高方差模型**:在训练数据集的每个样本上训练相同的高方差模型。
*   **平均预测**:使用统计数据组合预测。

我们可以将规范装袋方法映射到这些元素上，如下所示:

*   **不同的训练数据集** : Bootstrap 样本。
*   **高方差模型**:决策树。
*   **平均预测**:回归平均值，分类模式。

这提供了一个框架，我们可以在其中为模型的每个基本元素考虑替代方法。

例如，我们可以将算法更改为另一种高方差技术，这种技术的学习行为有些不稳定，可能就像 k 超参数值适中的 k 近邻。

> 通常，装袋产生的组合模型优于使用原始数据的单个实例构建的模型。[……]这是真的，尤其是对于不稳定的诱导剂，因为装袋可以消除它们的不稳定性。在这种情况下，如果学习集中的扰动可以在构建的分类器中产生显著变化，则诱导子被认为是不稳定的。

—第 28 页，[使用集成方法的模式分类](https://amzn.to/2zxc0F7)，2010。

我们也可以将采样方法从引导改为另一种采样技术，或者更一般地说，完全不同的方法。事实上，这是文献中描述的 bagging 的许多扩展的基础。具体来说，就是尝试让集成成员更加独立，同时保持技巧。

> 我们知道独立基础学习器的组合将导致错误的急剧减少，因此，我们希望让基础学习器尽可能独立。

—第 48 页，[集合方法](https://amzn.to/2XZzrjG)，2012。

让我们仔细看看可能被认为是 bagging 系列的一部分的其他集成方法。

## 装袋集成家族

许多集成机器学习技术可以被认为是 bagging 的后代。

因此，我们可以将它们映射到我们的基本装袋框架中。这是一个有用的练习，因为它既强调了方法之间的差异，也强调了每种技术的独特性。也许更重要的是，它还可以激发您在自己的预测建模项目中想要探索的其他变体的想法。

让我们仔细看看与装袋相关的三种更流行的集成方法。

### 随机子空间集成

随机子空间方法，或随机子空间集成，包括为每个集成成员选择训练数据集中特征(列)的随机子集。

每个训练数据集都有所有行，因为它只是随机采样的列。

*   **不同的训练数据集**:随机抽取样本列。

### 随机森林集合

[随机森林](https://machinelearningmastery.com/random-forest-ensemble-in-python/)方法可能是最成功和最广泛使用的集成方法之一，因为它易于实现，并且在广泛的预测建模问题上通常具有优异的表现。

该方法通常包括选择训练数据集的自举样本和小的随机列子集，以便在选择每个集成成员中的每个分割点时考虑。

以这种方式，它就像是 bagging 和随机子空间方法的结合，尽管随机子空间是唯一用于决策树构造的方式。

*   **不同的训练数据集** : Bootstrap 样本。
*   **高方差模型**:随机列子集上有分割点的决策树。

### 额外树木组合

[额外树集合](https://machinelearningmastery.com/extra-trees-ensemble-with-python/)使用整个训练数据集，尽管它配置决策树算法来随机选择分割点。

*   **不同的训练数据集**:整个数据集。
*   **高方差模型**:随机分割点的决策树。

## 定制装袋集成

我们简要回顾了规范随机子空间、随机森林和额外树方法，尽管这些方法没有理由不能共享更多的实现细节。

事实上，像 bagging 和 random forest 这样的算法的现代实现证明了足够的配置来组合这些特性。

我们可以设计自己的扩展，映射到 bagging 框架中，而不是穷尽所有文献。这可能会启发您探索一种不太常见的方法，或者针对您的数据集或模型选择设计自己的装袋方法。

可能有几十个或几百个 bagging 扩展，对每个集成成员的训练数据集的准备方式或如何从训练数据集构建模型的细节稍作修改。

这些变化是围绕基本装袋方法的三个主要元素构建的，通常通过探索足够熟练的集成成员之间的平衡来寻求更好的表现，同时保持预测或预测误差之间足够的多样性。

例如，我们可以将训练数据集的采样改为无替换的随机采样，而不是自举采样。这就是所谓的*粘贴*

*   **不同的训练数据集**:行的随机子样本。

我们可以更进一步，为每个决策树选择一个随机的行子样本(比如粘贴)和一个随机的列子样本(随机子样本)。这就是所谓的“*随机补丁*”

*   **不同的训练数据集**:行和列的随机子样本。

我们也可以考虑自己对这个想法的简单扩展。

例如，通常使用[特征选择技术](https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/)来选择输入变量的子集，以降低预测问题的复杂性(更少的列)并获得更好的表现(更少的噪声)。我们可以想象一个 bagging 集成，其中每个模型都适合于由不同的特征选择或特征重要性方法选择的训练数据集的不同“*视图*”。

*   **不同训练数据集**:不同特征选择方法选择的列。

作为建模管道的一部分，用许多不同的[数据转换](https://machinelearningmastery.com/framework-for-data-preparation-for-machine-learning/)测试模型也是很常见的。之所以这样做，是因为我们无法事先知道训练数据集的哪种表示将最好地将数据集的未知底层结构暴露给学习算法。我们可以想象一个 bagging 集成，其中每个模型都适合训练数据集的不同变换。

*   **不同的训练数据集**:原始训练数据集的数据变换。

这些也许是几个显而易见的例子，说明如何探索装袋方法的本质，希望能启发更多的想法。我鼓励你集思广益，如何让这些方法适应你自己的特定项目。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 教程

*   [如何用 Python 开发装袋集成](https://machinelearningmastery.com/bagging-ensemble-with-python/)
*   [装袋和随机森林不平衡分类](https://machinelearningmastery.com/bagging-and-random-forest-for-imbalanced-classification/)
*   [如何在 Keras 创建深度学习模型的装袋集成](https://machinelearningmastery.com/how-to-create-a-random-split-cross-validation-and-bagging-ensemble-for-deep-learning-in-keras/)
*   [如何用 Python 实现从零开始装袋](https://machinelearningmastery.com/implement-bagging-scratch-python/)
*   [机器学习建模管道的温和介绍](https://machinelearningmastery.com/machine-learning-modeling-pipelines/)

### 书

*   [使用集成方法的模式分类](https://amzn.to/2zxc0F7)，2010。
*   [集成方法](https://amzn.to/2XZzrjG)，2012。
*   [集成机器学习](https://amzn.to/2C7syo5)，2012。

### 文章

*   [引导聚合，维基百科](https://en.wikipedia.org/wiki/Bootstrap_aggregating)。
*   [随机子空间法，维基百科](https://en.wikipedia.org/wiki/Random_subspace_method)。
*   [随机森林，维基百科](https://en.wikipedia.org/wiki/Random_forest)。

## 摘要

在本教程中，您发现了机器学习集成的引导聚合方法的本质。

具体来说，您了解到:

*   基于自举样本和决策树的机器学习 bagging 集成方法。
*   如何从装袋方法中提取基本元素，以及像随机森林这样的流行扩展如何与装袋直接相关。
*   如何通过为方法的基本要素选择新的程序来设计装袋的新扩展？

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。