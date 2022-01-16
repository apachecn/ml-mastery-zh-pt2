# 机器学习堆叠集成的本质

> 原文:[https://machinelearning master . com/堆叠本质-机器学习集成/](https://machinelearningmastery.com/essence-of-stacking-ensembles-for-machine-learning/)

最后更新于 2021 年 4 月 27 日

堆叠一般化，或**堆叠**，可能是一个不太受欢迎的机器学习集成，因为它描述的是一个框架，而不是一个特定的模型。

也许它在主流机器学习中不太受欢迎的原因是，在不遭受数据泄漏的情况下，正确训练堆叠模型可能会很棘手。这意味着这项技术主要被高技能专家在高风险环境中使用，如机器学习比赛，并被赋予新的名称，如混合合奏。

然而，现代机器学习框架使得堆叠例程能够实现和评估分类和回归预测建模问题。因此，我们可以通过堆叠框架的镜头来回顾与堆叠相关的集成学习方法。这种更广泛的堆叠技术家族也有助于了解未来在探索我们自己的预测建模项目时如何定制该技术的配置。

在本教程中，您将发现机器学习集成的堆叠泛化方法的本质。

完成本教程后，您将知道:

*   机器学习的堆叠集成方法使用元模型来组合来自贡献成员的预测。
*   如何从堆叠方法中提取基本元素，以及像混合和超级合奏这样的流行扩展是如何关联的。
*   如何通过为方法的基本元素选择新的过程来设计新的堆栈扩展。

**用我的新书[Python 集成学习算法](https://machinelearningmastery.com/ensemble-learning-algorithms-with-python/)启动你的项目**，包括*分步教程*和所有示例的 *Python 源代码*文件。

我们开始吧。

![Essence of Stacking Ensembles for Machine Learning](img/00965ab2e474e9c4b0a16cf38cc4764a.png)

机器学习堆叠套装精华
图片由[托马斯](https://www.flickr.com/photos/photommo/31459357916/)提供，保留部分权利。

## 教程概述

本教程分为四个部分；它们是:

1.  堆叠一般化
2.  堆叠集合的本质
3.  堆叠系综族
    1.  投票团
    2.  加权平均值
    3.  混合合奏
    4.  超级学习者乐团
4.  定制堆叠套装

## 堆叠一般化

[堆叠泛化](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)，简称堆叠，是一种集成机器学习算法。

堆叠包括使用机器学习模型来学习如何最好地组合来自贡献的集成成员的预测。

在[投票](https://machinelearningmastery.com/voting-ensembles-with-python/)中，集成成员通常是模型类型的不同集合，例如决策树、朴素贝叶斯和支持向量机。预测是通过平均预测进行的，例如选择投票最多的班级(统计模式)或总概率最大的班级。

> ……(未加权)投票只有在学习方案表现相当好的情况下才有意义。

—第 497 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2WehhJC)，2016。

投票的一个扩展是在预测中加权每个集合成员的贡献，提供加权和预测。这允许将更多的权重放在平均表现更好的模型上，而将较少的权重放在那些表现不佳但仍有一些预测技能的模型上。

必须了解分配给每个贡献成员的权重，例如每个模型在训练数据集或保持数据集上的性能。

堆叠概括了这种方法，并允许使用任何机器学习模型来学习如何最好地组合来自贡献成员的预测。组合预测的模型称为元模型，而集合成员称为基本模型。

> 投票的问题在于，不清楚该信任哪个分类器。堆叠试图学习哪些分类器是可靠的，使用另一种学习算法——元载体——来发现如何最好地组合基础学习者的输出。

—第 497 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2WehhJC)，2016。

在介绍该技术的论文中，基础模型被称为 0 级学习者，元模型被称为 1 级模型。

自然，模型的堆叠可以继续到任何期望的水平。

> 堆叠是一个通用的过程，在这个过程中，一个学习者被训练来组合各个学习者。在这里，个体学习者被称为一级学习者，而组合者被称为二级学习者或元学习者。

—第 83 页，[集合方法](https://amzn.to/2XZzrjG)，2012。

重要的是，元模型的训练方式不同于基础模型的训练方式。

元模型的输入是基础模型做出的预测，而不是数据集的原始输入。目标是相同的预期目标值。用于训练元模型的基础模型所做的预测是用于没有用于训练基础模型的例子，这意味着它们是不符合样本的。

例如，数据集可以分为训练数据集、验证数据集和测试数据集。然后，每个基础模型可以适合训练集，并在验证数据集上进行预测。来自验证集的预测然后被用于训练元模型。

这意味着当基础模型进行[样本外预测](https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/)时，元模型被训练为最佳组合基础模型的能力，例如在训练期间没有看到的例子。

> …我们保留一些实例来形成 1 级学习者的训练数据，并从剩余数据中构建 0 级分类器。一旦建立了 0 级分类器，它们就被用来对保持集中的实例进行分类，形成 1 级训练数据。

—第 498 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2WehhJC)，2016。

一旦训练了元模型，就可以在组合的训练和验证数据集上重新训练基础模型。然后，可以在测试集上对整个系统进行评估，方法是首先通过基础模型传递示例以收集基础级别的预测，然后通过元模型传递这些预测以获得最终预测。当对新数据进行预测时，该系统可以以同样的方式使用。

这种训练、评估和使用堆叠模型的方法可以进一步推广到 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)中。

通常，基础模型是使用不同的算法准备的，这意味着集合是模型类型的异构集合，为所做的预测提供了期望的多样性水平。然而，情况并非必须如此，可以使用相同模型的不同配置，或者在不同的数据集上训练相同的模型。

> 一级学习者通常是通过应用不同的学习算法产生的，因此，堆叠的集成通常是异构的

—第 83 页，[集合方法](https://amzn.to/2XZzrjG)，2012。

在分类问题上，当基础模型被配置为预测概率而不是简单的类标签时，堆叠集成通常表现得更好，因为预测中增加的不确定性为元模型在学习如何最好地组合预测时提供了更多的上下文。

> …大多数学习方案能够输出每个类别标签的概率，而不是进行单一的分类预测。通过使用概率形成 1 级数据，可以利用这一点来提高堆叠的性能。

—第 498 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2WehhJC)，2016。

元模型通常是简单的线性模型，如回归问题的线性回归或分类的逻辑回归模型。同样，情况也不一定如此，任何机器学习模型都可以用作元学习器。

> ……因为大部分工作已经由 0 级学习者完成，1 级分类器基本上只是一个仲裁器，为此选择一个相当简单的算法是有意义的。[……]简单的线性模型或叶子上有线性模型的树通常效果很好。

—第 499 页，[数据挖掘:实用机器学习工具与技术](https://amzn.to/2WehhJC)，2016。

这是堆叠集成方法的高级总结，但是我们可以概括该方法并提取基本元素。

## 堆叠集合的本质

堆叠的本质是学习如何组合贡献的合奏成员。

这样，我们可能会认为堆叠是假设简单的群体智慧(例如平均)是好的，但不是最优的，如果我们能够识别并给予群体中的专家更多的权重，就可以获得更好的结果。

专家和次要专家是根据他们在新情况下的技能来确定的，例如样本外数据。这是与简单平均和投票的一个重要区别，尽管它引入了一定程度的复杂性，使得该技术难以正确实现并避免[数据泄漏](https://machinelearningmastery.com/data-preparation-without-data-leakage/)，进而导致不正确和乐观的性能。

然而，我们可以看到堆叠是一种非常通用的集成学习方法。

广义地说，我们可以把集成模型的加权平均看作是对投票集成的推广和改进，而叠加则是加权平均模型的进一步推广。

因此，堆叠程序的结构可以分为三个基本要素；它们是:

*   **多样的集合成员**:创建一组多样的模型，做出不同的预测。
*   **成员考核**:考核合奏成员的表现。
*   **与模型结合**:使用模型结合成员的预测。

我们可以将规范堆叠映射到这些元素上，如下所示:

*   **不同的集成成员**:使用不同的算法来拟合每个贡献模型。
*   **成员评估**:评估模型在样本外预测上的表现。
*   **结合模型**:机器学习模型结合预测。

这提供了一个我们可以考虑相关集成算法的框架。

让我们仔细看看其他可能被认为是堆叠系列的一部分的集成方法。

## 堆叠系综族

许多集成机器学习技术可以被认为是堆叠的前身或后代。

因此，我们可以将它们映射到我们的基本堆叠框架上。这是一个有用的练习，因为它既强调了方法之间的差异，也强调了每种技术的独特性。也许更重要的是，它还可能激发出您想要在自己的预测建模项目中探索的其他变体的想法。

让我们仔细看看与堆叠相关的四种更常见的集成方法。

### 投票团

投票集成是最简单的集成学习技术之一。

投票集合通常包括使用不同的算法来准备每个集合成员，就像堆叠一样。不用学习如何组合预测，而是使用一个简单的统计数据。

在回归问题上，投票集合可以预测来自集合成员的预测的平均值或中值。对于分类问题，预测票数最多的标签，称为硬投票，或者预测获得最大和概率的标签，称为软投票。

与堆叠的重要区别在于，没有基于模型性能的模型称重。假设所有模型的平均技能水平相同。

*   **成员考核**:假设所有模特技术都一样。
*   **结合模型**:简单统计。

### 加权平均系综

加权平均可能被认为比整体投票高出一步。

像堆叠和投票集合一样，加权平均使用不同的模型类型集合作为贡献成员。

与投票不同，加权平均假设一些贡献成员比其他成员更好，并相应地对模型贡献进行加权。

最简单的加权平均集成基于每个模型在训练数据集上的性能对其进行加权。对这种天真方法的一个改进是根据每个成员在保留数据集上的表现来加权它，例如在 k 倍交叉验证期间的验证集或超倍预测。

下一步可能涉及使用优化算法调整每个模型的系数权重，以及保持数据集的性能。

加权平均模型的这些持续改进开始类似于原始叠加模型，其线性模型被训练来组合预测。

*   **成员评估**:成员在训练数据集上的表现。
*   **结合模型**:预测的加权平均值。

### 混合合奏

混合显然是一种具有特定配置的堆叠概括模型。

堆叠的一个限制是没有普遍接受的配置。这使得该方法对初学者具有挑战性，因为基本上任何模型都可以用作基础模型和元模型，并且任何重采样方法都可以用于为元模型准备训练数据集。

混合是一个特定的叠加集合，它会开出两个处方。

第一种是使用保持验证数据集来准备用于训练元模型的样本外预测。第二种是使用线性模型作为元模型。

这项技术诞生于从事机器学习竞赛的从业者的需求，该竞赛涉及大量基础学习者模型的开发，这些模型可能来自不同的来源(或团队)，这反过来可能计算量太大，并且太具挑战性，难以使用数据集的 k 重交叉验证分区进行协调验证。

*   **成员预测**:验证数据集中的样本外预测。
*   **结合模型**:线性模型(如线性回归或逻辑回归)。

考虑到混合系综的流行，叠加有时会专门指使用 k 倍交叉验证来准备元模型的样本外预测。

### 超级学习者乐团

像混合一样，超级系综是堆叠系综的特定配置。

超级学习中的元模型是使用 k 倍交叉验证期间收集的基础学习者的超倍预测准备的。

因此，我们可以将超级学习者集合视为混合的兄弟，主要区别在于如何为元学习者准备样本外预测的选择。

*   **多样的集成成员**:使用不同的算法和相同算法的不同配置。
*   **成员评估**:k 倍交叉验证的出格预测。

## 定制堆叠套装

我们回顾了规范叠加作为一个结合来自不同模型类型集合的预测的框架。

堆叠是一种广泛的方法，这可能会使它很难开始使用。我们可以看到投票集成和加权平均集成是堆叠方法的简化，混合集成和超级学习器集成是堆叠的特定配置。

这篇综述强调了不同堆叠方法的重点在于元模型的复杂性，例如使用统计学、加权平均或真正的机器学习模型。重点还在于元模型的训练方式，例如从验证数据集中的样本预测或 k 倍交叉验证。

堆叠的另一个探索领域可能是集合成员的多样性，而不仅仅是使用不同的算法。

与使用决策树规定的提升和打包相比，堆叠在模型类型中不是规定性的。这允许在自定义和探索方法在数据集上的使用时有很大的灵活性。

例如，我们可以想象在训练数据集的引导样本上拟合大量决策树，就像我们在 bagging 中所做的那样，然后测试一组不同的模型，以了解如何最好地组合来自这些树的预测。

*   **不同的集成成员**:在引导样本上训练的决策树。

或者，我们可以想象网格搜索单个机器学习模型的大量配置(这在机器学习项目中很常见)，并保持所有模型的匹配。然后，这些模型可以用作堆叠集合中的成员。

*   **不同的集成成员**:相同算法的交替配置。

我们可能还会看到“专家混合”技术适合堆叠方法。

专家混合(简称 MoE)是一种技术，它将问题明确地划分为子问题，并在每个子问题上训练一个模型，然后使用该模型来学习如何最好地权衡或组合专家的预测。

堆叠和专家混合之间的重要区别是 MoE 的明确分治方法，以及使用门控网络组合预测的更复杂方式。

然而，我们设想将一个输入特征空间划分为一个子空间网格，在每个子空间上训练一个模型，并使用一个元模型，该元模型从基本模型和原始输入样本中获取预测，并学习哪个基本模型对输入数据最有条件信任或最有权重。

*   **多样集成成员**:将输入特征空间划分为统一的子空间。

这可以进一步扩展到首先在每个子空间的众多模型类型中选择一个表现良好的模型类型，只保留每个子空间中表现最好的专家，然后学习如何最好地组合他们的预测。

最后，我们可能认为元模型是对基本模型的修正。我们可能会探索这个想法，并让多个元模型尝试纠正贡献成员的重叠或非重叠池以及堆叠在它们之上的额外模型层。这种更深层次的模型堆叠有时会在机器学习竞赛中使用，训练起来可能会变得复杂和具有挑战性，但在预测任务中可能会带来额外的好处，因为更好的模型技能远远超过了反思模型的能力。

我们可以看到，堆叠方法的通用性为实验和定制留下了很大的空间，来自增强和打包的想法可以直接结合在一起。

## 进一步阅读

如果您想更深入地了解这个主题，本节将提供更多资源。

### 相关教程

*   [用 Python 堆叠集成机器学习](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
*   [如何用 Keras 开发 Python 深度学习神经网络的堆叠集成](https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/)
*   [如何用 Python 从头开始实现堆叠泛化(堆叠)](https://machinelearningmastery.com/implementing-stacking-scratch-python/)
*   [如何用 Python 开发投票套装](https://machinelearningmastery.com/voting-ensembles-with-python/)
*   [如何在 Python 中开发超级学习者套装](https://machinelearningmastery.com/super-learner-ensemble-in-python/)

### 书

*   [使用集成方法的模式分类](https://amzn.to/2zxc0F7)，2010。
*   [集成方法](https://amzn.to/2XZzrjG)，2012。
*   [集成机器学习](https://amzn.to/2C7syo5)，2012。
*   [数据挖掘:实用机器学习工具与技术](https://amzn.to/2WehhJC)，2016。

## 摘要

在本教程中，您发现了机器学习集成的堆叠泛化方法的本质。

具体来说，您了解到:

*   机器学习的堆叠集成方法使用元模型来组合来自贡献成员的预测。
*   如何从堆叠方法中提取基本元素，以及像混合和超级合奏这样的流行扩展是如何关联的。
*   如何通过为方法的基本元素选择新的过程来设计新的堆栈扩展。

**你有什么问题吗？**
在下面的评论中提问，我会尽力回答。